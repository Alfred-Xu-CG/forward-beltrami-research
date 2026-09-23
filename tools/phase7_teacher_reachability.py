"""Diagnostic inverse latents for known maps; never an image-trained result."""

from __future__ import annotations

import argparse
import json
import math

import torch

from qcopt.neural_bijection.dense import exact_dyadic_p1_refine
from qcopt.neural_bijection.dense.colored_vertex_relaxation import SafeColoredVertexRelaxation


def target_map(side: int, device: torch.device, dtype: torch.dtype, kind: str) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    if kind == "local_swirl":
        dx, dy = xx - 0.57, yy - 0.43
        envelope = torch.sin(math.pi * xx) * torch.sin(math.pi * yy)
        profile = 0.25 * envelope * torch.exp(-(dx * dx + dy * dy) / 0.12**2)
        mapped = torch.stack((xx - profile * dy, yy + profile * dx), dim=-1)
    else:
        low = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
        if kind == "base":
            cycles, ax, ay, af = 8, 0.03, 0.05, 0.003
        elif kind == "high32":
            cycles, ax, ay, af = 32, 0.015, 0.025, 0.0025
        elif kind == "high64":
            cycles, ax, ay, af = 64, 0.015, 0.025, 0.00125
        else:
            raise ValueError("unknown target kind")
        high = torch.sin(2 * math.pi * cycles * xx) * torch.sin(2 * math.pi * cycles * yy)
        mapped = torch.stack((xx + ax * low + af * high, yy + ay * low + af * high), dim=-1)
    boundary = torch.zeros((side, side), dtype=torch.bool, device=device)
    boundary[0] = boundary[-1] = True
    boundary[:, 0] = boundary[:, -1] = True
    return torch.where(boundary[..., None], torch.stack((xx, yy), dim=-1), mapped)[None]


def area_stats(mapped: torch.Tensor) -> tuple[float, int]:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]
    def cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    areas = torch.stack((cross(b - a, c - a), cross(c - a, d - a)))
    return float(areas.amin() * (mapped.shape[1] - 1) ** 2), int((areas <= 0).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--seed-side", type=int, default=5)
    parser.add_argument("--target-kind", choices=("base", "high32", "high64", "local_swirl"), default="high32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--minimum-jacobian", type=float, default=0.05)
    parser.add_argument("--raw-span", type=float, default=2.0)
    args = parser.parse_args()
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    side = args.seed_side
    current = target_map(side, device, dtype, args.target_kind)
    # The initial seed operation is explicitly tested rather than assumed.
    axis = torch.arange(side, device=device, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    alpha = args.raw_span
    seed_delta = current - identity
    seed_max_ratio = float(seed_delta.abs().amax() / (alpha / (side - 1)))
    if seed_max_ratio >= 1:
        raise RuntimeError("seed target needs more than one raw-span pass")
    seed_logits = torch.atanh(seed_delta[:, 1:-1, 1:-1] / (alpha / (side - 1)))
    seed_layer = SafeColoredVertexRelaxation(side, motion_mode="radial", raw_span=alpha, safety_fraction=0.85).to(device)
    seed_floor = identity.new_full((1,), args.minimum_jacobian / (side - 1) ** 2)
    represented = seed_layer(identity, seed_logits, area_floor=seed_floor)
    steps = [{
        "side": side,
        "maximum_target_residual": float((represented - current).abs().amax()),
        "maximum_raw_span_ratio": seed_max_ratio,
        "target_minimum_jacobian": area_stats(current)[0],
        "represented_minimum_jacobian": area_stats(represented)[0],
        "represented_nonpositive_faces": area_stats(represented)[1],
    }]
    current = represented
    while side < args.side:
        side = 2 * side - 1
        target = target_map(side, device, dtype, args.target_kind)
        base = exact_dyadic_p1_refine(current)
        delta = target - base
        # Reused even/even vertices must not move in a refinement pass.
        new_mask = torch.ones((side, side), dtype=torch.bool, device=device)
        new_mask[::2, ::2] = False
        new_mask[0] = False
        new_mask[-1] = False
        new_mask[:, 0] = False
        new_mask[:, -1] = False
        requested = delta[:, new_mask]
        raw_limit = alpha / (side - 1)
        max_ratio = float(requested.abs().amax() / raw_limit)
        logits = torch.zeros((1, side - 2, side - 2, 2), device=device, dtype=dtype)
        safe = (requested / raw_limit).clamp(-1 + 1e-7, 1 - 1e-7)
        logit_grid = torch.zeros_like(base)
        logit_grid[:, new_mask] = torch.atanh(safe)
        logits[:] = logit_grid[:, 1:-1, 1:-1]
        layer = SafeColoredVertexRelaxation(side, motion_mode="radial", raw_span=alpha, safety_fraction=0.85).to(device)
        floor = base.new_full((1,), args.minimum_jacobian / (side - 1) ** 2)
        current = layer(base, logits, area_floor=floor)
        target_min, _ = area_stats(target)
        actual_min, nonpositive = area_stats(current)
        steps.append({
            "side": side,
            "maximum_target_residual": float((current - target).abs().amax()),
            "vertex_rmse": float(((current - target).square().sum(dim=-1).mean()).sqrt()),
            "maximum_raw_span_ratio": max_ratio,
            "target_minimum_jacobian": target_min,
            "represented_minimum_jacobian": actual_min,
            "represented_nonpositive_faces": nonpositive,
        })
    if side != args.side:
        raise ValueError("side is not reachable from seed_side by dyadic refinement")
    print(json.dumps({
        "method": "phase7_teacher_inverse_latent_diagnostic",
        "target_kind": args.target_kind,
        "side": args.side,
        "device": str(device),
        "dtype": args.dtype,
        "minimum_jacobian": args.minimum_jacobian,
        "raw_span": args.raw_span,
        "steps": steps,
        "note": "Known target map supplies inverse latent; not image-to-latent training.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
