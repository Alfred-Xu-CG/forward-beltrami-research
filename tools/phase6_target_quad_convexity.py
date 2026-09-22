"""Audit a necessary convex-cell condition for the Phase VI A2+ decoder."""

from __future__ import annotations

import argparse
import json
import math

import torch

from phase6_train_multisample_image import make_dataset


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _sampled_target(side: int, coefficients: torch.Tensor, fine_cycles: int) -> torch.Tensor:
    line = torch.linspace(0, 1, side, dtype=torch.float64)
    y, x = torch.meshgrid(line, line, indexing="ij")
    ax, ay, af = coefficients.to(torch.float64).unbind(-1)
    low = torch.sin(2 * math.pi * x) * torch.sin(2 * math.pi * y)
    high = torch.sin(2 * math.pi * fine_cycles * x) * torch.sin(2 * math.pi * fine_cycles * y)
    return torch.stack((x[None] + ax[:, None, None] * low + af[:, None, None] * high,
                        y[None] + ay[:, None, None] * low + af[:, None, None] * high), dim=-1)


def _segment_distance(a: torch.Tensor, m: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    direction = b - a
    t = ((m - a) * direction).sum(dim=-1) / direction.square().sum(dim=-1)
    projected = a + t.clamp(0.0, 1.0)[..., None] * direction
    return (m - projected).norm(dim=-1)


def audit(side: int, family: str, count: int) -> dict:
    if side < 3 or side - 1 & side - 2:
        raise ValueError("side must have the form 2^L+1")
    _, _, _, coefficients = make_dataset(
        count, 512, 99317, return_coefficients=True, target_family=family
    )
    mapped = _sampled_target(side, coefficients, 8 if family == "base" else 32)
    levels = []
    step = side - 1
    while step >= 1:
        q00 = mapped[:, :-step:step, :-step:step]
        q10 = mapped[:, :-step:step, step::step]
        q11 = mapped[:, step::step, step::step]
        q01 = mapped[:, step::step, :-step:step]
        e0, e1, e2, e3 = q10 - q00, q11 - q10, q01 - q11, q00 - q01
        corner = torch.stack((_cross(e0, e1), _cross(e1, e2), _cross(e2, e3), _cross(e3, e0)), dim=-1)
        normalized = corner / (step / (side - 1)) ** 2
        bad_cells = (normalized <= 0).any(dim=-1)
        face0 = _cross(q10 - q00, q11 - q00) / (step / (side - 1)) ** 2
        face1 = _cross(q11 - q00, q01 - q00) / (step / (side - 1)) ** 2
        edge_alignment = None
        if step > 1:
            horizontal = _segment_distance(
                mapped[:, ::step, :-step:step],
                mapped[:, ::step, step // 2::step],
                mapped[:, ::step, step::step],
            )
            vertical = _segment_distance(
                mapped[:, :-step:step, ::step],
                mapped[:, step // 2::step, ::step],
                mapped[:, step::step, ::step],
            )
            distance = torch.cat((horizontal.flatten(1), vertical.flatten(1)), dim=1)
            edge_alignment = {
                "maximum_midpoint_to_segment_distance": distance.max().item(),
                "rms_midpoint_to_segment_distance": distance.square().mean().sqrt().item(),
                "maximum_by_sample": [float(v) for v in distance.max(dim=1).values.tolist()],
            }
        levels.append({
            "cells_per_side": (side - 1) // step,
            "cell_count_per_sample": bad_cells[0].numel(),
            "nonconvex_cells_total": int(bad_cells.sum().item()),
            "nonconvex_cells_by_sample": [int(v) for v in bad_cells.flatten(1).sum(dim=1).tolist()],
            "minimum_normalized_corner_cross": normalized.min().item(),
            "minimum_normalized_face_area": min(face0.min().item(), face1.min().item()),
            "edge_alignment": edge_alignment,
        })
        step //= 2
    return {"control_side": side, "control_vertices": side**2, "sample_count": count,
            "family": family, "coefficient_seed": 99317, "levels_coarse_to_fine": levels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--family", choices=("base", "high32"), required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(audit(args.side, args.family, args.count), sort_keys=True))


if __name__ == "__main__":
    main()
