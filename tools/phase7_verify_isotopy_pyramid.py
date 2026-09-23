"""Verify one safe P1 update per dyadic level on a non-patch target."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    CertifiedForwardP1Pyramid, ForwardP1Pyramid, ResidualPatchP1Pyramid,
    certify_p1_or_identity, exact_dyadic_p1_refine,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian, vector_rmse


def target(side: int, dtype: torch.dtype, device: torch.device,
           strength: float) -> torch.Tensor:
    axis = torch.arange(side, dtype=dtype, device=device) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    ux = strength * torch.sin(2 * math.pi * xx) * torch.sin(math.pi * yy)
    uy = (-0.8 * strength * torch.sin(math.pi * xx) * torch.sin(2 * math.pi * yy))
    return torch.stack((xx + ux, yy + uy), dim=-1)[None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--seed-side", type=int, default=17)
    parser.add_argument("--seed-passes", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.03)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--checkpoint-passes", action="store_true")
    parser.add_argument("--certified", action="store_true")
    parser.add_argument("--mechanism", choices=("vertex", "patch"), default="vertex")
    parser.add_argument("--patch-cells", type=int, default=4)
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    if args.mechanism == "vertex":
        layer_type = CertifiedForwardP1Pyramid if args.certified else ForwardP1Pyramid
        layer = layer_type(
            args.seed_side, args.side, seed_passes=args.seed_passes,
            safety_fraction=0.85, raw_span=2.0, minimum_jacobian=0.05,
            checkpoint_passes=args.checkpoint_passes,
        ).to(device)
        span = lambda n: 2 / (n - 1)
    else:
        if args.checkpoint_passes:
            raise ValueError("patch mechanism does not implement checkpoint passes")
        layer = ResidualPatchP1Pyramid(
            args.seed_side, args.side, patch_cells=args.patch_cells,
            seed_passes=args.seed_passes, minimum_jacobian=0.05,
        ).to(device)
        span = lambda n: 0.5 * args.patch_cells / (n - 1)
    seed_target = target(args.seed_side, dtype, device, args.strength)
    identity = target(args.seed_side, dtype, device, 0.0)
    seed_latents = []
    for step in range(args.seed_passes):
        before = identity + (step / args.seed_passes) * (seed_target - identity)
        after = identity + ((step + 1) / args.seed_passes) * (seed_target - identity)
        raw = ((after - before)[:, 1:-1, 1:-1]
               / span(args.seed_side))
        seed_latents.append(torch.atanh(raw))
    level_latents = []
    prior_target = seed_target
    for side in layer.level_sides:
        next_target = target(side, dtype, device, args.strength)
        base = exact_dyadic_p1_refine(prior_target)
        raw = ((next_target - base)[:, 1:-1, 1:-1] / span(side))
        level_latents.append(torch.atanh(raw))
        prior_target = next_target
    cotangent = torch.randn(prior_target.shape, dtype=dtype, device=device,
                             generator=torch.Generator(device=device).manual_seed(713))
    all_latents = [z.detach().requires_grad_() for z in seed_latents + level_latents]
    def run() -> torch.Tensor:
        value = layer(all_latents[:args.seed_passes], all_latents[args.seed_passes:])
        if args.mechanism == "patch" and args.certified:
            value, _ = certify_p1_or_identity(
                value, target(args.side, dtype, device, 0.0),
            )
        return value
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    gradient_maxima = []
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = run()
        gradients = torch.autograd.grad((output * cotangent).mean(), all_latents)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
        gradient_maxima = [float(g.abs().amax()) for g in gradients]
    print(json.dumps({
        "method": "phase7_isotopy_pyramid_teacher",
        "mechanism": args.mechanism,
        "patch_cells": args.patch_cells if args.mechanism == "patch" else None,
        "seed_side": args.seed_side,
        "seed_passes": args.seed_passes,
        "side": args.side,
        "vertices": args.side ** 2,
        "faces": 2 * (args.side - 1) ** 2,
        "level_sides": layer.level_sides,
        "strength": args.strength,
        "checkpoint_passes": args.checkpoint_passes,
        "certified": args.certified,
        "dtype": args.dtype,
        "device": str(device),
        "torch_version": torch.__version__,
        "target_minimum_jacobian": minimum_jacobian(prior_target),
        "output_minimum_jacobian": minimum_jacobian(output),
        "vertex_rmse": vector_rmse(output, prior_target),
        "max_abs_error": float((output - prior_target).abs().amax()),
        "gradient_maxima": gradient_maxima,
        "median_forward_and_vjp_seconds": statistics.median(times),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
