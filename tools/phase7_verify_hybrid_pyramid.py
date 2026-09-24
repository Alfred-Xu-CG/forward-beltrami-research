"""Teacher reachability and joint VJP for F2-seed/F1-refinement P1 layer."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    HybridPatchSeedVertexP1Pyramid, exact_dyadic_p1_refine,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian, vector_rmse


def target(side: int, strength: float, device: torch.device) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((
        xx + strength * torch.sin(2 * math.pi * xx) * torch.sin(math.pi * yy),
        yy - .8 * strength * torch.sin(math.pi * xx) * torch.sin(2 * math.pi * yy),
    ), dim=-1)[None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--seed-side", type=int, default=17)
    parser.add_argument("--seed-cycles", type=int, default=4)
    parser.add_argument("--patch-cells", type=int, default=4)
    parser.add_argument("--strength", type=float, default=.08)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpoint-levels", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    layer = HybridPatchSeedVertexP1Pyramid(
        args.seed_side, args.side,
        patch_cells=args.patch_cells, seed_cycles=args.seed_cycles,
        checkpoint_levels=args.checkpoint_levels,
    ).to(device)
    seed_target = target(args.seed_side, args.strength, device)
    seed_identity = target(args.seed_side, 0., device)
    seed_span = .5 * args.patch_cells / (args.seed_side - 1)
    seed_raw = ((seed_target - seed_identity)[:, 1:-1, 1:-1] / seed_span)
    if float(seed_raw.abs().amax()) >= 1:
        raise ValueError("seed teacher exceeds latent range")
    seed = torch.atanh(seed_raw).requires_grad_()
    levels = []
    prior = seed_target
    for side in layer.level_sides:
        next_target = target(side, args.strength, device)
        base = exact_dyadic_p1_refine(prior)
        raw = ((next_target - base)[:, 1:-1, 1:-1] / (2 / (side - 1)))
        if float(raw.abs().amax()) >= 1:
            raise ValueError(f"level {side} teacher exceeds latent range")
        levels.append(torch.atanh(raw).requires_grad_())
        prior = next_target
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times_forward, times_vjp = [], []
    output = None
    gradients = None
    for _ in range(args.repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        output = layer(seed, levels)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_forward.append(time.perf_counter() - tick)
        loss = (output[..., 0] + .37 * output[..., 1]).mean()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        gradients = torch.autograd.grad(loss, (seed, *levels))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_vjp.append(time.perf_counter() - tick)
    assert output is not None and gradients is not None
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    gradient_maxima = [float(g.abs().amax()) for g in gradients]
    finite = [bool(torch.isfinite(g).all()) for g in gradients]
    if not all(finite) or not all(v > 0 for v in gradient_maxima):
        raise AssertionError("one or more latent groups lost finite nonzero VJP")
    min_j = minimum_jacobian(output)
    if min_j <= 0:
        raise AssertionError("invalid P1 output")
    print(json.dumps(dict(
        experiment="phase7_hybrid_f2_seed_f1_levels_teacher",
        seed_side=args.seed_side,
        seed_cycles=args.seed_cycles,
        patch_cells=args.patch_cells,
        side=args.side,
        control_vertices=args.side ** 2,
        control_faces=2 * (args.side - 1) ** 2,
        strength=args.strength,
        level_sides=layer.level_sides,
        repeats=args.repeats,
        batch=1,
        device=str(device),
        dtype="float64",
        checkpoint_levels=args.checkpoint_levels,
        target_minimum_jacobian=minimum_jacobian(prior),
        output_minimum_jacobian=min_j,
        vector_rmse=vector_rmse(output, prior),
        maximum_coordinate_error=float((output - prior).abs().amax()),
        gradient_maxima=gradient_maxima,
        median_forward_seconds=statistics.median(times_forward),
        median_vjp_seconds=statistics.median(times_vjp),
        peak_cuda_allocated_bytes=peak,
        torch_version=torch.__version__,
    ), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
