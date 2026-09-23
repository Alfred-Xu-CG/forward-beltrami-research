"""Fit one tied dense latent field through repeated safe staggered patch passes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    ResidualStaggeredPatchP1Layer, TiedStaggeredPatchP1Layer,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian, vector_rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--patch-cells", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--parameterization", choices=("tied", "residual"), default="tied")
    parser.add_argument("--amplitude", type=float, default=0.006)
    parser.add_argument("--high-amplitude", type=float, default=0.0)
    parser.add_argument("--offset-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-jacobian", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.side - 1) % args.patch_cells:
        raise ValueError("patch_cells must divide side-1")
    device = torch.device(args.device)
    side = args.side
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    width = args.patch_cells / (side - 1)
    x0 = 0.25 + args.offset_fraction * width
    y0 = x0
    tx, ty = (xx - x0) / width, (yy - y0) / width
    wx = torch.where((tx >= 0) & (tx <= 1), torch.sin(math.pi * tx).square(), 0)
    wy = torch.where((ty >= 0) & (ty <= 1), torch.sin(math.pi * ty).square(), 0)
    displacement = args.amplitude * wx * wy
    if args.high_amplitude:
        axis64 = torch.arange(side, device=device, dtype=torch.float64) / (side - 1)
        yy64, xx64 = torch.meshgrid(axis64, axis64, indexing="ij")
        high = (args.high_amplitude * torch.sin(256 * math.pi * xx64)
                * torch.sin(256 * math.pi * yy64)
                * torch.sin(math.pi * xx64).square())
        displacement = displacement + high.float()
    target = identity + torch.stack((displacement, displacement), dim=-1)[None]
    layer_class = (TiedStaggeredPatchP1Layer if args.parameterization == "tied"
                   else ResidualStaggeredPatchP1Layer)
    layer = layer_class(
        side, args.patch_cells, cycles=args.cycles,
        raw_span=0.5, safety_fraction=0.85,
        minimum_jacobian=args.minimum_jacobian,
    ).to(device)
    span = 0.5 * width
    teacher = torch.atanh(((target - identity)[:, 1:-1, 1:-1] / span)
                           .clamp(-0.999999, 0.999999))
    with torch.no_grad():
        teacher_output = layer(identity, teacher)
    latent = torch.nn.Parameter(torch.zeros_like(teacher))
    optimizer = torch.optim.Adam([latent], lr=args.lr, eps=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.0001,
    )
    times = []
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = layer(identity, latent)
        loss = 1e6 * (output - target).square().sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - tick)
        if step in (1, args.steps) or step % max(1, args.steps // 5) == 0:
            with torch.no_grad():
                history.append({
                    "step": step,
                    "vertex_vector_rmse": vector_rmse(output, target),
                    "minimum_jacobian": minimum_jacobian(output),
                })
    with torch.no_grad():
        final = layer(identity, latent)
    generator = torch.Generator(device=device).manual_seed(1128)
    cotangent = torch.randn(target.shape, generator=generator, device=device)
    gradient = torch.autograd.grad((layer(identity, latent) * cotangent).mean(),
                                   latent)[0]
    print(json.dumps({
        "method": "phase7_shared_staggered_patch_motion",
        "parameterization": args.parameterization,
        "side": side,
        "vertices": side * side,
        "faces": 2 * (side - 1) ** 2,
        "patch_cells": args.patch_cells,
        "cycles": args.cycles,
        "pass_count": 4 * args.cycles,
        "amplitude": args.amplitude,
        "high_amplitude": args.high_amplitude,
        "offset_fraction": args.offset_fraction,
        "minimum_jacobian_parameter": args.minimum_jacobian,
        "latent_elements": latent.numel(),
        "target_minimum_jacobian": minimum_jacobian(target),
        "teacher_vertex_vector_rmse": vector_rmse(teacher_output, target),
        "teacher_minimum_jacobian": minimum_jacobian(teacher_output),
        "steps": args.steps,
        "learning_rate": args.lr,
        "history": history,
        "final_vertex_vector_rmse": vector_rmse(final, target),
        "final_maximum_coordinate_error": float((final - target).abs().amax()),
        "final_minimum_jacobian": minimum_jacobian(final),
        "random_cotangent_vjp_finite_nonzero": bool(
            torch.isfinite(gradient).all() and gradient.abs().amax() > 0
        ),
        "median_training_step_seconds": statistics.median(
            times[10:] if len(times) > 10 else times
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "dtype": "float32",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
