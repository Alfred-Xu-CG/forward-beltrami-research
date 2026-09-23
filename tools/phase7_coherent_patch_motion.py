"""Compare F2 one-patch coherent motion with repeated F1 colored updates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
from torch.utils.checkpoint import checkpoint

from qcopt.neural_bijection.dense import (
    SafeColoredVertexRelaxation,
    SafePatchFieldPass,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian, vector_rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--patch-cells", type=int, default=64)
    parser.add_argument("--amplitude", type=float, default=0.006)
    parser.add_argument("--f1-passes", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-steps", type=int, default=0)
    parser.add_argument("--fit-lr", type=float, default=0.01)
    args = parser.parse_args()
    if (args.side - 1) % args.patch_cells:
        raise ValueError("patch size must divide side-1")
    device = torch.device(args.device)
    side = args.side
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    patch_width = args.patch_cells / (side - 1)
    x0 = math.floor(0.25 / patch_width) * patch_width
    y0 = math.floor(0.25 / patch_width) * patch_width
    tx, ty = (xx - x0) / patch_width, (yy - y0) / patch_width
    wx = torch.where((tx >= 0) & (tx <= 1),
                     torch.sin(math.pi * tx).square(), 0)
    wy = torch.where((ty >= 0) & (ty <= 1),
                     torch.sin(math.pi * ty).square(), 0)
    displacement = args.amplitude * wx * wy
    target = identity + torch.stack((displacement, displacement), dim=-1)[None]
    if minimum_jacobian(target) <= 0:
        raise ValueError("independent target has nonpositive P1 faces")

    patch = SafePatchFieldPass(
        side, args.patch_cells, raw_span=0.5,
        safety_fraction=0.85, minimum_jacobian=None,
    ).to(device)
    selected = (target - identity).reshape(1, -1, 2)[:, patch.interior_ids]
    patch_raw_scale = 0.5 * args.patch_cells / (side - 1)
    patch_teacher = torch.atanh(selected / patch_raw_scale)
    with torch.no_grad():
        patch_output = patch(identity, patch_teacher)

    colored = SafeColoredVertexRelaxation(
        side, safety_fraction=0.85, motion_mode="radial", raw_span=2.0,
    ).to(device)
    colored_scale = 2.0 / (side - 1)
    colored_results = []
    teacher_sets = {}
    for count in args.f1_passes:
        current = identity
        teachers = []
        with torch.no_grad():
            for step in range(1, count + 1):
                intermediate = identity + step / count * (target - identity)
                desired = ((intermediate[:, 1:-1, 1:-1]
                            - current[:, 1:-1, 1:-1]) / colored_scale)
                # Clamping is only for the oracle construction when a one-pass
                # proposal exceeds the bounded raw motion. It is not part of
                # the decoder and does not affect its topology certificate.
                logits = torch.atanh(desired.clamp(-0.999999, 0.999999))
                teachers.append(logits)
                current = colored(current, logits)
        error = vector_rmse(current, target)
        colored_results.append({
            "passes": count,
            "vertex_vector_rmse": error,
            "maximum_coordinate_error": float((current - target).abs().amax()),
            "minimum_jacobian": minimum_jacobian(current),
        })
        teacher_sets[count] = [z.detach().cpu() for z in teachers]
    del current, teachers, logits, intermediate, desired

    def timed_vjp(name: str, inputs: list[torch.Tensor], run) -> dict:
        generator = torch.Generator(device=device).manual_seed(443)
        cotangent = torch.randn(target.shape, generator=generator, device=device)
        times = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(3):
            for z in inputs:
                z.grad = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            output = run(inputs)
            (output * cotangent).mean().backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - tick)
        return {
            "name": name,
            "median_forward_and_vjp_seconds": statistics.median(times),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "all_vjps_finite_nonzero": all(
                bool(torch.isfinite(z.grad).all() and z.grad.abs().amax() > 0)
                for z in inputs
            ),
        }

    patch_input = patch_teacher.detach().clone().requires_grad_()
    patch_vjp = timed_vjp("F2_one_patch_pass", [patch_input],
                          lambda fields: patch(identity, fields[0]))
    def run_colored(fields: list[torch.Tensor]) -> torch.Tensor:
        current = identity
        for z in fields:
            current = checkpoint(colored, current, z, use_reentrant=False)
        return current
    colored_vjps = []
    for count in args.f1_passes:
        if count not in (4, 8, 16):
            continue
        inputs = [z.to(device).requires_grad_() for z in teacher_sets[count]]
        colored_vjps.append(timed_vjp(
            f"F1_{count}_checkpointed_passes", inputs, run_colored,
        ))
        del inputs
    fit_result = None
    if args.fit_steps:
        fit_latent = torch.nn.Parameter(torch.zeros_like(patch_teacher))
        optimizer = torch.optim.Adam([fit_latent], lr=args.fit_lr, eps=1e-12)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.fit_steps, eta_min=args.fit_lr * 0.0001,
        )
        fit_times = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.fit_steps):
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            fitted = patch(identity, fit_latent)
            loss = 1e6 * (fitted - target).square().sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fit_times.append(time.perf_counter() - tick)
        with torch.no_grad():
            fitted = patch(identity, fit_latent)
        fit_result = {
            "steps": args.fit_steps,
            "learning_rate": args.fit_lr,
            "final_vertex_vector_rmse": vector_rmse(fitted, target),
            "final_minimum_jacobian": minimum_jacobian(fitted),
            "median_training_step_seconds": statistics.median(
                fit_times[10:] if len(fit_times) > 10 else fit_times,
            ),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
    print(json.dumps({
        "method": "phase7_coherent_patch_motion_compare",
        "side": side,
        "vertices": side ** 2,
        "faces": 2 * (side - 1) ** 2,
        "patch_cells": args.patch_cells,
        "patch_origin": [x0, y0],
        "amplitude": args.amplitude,
        "target_minimum_jacobian": minimum_jacobian(target),
        "device": str(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "F2_one_pass": {
            "vertex_vector_rmse": vector_rmse(patch_output, target),
            "maximum_coordinate_error": float((patch_output - target).abs().amax()),
            "minimum_jacobian": minimum_jacobian(patch_output),
            "active_latent_vectors": patch.interior_ids.numel(),
            "teacher_logit_max_abs": float(patch_teacher.abs().amax()),
        },
        "F1_oracle_pass_counts": colored_results,
        "VJP": [patch_vjp, *colored_vjps],
        "F2_direct_latent_fit": fit_result,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
