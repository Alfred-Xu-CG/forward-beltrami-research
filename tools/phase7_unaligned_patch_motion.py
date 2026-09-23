"""Stress shifted F2 patch schedules on a target crossing patch seams."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
from torch.utils.checkpoint import checkpoint

from qcopt.neural_bijection.dense import SafePatchFieldPass, StaggeredPatchP1Layer
from phase7_multiscale_fiber_reachability import minimum_jacobian, vector_rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--patch-cells", type=int, default=64)
    parser.add_argument("--amplitude", type=float, default=0.006)
    parser.add_argument("--minimum-jacobian", type=float, default=None)
    parser.add_argument("--offset-fraction", type=float, default=0.5)
    parser.add_argument("--cycles", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--overlap-policy", choices=["unique", "all"], default="unique")
    parser.add_argument("--fit-cycles", type=int, default=0)
    parser.add_argument("--fit-steps", type=int, default=0)
    parser.add_argument("--fit-lr", type=float, default=0.01)
    parser.add_argument("--stage-loss-weight", type=float, default=0.0)
    parser.add_argument("--diagnose-fit", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.side - 1) % args.patch_cells or args.patch_cells % 2:
        raise ValueError("even patch_cells must divide side-1")
    if not 0 <= args.offset_fraction < 1:
        raise ValueError("offset_fraction must lie in [0,1)")
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
    target = identity + torch.stack((displacement, displacement), dim=-1)[None]
    target_min_j = minimum_jacobian(target)
    if target_min_j <= 0:
        raise ValueError("target has nonpositive P1 face")

    single = SafePatchFieldPass(side, args.patch_cells, raw_span=0.5,
                                safety_fraction=0.85,
                                minimum_jacobian=args.minimum_jacobian).to(device)
    raw_scale = 0.5 * width
    selected = (target - identity).reshape(1, -1, 2)[:, single.interior_ids]
    single_latent = torch.atanh((selected / raw_scale).clamp(-0.999999, 0.999999))
    with torch.no_grad():
        single_output = single(identity, single_latent)
    single_fixed_mask = torch.ones(side * side, dtype=torch.bool, device=device)
    single_fixed_mask[single.interior_ids] = False
    fixed_target_error = (target - identity).reshape(-1, 2)[single_fixed_mask].norm(dim=-1)

    staggered = StaggeredPatchP1Layer(side, patch_cells=args.patch_cells,
                                      raw_span=0.5, safety_fraction=0.85,
                                      minimum_jacobian=args.minimum_jacobian).to(device)
    owner = torch.full((side * side,), -1, device=device, dtype=torch.int32)
    masks = []
    for index, patch_pass in enumerate(staggered.passes):
        ids = patch_pass.interior_ids
        unowned = owner[ids] < 0
        owner[ids[unowned]] = index
        mask = unowned if args.overlap_policy == "unique" else torch.ones_like(unowned)
        masks.append(mask)
    interior_ids = torch.arange(side * side, device=device).reshape(side, side)[1:-1, 1:-1]
    if not bool((owner[interior_ids.reshape(-1)] >= 0).all()):
        raise AssertionError("staggered passes did not cover all interior vertices")

    results = []
    for cycles in args.cycles:
        with torch.no_grad():
            current = identity
            teachers = []
            for cycle in range(cycles):
                intermediate = identity + ((cycle + 1) / cycles) * (target - identity)
                for patch_pass, mask in zip(staggered.passes, masks):
                    ids = patch_pass.interior_ids
                    desired = ((intermediate.reshape(1, -1, 2)[:, ids]
                                - current.reshape(1, -1, 2)[:, ids]) / raw_scale)
                    compact = torch.zeros_like(desired)
                    compact[:, mask] = torch.atanh(desired[:, mask].clamp(-0.999999, 0.999999))
                    teachers.append(compact.cpu())
                    current = patch_pass(current, compact)
            result = {
                "cycles": cycles,
                "total_patch_passes": 4 * cycles,
                "vertex_vector_rmse": vector_rmse(current, target),
                "maximum_coordinate_error": float((current - target).abs().amax()),
                "minimum_jacobian": minimum_jacobian(current),
            }
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cotangent = torch.randn(target.shape, device=device)
        inputs = [field.to(device).requires_grad_() for field in teachers]
        times = []
        for _ in range(3):
            for field in inputs:
                field.grad = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            output = identity
            for index, field in enumerate(inputs):
                patch_pass = staggered.passes[index % 4]
                output = checkpoint(patch_pass, output, field, use_reentrant=False)
            (output * cotangent).mean().backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - tick)
        result.update({
            "median_forward_and_vjp_seconds": statistics.median(times),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "all_vjps_finite_nonzero": all(
                bool(torch.isfinite(field.grad).all() and field.grad.abs().amax() > 0)
                for field in inputs
            ),
        })
        results.append(result)
        del inputs, teachers, current

    fit_result = None
    if args.fit_steps:
        if args.fit_cycles < 1:
            raise ValueError("fit-cycles must be positive when fit-steps is positive")
        fields = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((1, patch_pass.interior_ids.numel(), 2),
                                            device=device))
            for _ in range(args.fit_cycles) for patch_pass in staggered.passes
        ])
        optimizer = torch.optim.Adam(fields.parameters(), lr=args.fit_lr, eps=1e-12)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.fit_steps, eta_min=args.fit_lr * 0.0001,
        )
        times = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.fit_steps):
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = identity
            auxiliary_loss = torch.zeros((), device=device)
            for index, field in enumerate(fields):
                patch_pass = staggered.passes[index % 4]
                output = checkpoint(patch_pass, output, field, use_reentrant=False)
                if (index + 1) % 4 == 0 and index + 1 < len(fields):
                    cycle = (index + 1) // 4
                    intermediate = identity + cycle / args.fit_cycles * (target - identity)
                    auxiliary_loss = auxiliary_loss + (
                        output - intermediate
                    ).square().sum(dim=-1).mean()
            loss = 1e6 * (output - target).square().sum(dim=-1).mean()
            loss = loss + args.stage_loss_weight * 1e6 * auxiliary_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - tick)
        with torch.no_grad():
            output = identity
            stage_min_j = []
            for index, field in enumerate(fields):
                output = staggered.passes[index % 4](output, field)
                if args.diagnose_fit:
                    stage_min_j.append(minimum_jacobian(output))
        diagnostic = None
        if args.diagnose_fit:
            output_double_min_j = minimum_jacobian(output.double())
            with torch.no_grad():
                staggered.double()
                output64 = identity.double()
                for index, field in enumerate(fields):
                    output64 = staggered.passes[index % 4](output64, field.double())
            diagnostic = {
                "stage_minimum_jacobians_float32": stage_min_j,
                "final_float32_vertices_checked_in_float64_min_j": output_double_min_j,
                "same_latents_float64_decoder_min_j": minimum_jacobian(output64),
                "same_latents_decoder_output_max_coordinate_disagreement": float(
                    (output.double() - output64).abs().amax()
                ),
            }
        fit_result = {
            "cycles": args.fit_cycles,
            "steps": args.fit_steps,
            "learning_rate": args.fit_lr,
            "stage_loss_weight": args.stage_loss_weight,
            "latent_elements": sum(field.numel() for field in fields),
            "final_vertex_vector_rmse": vector_rmse(output, target),
            "final_maximum_coordinate_error": float((output - target).abs().amax()),
            "final_minimum_jacobian": minimum_jacobian(output),
            "median_training_step_seconds": statistics.median(
                times[10:] if len(times) > 10 else times
            ),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "diagnostic": diagnostic,
        }

    print(json.dumps({
        "method": "phase7_unaligned_patch_motion",
        "side": side,
        "vertices": side * side,
        "faces": 2 * (side - 1) ** 2,
        "patch_cells": args.patch_cells,
        "offset_fraction": args.offset_fraction,
        "patch_origin": [x0, y0],
        "amplitude": args.amplitude,
        "minimum_jacobian_parameter": args.minimum_jacobian,
        "target_minimum_jacobian": target_min_j,
        "single_pass": {
            "vertex_vector_rmse": vector_rmse(single_output, target),
            "maximum_coordinate_error": float((single_output - target).abs().amax()),
            "fixed_vertex_error_max": float(fixed_target_error.amax()),
            "minimum_jacobian": minimum_jacobian(single_output),
        },
        "overlap_policy": args.overlap_policy,
        "staggered": results,
        "direct_latent_fit": fit_result,
        "active_vertices_by_pass": [int(mask.sum()) for mask in masks],
        "device": str(device),
        "torch_version": torch.__version__,
        "dtype": "float32",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
