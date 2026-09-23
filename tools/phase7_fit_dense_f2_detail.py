"""Dense two-component F2 patch latent fit with unique-owner staggered passes."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    MultiscaleMonotoneFiberP1Layer,
    StaggeredPatchP1Layer,
)
from phase7_multiscale_fiber_reachability import (
    exact_teacher_latent,
    minimum_jacobian,
    resize,
    teacher_map,
    vector_rmse,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--base-side", type=int, default=257)
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--detail-amplitude", type=float, default=0.00002)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    side = args.side
    base_target = teacher_map(side, "shear", device)
    high_target = teacher_map(side, "high128", device)
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    detail = args.detail_amplitude / 0.0001 * (
        high_target[..., 0] - axis[None, None]
    )
    target = base_target + torch.stack((detail, detail), dim=-1)
    if minimum_jacobian(target) <= 0:
        raise ValueError("independent target is not P1 oriented")
    base_layer = MultiscaleMonotoneFiberP1Layer(side).to(device)
    base_latent = resize(
        exact_teacher_latent(base_target, 0.05, 8.0),
        args.base_side - 2, args.base_side - 1,
    )
    with torch.no_grad():
        base = base_layer([base_latent])
    layer = StaggeredPatchP1Layer(
        side, patch_cells=args.patch_cells, raw_span=0.5,
        safety_fraction=0.85, minimum_jacobian=None,
    ).to(device)
    owner = torch.full((side * side,), -1, device=device, dtype=torch.int32)
    masks = []
    for pass_number, patch_pass in enumerate(layer.passes):
        ids = patch_pass.interior_ids
        mask = owner[ids] < 0
        owner[ids[mask]] = pass_number
        masks.append(mask)
    interior = torch.arange(side * side, device=device).reshape(side, side)[1:-1, 1:-1]
    if not bool((owner[interior.reshape(-1)] >= 0).all()):
        raise AssertionError("the staggered patch schedule must cover all interior vertices")

    def compact_fields(full: torch.Tensor) -> tuple[torch.Tensor, ...]:
        flat = full.reshape(full.shape[0], -1, 2)
        return tuple(flat[:, patch_pass.latent_ids.reshape(-1)] * mask[None, :, None]
                     for patch_pass, mask in zip(layer.passes, masks))

    with torch.no_grad():
        raw_scale = 0.5 * args.patch_cells / (side - 1)
        teacher = torch.zeros((1, side - 2, side - 2, 2), device=device)
        teacher_flat = teacher.reshape(1, -1, 2)
        current = base
        for patch_pass, mask in zip(layer.passes, masks):
            ids = patch_pass.interior_ids
            desired = ((target.reshape(1, -1, 2)[:, ids]
                        - current.reshape(1, -1, 2)[:, ids]) / raw_scale)
            if float(desired[:, mask].abs().amax()) >= 1:
                raise ValueError("target exceeds one patch pass's raw span")
            compact = torch.zeros_like(desired)
            compact[:, mask] = torch.atanh(desired[:, mask])
            teacher_flat[:, patch_pass.latent_ids.reshape(-1)[mask]] = compact[:, mask]
            current = patch_pass(current, compact)
        teacher_output = current

    latent = torch.nn.Parameter(torch.zeros_like(teacher))
    optimizer = torch.optim.Adam([latent], lr=args.lr, eps=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.0001,
    )
    times, best_rmse, best_step = [], float("inf"), None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = layer(base, compact_fields(latent))
        loss = 1e6 * (output - target).square().sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - tick)
        rmse = float((loss.detach() / 1e6).sqrt())
        if rmse < best_rmse:
            best_rmse, best_step = rmse, step
    with torch.no_grad():
        final = layer(base, compact_fields(latent))
    generator = torch.Generator(device=device).manual_seed(91018)
    cotangent = torch.randn(target.shape, generator=generator, device=device)
    random_vjp = torch.autograd.grad(
        (layer(base, compact_fields(latent)) * cotangent).mean(), latent,
    )[0]
    print(json.dumps({
        "method": "phase7_dense_two_component_f2_detail_fit",
        "side": side,
        "base_side": args.base_side,
        "patch_cells": args.patch_cells,
        "vertices": side ** 2,
        "faces": 2 * (side - 1) ** 2,
        "base_latent_elements": base_latent.numel(),
        "fine_latent_elements": latent.numel(),
        "active_vertices_by_pass": [int(m.sum()) for m in masks],
        "steps": args.steps,
        "learning_rate": args.lr,
        "detail_amplitude": args.detail_amplitude,
        "target_minimum_jacobian": minimum_jacobian(target),
        "device": str(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "initial_vertex_vector_rmse": vector_rmse(base, target),
        "teacher_vertex_vector_rmse": vector_rmse(teacher_output, target),
        "teacher_minimum_jacobian": minimum_jacobian(teacher_output),
        "best_training_step": best_step,
        "best_training_vertex_vector_rmse": best_rmse,
        "final_vertex_vector_rmse": vector_rmse(final, target),
        "final_minimum_jacobian": minimum_jacobian(final),
        "random_cotangent_vjp_finite": bool(torch.isfinite(random_vjp).all()),
        "random_cotangent_vjp_max_abs": float(random_vjp.abs().amax()),
        "median_training_step_seconds": statistics.median(
            times[10:] if len(times) > 10 else times,
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
