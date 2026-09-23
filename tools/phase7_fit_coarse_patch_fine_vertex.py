"""Train coarse residual F2 then fine F1 on one shifted smooth P1 target."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    ResidualStaggeredPatchP1Layer,
    SafeColoredVertexRelaxation,
    exact_dyadic_p1_refine,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian, vector_rmse


def target_map(side: int, *, amplitude: float, offset_fraction: float,
               width: float, high_amplitude: float,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    origin = 0.25 + offset_fraction * width
    tx, ty = (xx - origin) / width, (yy - origin) / width
    wx = torch.where((tx >= 0) & (tx <= 1), torch.sin(math.pi * tx).square(), 0)
    wy = torch.where((ty >= 0) & (ty <= 1), torch.sin(math.pi * ty).square(), 0)
    displacement = amplitude * wx * wy
    if high_amplitude:
        axis64 = torch.arange(side, device=device, dtype=torch.float64) / (side - 1)
        yy64, xx64 = torch.meshgrid(axis64, axis64, indexing="ij")
        high = (high_amplitude * torch.sin(256 * math.pi * xx64)
                * torch.sin(256 * math.pi * yy64)
                * torch.sin(math.pi * xx64).square())
        displacement = displacement + high.float()
    return identity, identity + torch.stack((displacement, displacement), dim=-1)[None]


def fit(layer, base: torch.Tensor, target: torch.Tensor, latent: torch.nn.Parameter,
        *, steps: int, lr: float, device: torch.device) -> tuple[torch.Tensor, dict]:
    optimizer = torch.optim.Adam([latent], lr=lr, eps=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.0001,
    )
    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(steps):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = layer(base, latent)
        loss = 1e6 * (output - target).square().sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - tick)
    with torch.no_grad():
        output = layer(base, latent)
    return output, {
        "steps": steps,
        "learning_rate": lr,
        "latent_elements": latent.numel(),
        "vertex_vector_rmse": vector_rmse(output, target),
        "maximum_coordinate_error": float((output - target).abs().amax()),
        "minimum_jacobian": minimum_jacobian(output),
        "median_training_step_seconds": statistics.median(
            times[10:] if len(times) > 10 else times
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-side", type=int, default=257)
    parser.add_argument("--fine-side", type=int, default=1025)
    parser.add_argument("--coarse-patch-cells", type=int, default=16)
    parser.add_argument("--coarse-cycles", type=int, default=2)
    parser.add_argument("--amplitude", type=float, default=0.006)
    parser.add_argument("--high-amplitude", type=float, default=0.0)
    parser.add_argument("--offset-fraction", type=float, default=0.5)
    parser.add_argument("--coarse-steps", type=int, default=600)
    parser.add_argument("--fine-steps", type=int, default=400)
    parser.add_argument("--coarse-lr", type=float, default=0.003)
    parser.add_argument("--fine-lr", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.fine_side != 4 * (args.coarse_side - 1) + 1:
        raise ValueError("fine side must be two dyadic refinements of coarse side")
    device = torch.device(args.device)
    width = args.coarse_patch_cells / (args.coarse_side - 1)
    coarse_identity, coarse_target = target_map(
        args.coarse_side, amplitude=args.amplitude,
        offset_fraction=args.offset_fraction, width=width,
        high_amplitude=args.high_amplitude, device=device,
    )
    _, fine_target = target_map(
        args.fine_side, amplitude=args.amplitude,
        offset_fraction=args.offset_fraction, width=width,
        high_amplitude=args.high_amplitude, device=device,
    )
    coarse_layer = ResidualStaggeredPatchP1Layer(
        args.coarse_side, args.coarse_patch_cells, cycles=args.coarse_cycles,
        minimum_jacobian=0.05,
    ).to(device)
    coarse_latent = torch.nn.Parameter(torch.zeros(
        (1, args.coarse_side - 2, args.coarse_side - 2, 2), device=device,
    ))
    coarse_output, coarse_result = fit(
        coarse_layer, coarse_identity, coarse_target, coarse_latent,
        steps=args.coarse_steps, lr=args.coarse_lr, device=device,
    )
    with torch.no_grad():
        fine_base = exact_dyadic_p1_refine(exact_dyadic_p1_refine(coarse_output))
    initial_fine_rmse = vector_rmse(fine_base, fine_target)
    fine_layer = SafeColoredVertexRelaxation(
        args.fine_side, safety_fraction=0.85, motion_mode="radial", raw_span=2.0,
        floor_fraction=0.05,
    ).to(device)
    span = 2.0 / (args.fine_side - 1)
    desired = ((fine_target - fine_base)[:, 1:-1, 1:-1] / span)
    teacher = torch.atanh(desired.clamp(-0.999999, 0.999999))
    with torch.no_grad():
        teacher_output = fine_layer(fine_base, teacher)
    fine_latent = torch.nn.Parameter(torch.zeros_like(teacher))
    final, fine_result = fit(
        fine_layer, fine_base, fine_target, fine_latent,
        steps=args.fine_steps, lr=args.fine_lr, device=device,
    )
    cotangent = torch.randn_like(final)
    gradient = torch.autograd.grad(
        (fine_layer(fine_base, fine_latent) * cotangent).mean(), fine_latent,
    )[0]
    joint_times = []
    joint_gradients = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        joint_coarse = coarse_layer(coarse_identity, coarse_latent)
        joint_base = exact_dyadic_p1_refine(exact_dyadic_p1_refine(joint_coarse))
        joint_output = fine_layer(joint_base, fine_latent)
        joint_gradients = torch.autograd.grad(
            (joint_output * cotangent).mean(), (coarse_latent, fine_latent),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        joint_times.append(time.perf_counter() - tick)
    print(json.dumps({
        "method": "phase7_coarse_patch_fine_vertex_fit",
        "coarse_side": args.coarse_side,
        "fine_side": args.fine_side,
        "coarse_vertices": args.coarse_side ** 2,
        "fine_vertices": args.fine_side ** 2,
        "fine_faces": 2 * (args.fine_side - 1) ** 2,
        "physical_patch_width": width,
        "amplitude": args.amplitude,
        "high_amplitude": args.high_amplitude,
        "offset_fraction": args.offset_fraction,
        "target_minimum_jacobian": minimum_jacobian(fine_target),
        "coarse": coarse_result,
        "refined_coarse_initial_fine_rmse": initial_fine_rmse,
        "refined_coarse_minimum_jacobian": minimum_jacobian(fine_base),
        "fine_teacher_logit_max_abs": float(teacher.abs().amax()),
        "fine_teacher_rmse": vector_rmse(teacher_output, fine_target),
        "fine_teacher_minimum_jacobian": minimum_jacobian(teacher_output),
        "fine": fine_result,
        "fine_random_vjp_finite_nonzero": bool(
            torch.isfinite(gradient).all() and gradient.abs().amax() > 0
        ),
        "joint_full_chain_vjp": {
            "median_forward_and_vjp_seconds": statistics.median(joint_times),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "coarse_gradient_finite_nonzero": bool(
                torch.isfinite(joint_gradients[0]).all()
                and joint_gradients[0].abs().amax() > 0
            ),
            "fine_gradient_finite_nonzero": bool(
                torch.isfinite(joint_gradients[1]).all()
                and joint_gradients[1].abs().amax() > 0
            ),
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "dtype": "float32",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
