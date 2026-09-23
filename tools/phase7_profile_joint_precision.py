"""Compare fp32/fp64 for the 257-to-1025 fixed-P1 decoder and its VJP."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, ResidualStaggeredPatchP1Layer,
    SafeColoredVertexRelaxation,
    exact_dyadic_p1_refine,
)
from phase7_multiscale_fiber_reachability import minimum_jacobian, vector_rmse


def analytic_target(side: int, dtype: torch.dtype, device: torch.device,
                    high_amplitude: float) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.arange(side, device=device, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    tx, ty = (xx - 0.28125) / 0.0625, (yy - 0.28125) / 0.0625
    wx = torch.where((tx >= 0) & (tx <= 1), torch.sin(math.pi * tx).square(), 0)
    wy = torch.where((ty >= 0) & (ty <= 1), torch.sin(math.pi * ty).square(), 0)
    high = (high_amplitude * torch.sin(256 * math.pi * xx)
            * torch.sin(256 * math.pi * yy) * torch.sin(math.pi * xx).square())
    displacement = 0.006 * wx * wy + high
    target = identity + torch.stack((displacement, displacement), dim=-1)[None]
    return identity, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--random-stress", type=int, default=0)
    parser.add_argument("--absolute-fine-floor", action="store_true")
    parser.add_argument("--combined-layer", action="store_true")
    parser.add_argument("--no-certificate", action="store_true")
    parser.add_argument("--latent-dtype", choices=("float32", "float64"))
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    latent_dtype = (torch.float32 if args.latent_dtype == "float32" else torch.float64
                    if args.latent_dtype == "float64" else dtype)
    device = torch.device(args.device)
    coarse_id, coarse_target = analytic_target(257, dtype, device, 2e-5)
    _, fine_target = analytic_target(1025, dtype, device, 2e-5)
    coarse = ResidualStaggeredPatchP1Layer(
        257, 16, cycles=2, minimum_jacobian=0.05,
    ).to(device)
    fine = SafeColoredVertexRelaxation(
        1025, safety_fraction=0.85, motion_mode="radial",
        raw_span=2.0, floor_fraction=(0.0 if args.absolute_fine_floor else 0.05),
    ).to(device)
    coarse_teacher = torch.atanh(
        ((coarse_target - coarse_id)[:, 1:-1, 1:-1] / 0.03125)
        .clamp(-1 + 4 * torch.finfo(dtype).eps, 1 - 4 * torch.finfo(dtype).eps)
    )
    with torch.no_grad():
        coarse_output = coarse(coarse_id, coarse_teacher)
        fine_base = exact_dyadic_p1_refine(exact_dyadic_p1_refine(coarse_output))
    fine_teacher = torch.atanh(
        ((fine_target - fine_base)[:, 1:-1, 1:-1] / (2 / 1024))
        .clamp(-1 + 4 * torch.finfo(dtype).eps, 1 - 4 * torch.finfo(dtype).eps)
    )
    combined = (CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, coarse_cycles=2,
        minimum_jacobian=0.05, compute_dtype=dtype,
        certify_output=not args.no_certificate,
    ).to(device) if args.combined_layer else None)

    def run(zc: torch.Tensor, zf: torch.Tensor) -> torch.Tensor:
        if combined is not None:
            return combined(zc, zf)
        y_coarse = coarse(coarse_id, zc)
        y_base = exact_dyadic_p1_refine(exact_dyadic_p1_refine(y_coarse))
        area_floor = (y_base.new_full((y_base.shape[0],), 0.05 / 1024 ** 2)
                      if args.absolute_fine_floor else None)
        return fine(y_base, zf, area_floor=area_floor)

    with torch.no_grad():
        output = run(coarse_teacher.to(latent_dtype), fine_teacher.to(latent_dtype))
    generator = torch.Generator(device=device).manual_seed(1949)
    cotangent = torch.randn(output.shape, device=device, dtype=dtype,
                             generator=generator)
    zc = coarse_teacher.to(latent_dtype).detach().clone().requires_grad_()
    zf = fine_teacher.to(latent_dtype).detach().clone().requires_grad_()
    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        value = run(zc, zf)
        gradients = torch.autograd.grad((value * cotangent).mean(), (zc, zf))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - tick)
    stress_minima = []
    stress_rejected = 0
    with torch.no_grad():
        for _ in range(args.random_stress):
            random_coarse = 5 * torch.randn(zc.shape, device=device, dtype=dtype,
                                             generator=generator)
            random_fine = 5 * torch.randn(zf.shape, device=device, dtype=dtype,
                                           generator=generator)
            stressed = run(random_coarse, random_fine)
            stress_minima.append(minimum_jacobian(stressed))
            if combined is not None and torch.equal(stressed, combined.fine_identity):
                stress_rejected += 1
    print(json.dumps({
        "method": "phase7_joint_precision_profile",
        "coarse_side": 257,
        "fine_side": 1025,
        "fine_vertices": 1025 ** 2,
        "fine_faces": 2 * 1024 ** 2,
        "dtype": args.dtype,
        "absolute_fine_floor": args.absolute_fine_floor,
        "combined_layer": args.combined_layer,
        "certificate_enabled": not args.no_certificate if args.combined_layer else False,
        "latent_dtype": str(latent_dtype),
        "device": str(device),
        "torch_version": torch.__version__,
        "coarse_teacher_rmse": vector_rmse(coarse_output, coarse_target),
        "final_teacher_rmse": vector_rmse(output, fine_target),
        "final_minimum_jacobian": minimum_jacobian(output),
        "minimum_jacobian_after_cast_to_float64": minimum_jacobian(output.double()),
        "coarse_vjp_finite_nonzero": bool(
            torch.isfinite(gradients[0]).all() and gradients[0].abs().amax() > 0
        ),
        "fine_vjp_finite_nonzero": bool(
            torch.isfinite(gradients[1]).all() and gradients[1].abs().amax() > 0
        ),
        "median_forward_and_vjp_seconds": statistics.median(times),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "random_stress_count": args.random_stress,
        "random_stress_rejected": stress_rejected,
        "random_stress_minimum_jacobians": stress_minima,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
