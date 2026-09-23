"""Test fine fixed-grid P1 updates beyond an identity-contraction certificate."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from qcopt.neural_bijection.dense import (
    MonotoneFiberP1Layer, SineModeP1Refiner, exact_dyadic_p1_refine,
)


def minimum_jacobian(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]
    def cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return float(torch.minimum(
        cross(b - a, c - a).amin(), cross(c - a, d - a).amin(),
    ).detach() * (mapped.shape[1] - 1) ** 2)


def shear_vertices(side: int, shear: float, device: torch.device) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    ux = shear * torch.sin(math.pi * xx).square() * torch.sin(2 * math.pi * yy)
    return torch.stack((xx + ux, yy), dim=-1)[None].to(torch.float32)


def fine_displacement(side: int, amplitude: torch.Tensor) -> torch.Tensor:
    axis = torch.arange(side, device=amplitude.device, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    high = torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)
    high = torch.where(high.abs() < 1e-12, 0, high)
    result = amplitude.new_zeros((amplitude.shape[0], side, side, 2))
    for row in range(4):
        for col in range(4):
            tx, ty = 4 * (xx - col / 4), 4 * (yy - row / 4)
            wx = torch.where((tx >= 0) & (tx <= 1),
                             torch.sin(math.pi * tx).square(), 0.0)
            wy = torch.where((ty >= 0) & (ty <= 1),
                             torch.sin(math.pi * ty).square(), 0.0)
            mode = (high * wx * wy).to(amplitude.dtype)
            result[:, :, :, 1] += amplitude[:, 4 * row + col, None, None] * mode
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", choices=("colored", "patch"), required=True)
    parser.add_argument("--shears", type=float, nargs="+", default=(0.2, 0.28))
    parser.add_argument("--amplitude-limit", type=float, default=0.00005)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--checkpoint-updates", action="store_true")
    parser.add_argument("--fiber-seed", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    windows = tuple(
        (col / 4, (col + 1) / 4, row / 4, (row + 1) / 4)
        for row in range(4) for col in range(4)
    )
    refiner = SineModeP1Refiner(
        257, 1025, cycles=((128, 128),) * 16,
        directions=((0.0, 1.0),) * 16,
        windows=windows, mechanism=args.mechanism, sweeps=args.sweeps,
        checkpoint_updates=args.checkpoint_updates,
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(719913)
    latent_values = (2 * torch.rand((2, 16), generator=generator, device=device) - 1)
    latent_values = latent_values * args.amplitude_limit

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    for shear in args.shears:
        teacher_coarse = shear_vertices(257, shear, device)
        fiber_latent = None
        if args.fiber_seed:
            edge = (teacher_coarse[:, 1:-1, 1:, 0]
                    - teacher_coarse[:, 1:-1, :-1, 0])
            weights = (edge - 0.05 / 256) / 0.95
            if weights.amin() <= 0:
                raise ValueError("teacher shear is outside the fiber floor")
            centered = weights.log() - weights.log().mean(dim=-1, keepdim=True)
            if (centered / 8).abs().amax() >= 1:
                raise ValueError("teacher shear exceeds bounded logit span")
            fiber_latent = torch.atanh(centered / 8).expand(2, -1, -1)
            fiber_latent = fiber_latent.clone().detach().requires_grad_(True)
            fiber_layer = MonotoneFiberP1Layer(257).to(device)
            coarse = fiber_layer(fiber_latent)
        else:
            coarse = teacher_coarse.expand(2, -1, -1, -1)
        baseline = exact_dyadic_p1_refine(exact_dyadic_p1_refine(coarse))
        expected_representable = baseline + fine_displacement(1025, latent_values)
        expected_analytic = shear_vertices(1025, shear, device).expand(2, -1, -1, -1) + fine_displacement(1025, latent_values)
        latent = latent_values.clone().detach().requires_grad_(True)
        sync()
        tick = time.perf_counter()
        actual = refiner(coarse, latent)
        sync()
        forward_seconds = time.perf_counter() - tick
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cotangent = torch.randn(actual.shape, generator=generator, device=device)
        sync()
        tick = time.perf_counter()
        (actual * cotangent).sum().backward()
        sync()
        backward_seconds = time.perf_counter() - tick
        rows.append({
            "shear": shear,
            "identity_displacement_lipschitz_lower_bound": 2 * math.pi * shear,
            "coarse_minimum_jacobian": minimum_jacobian(coarse),
            "target_analytic_p1_minimum_jacobian": minimum_jacobian(expected_analytic),
            "output_minimum_jacobian": minimum_jacobian(actual),
            "representable_vertex_vector_rmse": float(
                (actual - expected_representable).square().sum(dim=-1).mean().sqrt().detach()
            ),
            "representable_maximum_coordinate_error": float(
                (actual - expected_representable).abs().amax().detach()
            ),
            "analytic_vertex_vector_rmse": float(
                (actual - expected_analytic).square().sum(dim=-1).mean().sqrt().detach()
            ),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "latent_vjp_finite": bool(torch.isfinite(latent.grad).all()),
            "latent_vjp_max_abs": float(latent.grad.abs().amax()),
            "fiber_seed_vjp_finite": (
                bool(torch.isfinite(fiber_latent.grad).all())
                if fiber_latent is not None else None
            ),
            "fiber_seed_vjp_max_abs": (
                float(fiber_latent.grad.abs().amax())
                if fiber_latent is not None else None
            ),
            "vjp_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        })
    print(json.dumps({
        "method": "phase7_shear_compact_detail_safety_stress",
        "mechanism": args.mechanism,
        "amplitude_limit": args.amplitude_limit,
        "sweeps": args.sweeps,
        "checkpoint_updates": args.checkpoint_updates,
        "fiber_seed": args.fiber_seed,
        "batch": 2,
        "control_side": 1025,
        "control_faces": 2 * 1024**2,
        "device": str(device),
        "rows": rows,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
