"""Million-vertex forward/VJP and exact-shear fit for a prefix-sum P1 layer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from qcopt.neural_bijection.dense import (
    MonotoneFiberP1Layer,
    SoftplusPotentialFiberP1Layer,
)


def minimum_jacobian(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]
    def cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return float(torch.minimum(
        cross(b - a, c - a).amin(), cross(c - a, d - a).amin(),
    ).detach() * (mapped.shape[1] - 1) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--axis", choices=("horizontal", "vertical"),
                        default="horizontal")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parameterization", choices=("density", "potential"),
                        default="density")
    args = parser.parse_args()
    device = torch.device(args.device)
    layer = (MonotoneFiberP1Layer(args.side, axis=args.axis).to(device)
             if args.parameterization == "density" else
             SoftplusPotentialFiberP1Layer(args.side, axis=args.axis).to(device))
    shape = ((args.batch, args.side - 2, args.side - 2)
             if args.parameterization == "potential" else
             ((args.batch, args.side - 2, args.side - 1)
              if args.axis == "horizontal" else
              (args.batch, args.side - 1, args.side - 2)))
    generator = torch.Generator(device=device).manual_seed(75102)
    latent = (0.05 * torch.randn(shape, generator=generator,
                                  device=device)).requires_grad_()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        layer(latent)
    cotangent = torch.randn((args.batch, args.side, args.side, 2),
                             generator=generator, device=device)
    (layer(latent) * cotangent).mean().backward()
    latent.grad = None
    sync()
    forward_times = []
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(5):
            sync()
            tick = time.perf_counter()
            mapped = layer(latent)
            sync()
            forward_times.append(time.perf_counter() - tick)
        forward_peak = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
    joint_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(5):
        latent.grad = None
        sync()
        tick = time.perf_counter()
        output = layer(latent)
        (output * cotangent).mean().backward()
        sync()
        joint_times.append(time.perf_counter() - tick)
    vjp_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )

    oracle = None
    if args.axis == "horizontal":
        side = args.side
        axis = torch.arange(side, device=device, dtype=torch.float64) / (side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        target = torch.stack((
            xx + 0.28 * torch.sin(math.pi * xx).square() * torch.sin(2 * math.pi * yy),
            yy,
        ), dim=-1)[None].to(torch.float32)
        if args.parameterization == "potential":
            displacement = target[:, 1:-1, 1:-1, 0] - xx[None, 1:-1, 1:-1].float()
            teacher = torch.atanh(displacement / layer.potential_span)
        else:
            edge = target[:, 1:-1, 1:, 0] - target[:, 1:-1, :-1, 0]
            normalized = (edge - 0.05 / (side - 1)) / 0.95
            logweight = normalized.log()
            centered = logweight - logweight.mean(dim=-1, keepdim=True)
            teacher = torch.atanh(centered / 8)
        produced = layer(teacher)
        oracle = {
            "shear": 0.28,
            "maximum_coordinate_error": float((produced - target).abs().amax()),
            "vertex_vector_rmse": float(
                (produced - target).square().sum(dim=-1).mean().sqrt()
            ),
            "minimum_jacobian": minimum_jacobian(produced),
        }
    print(json.dumps({
        "method": "phase7_monotone_fiber_p1_profile",
        "parameterization": args.parameterization,
        "side": args.side,
        "vertices": args.side**2,
        "faces": 2 * (args.side - 1)**2,
        "axis": args.axis,
        "batch": args.batch,
        "dtype": "float32",
        "device": str(device),
        "torch_version": torch.__version__,
        "random_minimum_jacobian": minimum_jacobian(mapped),
        "random_gradient_finite": bool(torch.isfinite(latent.grad).all()),
        "median_forward_seconds": statistics.median(forward_times),
        "median_forward_and_vjp_seconds": statistics.median(joint_times),
        "forward_peak_allocated_bytes": forward_peak,
        "forward_and_vjp_peak_allocated_bytes": vjp_peak,
        "shear_oracle": oracle,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
