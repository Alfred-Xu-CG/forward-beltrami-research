"""Fixed-grid P1 multiscale fiber reproduction and full-grid VJP measurements.

Two analytic horizontal-fiber homeomorphisms are evaluated: a strong smooth
shear and a high-frequency detail that vanishes on the 257-vertex lattice.
The full-resolution latent is obtained by inverting the layer's row softmax;
coarser latents are ordinary bilinear projections of that *log-density*.
Every decoder output remains a P1 homeomorphism independently of fit quality.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense import MultiscaleMonotoneFiberP1Layer


def minimum_jacobian(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return float(torch.minimum(cross(b - a, c - a).amin(),
                               cross(c - a, d - a).amin()).detach()
                 * (mapped.shape[1] - 1) ** 2)


def teacher_map(side: int, kind: str, device: torch.device) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    if kind == "shear":
        displacement = 0.28 * torch.sin(math.pi * xx).square() * torch.sin(2 * math.pi * yy)
    else:
        displacement = (0.0001 * torch.sin(256 * math.pi * xx)
                        * torch.sin(256 * math.pi * yy)
                        * torch.sin(math.pi * xx).square())
    return torch.stack((xx + displacement, yy), dim=-1)[None].float()


def exact_teacher_latent(target: torch.Tensor, floor_fraction: float,
                         logit_span: float) -> torch.Tensor:
    side = target.shape[1]
    edge = target[:, 1:-1, 1:, 0] - target[:, 1:-1, :-1, 0]
    weight = (edge - floor_fraction / (side - 1)) / (1 - floor_fraction)
    if float(weight.amin()) <= 0:
        raise ValueError("target edge violates the decoder's positive floor")
    centered_log = weight.log() - weight.log().mean(dim=-1, keepdim=True)
    if float((centered_log / logit_span).abs().amax()) >= 1:
        raise ValueError("target log-density lies beyond tanh logit span")
    return torch.atanh(centered_log / logit_span)


def resize(field: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return F.interpolate(field[:, None], size=(rows, cols),
                         mode="bilinear", align_corners=True)[:, 0]


def vector_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).square().sum(dim=-1).mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kinds", nargs="+", choices=("shear", "high128"),
                        default=["shear", "high128"])
    parser.add_argument("--projection", choices=("bilinear", "area"),
                        default="bilinear")
    args = parser.parse_args()
    device = torch.device(args.device)
    layer = MultiscaleMonotoneFiberP1Layer(args.side).to(device)
    side_levels = [33, 65, 129, 257, 513, args.side]
    side_levels = list(dict.fromkeys(s for s in side_levels if s <= args.side))
    results = []
    for kind in args.kinds:
        target = teacher_map(args.side, kind, device).expand(args.batch, -1, -1, -1)
        exact = exact_teacher_latent(target, layer.fiber.floor_fraction,
                                     layer.fiber.logit_span)
        # A Laplacian-style decomposition of the *latent field*, not of maps.
        levels = []
        reconstruction = torch.zeros_like(exact)
        for coarse_side in side_levels:
            residual = exact - reconstruction
            if args.projection == "area" and coarse_side < args.side:
                low = F.interpolate(residual[:, None],
                                    size=(coarse_side - 2, coarse_side - 1),
                                    mode="area")[:, 0]
            else:
                low = resize(residual, coarse_side - 2, coarse_side - 1)
            levels.append(low)
            reconstruction = reconstruction + resize(low, args.side - 2,
                                                     args.side - 1)
            with torch.no_grad():
                output = layer(levels)
            results.append({
                "target": kind,
                "through_side": coarse_side,
                "latent_elements_per_sample": sum(x[0].numel() for x in levels),
                "vertex_vector_rmse": vector_rmse(output, target),
                "minimum_jacobian": minimum_jacobian(output),
            })

        levels = [x.detach().clone().requires_grad_() for x in levels]
        cotangent = torch.randn(target.shape, device=device)
        def sync() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        forward_times, joint_times = [], []
        for _ in range(5):
            sync()
            tick = time.perf_counter()
            with torch.no_grad():
                output = layer(levels)
            sync()
            forward_times.append(time.perf_counter() - tick)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(5):
            for x in levels:
                x.grad = None
            sync()
            tick = time.perf_counter()
            output = layer(levels)
            (output * cotangent).mean().backward()
            sync()
            joint_times.append(time.perf_counter() - tick)
        results.append({
            "target": kind,
            "through_side": "all",
            "full_grid_gradient_finite": all(bool(torch.isfinite(x.grad).all()) for x in levels),
            "full_grid_max_abs_gradient_by_level": [float(x.grad.abs().amax()) for x in levels],
            "median_forward_seconds": statistics.median(forward_times),
            "median_forward_and_vjp_seconds": statistics.median(joint_times),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        })
    print(json.dumps({
        "method": "phase7_multiscale_positive_fiber_p1",
        "side": args.side,
        "vertices": args.side ** 2,
        "faces": 2 * (args.side - 1) ** 2,
        "batch": args.batch,
        "device": str(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "projection": args.projection,
        "results": results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
