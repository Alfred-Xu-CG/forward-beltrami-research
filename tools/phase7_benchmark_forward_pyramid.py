"""Full fixed-P1 pyramid, 512^2 query/image loss, and all-latent VJP timing."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ForwardP1Pyramid
from qcopt.neural_bijection.dense.convex_quad import certify_convex_quad_output
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-side", type=int, default=5)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--seed-passes", type=int, default=2)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.batch < 1 or args.image_side < 2:
        raise ValueError("repeats, batch and image side must be positive")
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    torch.manual_seed(20260924)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    began = time.perf_counter()
    decoder = ForwardP1Pyramid(
        args.seed_side, args.side, seed_passes=args.seed_passes,
        checkpoint_passes=args.checkpoint,
    ).to(device)
    seed = [
        (0.08 * torch.randn(args.batch, args.seed_side - 2, args.seed_side - 2, 2,
                            dtype=dtype, device=device)).requires_grad_()
        for _ in range(args.seed_passes)
    ]
    levels = [
        (0.03 * torch.randn(args.batch, n - 2, n - 2, 2,
                            dtype=dtype, device=device)).requires_grad_()
        for n in decoder.level_sides
    ]
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(device=device, dtype=dtype)
    axis = torch.arange(args.image_side, device=device, dtype=dtype) / (args.image_side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    moving = (0.3 * torch.sin(8 * math.pi * xx + 3 * math.pi * yy)
              + 0.2 * torch.cos(13 * math.pi * yy - 2 * math.pi * xx))[None, None].expand(args.batch, -1, -1, -1)
    target = moving.detach()
    sync()
    setup_seconds = time.perf_counter() - began

    def once() -> tuple[torch.Tensor, torch.Tensor]:
        control = decoder(seed, levels)
        dense = table.interpolate(control.reshape(args.batch, -1, 2))
        warped = F.grid_sample(moving, 2 * dense - 1, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return (warped - target).square().mean(), control

    for _ in range(2):
        loss, _ = once()
        loss.backward()
        for latent in seed + levels:
            latent.grad = None
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, vjp_times = [], []
    gradients_finite = True
    nonzero_gradient_latents = 0
    for _ in range(args.repeats):
        sync()
        start = time.perf_counter()
        loss, control = once()
        sync()
        forward_end = time.perf_counter()
        loss.backward()
        sync()
        end = time.perf_counter()
        gradients_finite = gradients_finite and all(
            latent.grad is not None and torch.isfinite(latent.grad).all().item()
            for latent in seed + levels
        )
        nonzero_gradient_latents = sum(
            int(latent.grad is not None and latent.grad.abs().sum().item() > 0)
            for latent in seed + levels
        )
        forward_times.append(forward_end - start)
        vjp_times.append(end - forward_end)
        for latent in seed + levels:
            latent.grad = None
    with torch.no_grad():
        minimum = certify_convex_quad_output(control)
    print(json.dumps({
        "method": "phase7_forward_p1_pyramid",
        "output": "single_fixed_original_grid_P1_homeomorphism",
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "seed_side": args.seed_side,
        "seed_passes": args.seed_passes,
        "level_sides": decoder.level_sides,
        "latent_scalars": sum(x.numel() for x in seed + levels),
        "batch": args.batch,
        "image_side": args.image_side,
        "image_queries": args.image_side ** 2,
        "dtype": args.dtype,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "checkpoint_passes": args.checkpoint,
        "setup_seconds": setup_seconds,
        "median_full_forward_seconds": statistics.median(forward_times),
        "median_full_vjp_seconds": statistics.median(vjp_times),
        "forward_seconds": forward_times,
        "vjp_seconds": vjp_times,
        "minimum_signed_area_ratio": minimum,
        "gradients_finite": gradients_finite,
        "nonzero_gradient_latents": nonzero_gradient_latents,
        "last_image_loss": float(loss),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
