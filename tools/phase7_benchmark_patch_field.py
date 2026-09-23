"""Staggered simultaneous-patch P1 forward/VJP on real control grids."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ForwardPatchP1Pyramid, StaggeredPatchP1Layer
from qcopt.neural_bijection.dense.convex_quad import certify_convex_quad_output
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--pyramid", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--seed-side", type=int, default=17)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-std", type=float, default=0.08)
    args = parser.parse_args()
    torch.manual_seed(20260924)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    begin = time.perf_counter()
    layer = (
        ForwardPatchP1Pyramid(args.seed_side, args.side, patch_cells=args.patch_cells).to(device)
        if args.pyramid else StaggeredPatchP1Layer(args.side, args.patch_cells).to(device)
    )
    axis = torch.arange(args.side, device=device, dtype=torch.float32) / (args.side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None].expand(args.batch, -1, -1, -1)
    def make_fields(module: StaggeredPatchP1Layer, n: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            (args.latent_std * torch.randn(
                (args.batch, patch_pass.interior_ids.numel(), 2)
                if args.compact else (args.batch, n - 2, n - 2, 2),
                device=device,
            )).requires_grad_()
            for patch_pass in module.passes
        )
    seed_fields = make_fields(layer.seed_layer, args.seed_side) if args.pyramid else ()
    level_fields = tuple(
        make_fields(module, n) for n, module in zip(layer.level_sides, layer.level_layers)
    ) if args.pyramid else ()
    latents = (
        seed_fields + tuple(field for level in level_fields for field in level)
        if args.pyramid else make_fields(layer, args.side)
    )
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float32)
    image_axis = torch.arange(args.image_side, device=device, dtype=torch.float32) / (args.image_side - 1)
    iy, ix = torch.meshgrid(image_axis, image_axis, indexing="ij")
    moving = (0.3 * torch.sin(8 * math.pi * ix + 3 * math.pi * iy)
              + 0.2 * torch.cos(13 * math.pi * iy - 2 * math.pi * ix))[None, None].expand(args.batch, -1, -1, -1)
    setup_seconds = time.perf_counter() - begin

    def once() -> tuple[torch.Tensor, torch.Tensor]:
        control = layer(seed_fields, level_fields) if args.pyramid else layer(base, latents)
        dense = table.interpolate(control.reshape(args.batch, -1, 2))
        warped = F.grid_sample(moving, 2 * dense - 1, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return (warped - moving).square().mean(), control

    for _ in range(2):
        loss, _ = once()
        loss.backward()
        for latent in latents:
            latent.grad = None
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, vjp_times = [], []
    gradients_finite = True
    for _ in range(args.repeats):
        sync()
        started = time.perf_counter()
        loss, control = once()
        sync()
        middle = time.perf_counter()
        loss.backward()
        sync()
        ended = time.perf_counter()
        forward_times.append(middle - started)
        vjp_times.append(ended - middle)
        gradients_finite = gradients_finite and all(
            latent.grad is not None and torch.isfinite(latent.grad).all().item()
            for latent in latents
        )
        for latent in latents:
            latent.grad = None
    allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    with torch.no_grad():
        minimum = certify_convex_quad_output(control)
    print(json.dumps({
        "method": "phase7_patch_p1_pyramid" if args.pyramid else "phase7_staggered_patch_field",
        "output_representation": "single_fixed_original_grid_P1",
        "numerically_certified_homeomorphism": True,
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "patch_cells": args.patch_cells,
        "passes": 4 * (1 + len(layer.level_sides)) if args.pyramid else 4,
        "level_sides": layer.level_sides if args.pyramid else (),
        "latent_scalars": sum(x.numel() for x in latents),
        "latent_layout": "compact_active_patch_interiors" if args.compact else "full_interior_field",
        "latent_std": args.latent_std,
        "batch": args.batch,
        "image_side": args.image_side,
        "image_queries": args.image_side ** 2,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "setup_seconds": setup_seconds,
        "median_full_forward_seconds": statistics.median(forward_times),
        "median_full_vjp_seconds": statistics.median(vjp_times),
        "forward_seconds": forward_times,
        "vjp_seconds": vjp_times,
        "gradients_finite": gradients_finite,
        "minimum_signed_area_ratio": minimum,
        "last_image_loss": float(loss.detach()),
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
