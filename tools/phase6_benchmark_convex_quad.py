"""Scaling of the free-center fixed-P1 hierarchy with a dense image query."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import HierarchicalConvexQuadFreeCenterLayer, SafeColoredVertexRelaxation, certify_convex_quad_output
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--local-relaxation", action="store_true", help="add one four-color original-grid P1 latent pass")
    parser.add_argument("--safety-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if args.side < 5 or args.image_side < 2 or args.batch < 1 or args.repeat < 1:
        raise ValueError("invalid grid, batch, or repeat")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    setup_start = time.perf_counter()
    decoder = HierarchicalConvexQuadFreeCenterLayer(args.side)
    relaxation = SafeColoredVertexRelaxation(args.side, safety_fraction=args.safety_fraction).to(device) if args.local_relaxation else None
    root = torch.nn.Parameter(0.08 * torch.randn(args.batch, 1, 1, 2, device=device, dtype=dtype))
    latents = tuple(
        (
            torch.nn.Parameter(0.08 * torch.randn(args.batch, current, current - 1, device=device, dtype=dtype)),
            torch.nn.Parameter(0.08 * torch.randn(args.batch, current - 1, current, device=device, dtype=dtype)),
            torch.nn.Parameter(0.08 * torch.randn(args.batch, current - 1, current - 1, 2, device=device, dtype=dtype)),
        )
        for current in decoder.latent_sides
    )
    local_logits = torch.nn.Parameter(0.08 * torch.randn(args.batch, args.side - 2, args.side - 2, 2, device=device, dtype=dtype)) if args.local_relaxation else None
    parameters = (root, *(parameter for level in latents for parameter in level), *((local_logits,) if local_logits is not None else ()))
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side,
        width=args.image_side,
    )
    table.prepare(device=device, dtype=dtype)
    line = torch.linspace(0.0, 1.0, args.image_side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    moving = (0.3 * torch.sin(8 * math.pi * xx + 3 * math.pi * yy) + 0.2 * torch.cos(13 * math.pi * yy - 2 * math.pi * xx))[None, None].expand(args.batch, -1, -1, -1)
    target = moving.detach()
    _sync(device)
    setup_seconds = time.perf_counter() - setup_start
    setup_rss = _rss_bytes()

    def decode() -> tuple[torch.Tensor, torch.Tensor]:
        control = decoder(root, latents)
        if relaxation is not None:
            control = relaxation(control, local_logits)
        dense = table.interpolate(control.reshape(args.batch, -1, 2))
        return control, dense

    for _ in range(2):
        control, dense = decode()
        warped = F.grid_sample(moving, 2 * dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
        (warped - target).square().mean().backward()
        for parameter in parameters:
            parameter.grad = None
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    decode_query_times = []
    warp_loss_times = []
    backward_times = []
    sampled_peak_rss = setup_rss
    for _ in range(args.repeat):
        began = time.perf_counter()
        control, dense = decode()
        _sync(device)
        after_decode = time.perf_counter()
        warped = F.grid_sample(moving, 2 * dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
        loss = (warped - target).square().mean()
        _sync(device)
        after_warp = time.perf_counter()
        loss.backward()
        _sync(device)
        ended = time.perf_counter()
        decode_query_times.append(after_decode - began)
        warp_loss_times.append(after_warp - after_decode)
        backward_times.append(ended - after_warp)
        current = _rss_bytes()
        if current is not None:
            sampled_peak_rss = max(sampled_peak_rss or 0, current)
        for parameter in parameters:
            parameter.grad = None
    minimum_area = certify_convex_quad_output(control)
    print(json.dumps({
        "method": "A3_colored_local" if args.local_relaxation else "A2_free_center",
        "representation": "original_grid_P1",
        "side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "batch": args.batch,
        "latent_sides": [2, *decoder.latent_sides],
        "latent_values": sum(parameter.numel() for parameter in parameters),
        "dtype": args.dtype,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "setup_seconds": setup_seconds,
        "decode_query_seconds": decode_query_times,
        "warp_loss_seconds": warp_loss_times,
        "backward_seconds": backward_times,
        "median_decode_query_seconds": statistics.median(decode_query_times),
        "median_warp_loss_seconds": statistics.median(warp_loss_times),
        "median_backward_seconds": statistics.median(backward_times),
        "minimum_signed_area_ratio": minimum_area,
        "last_image_loss": loss.item(),
        "sampled_peak_process_rss_bytes": sampled_peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
