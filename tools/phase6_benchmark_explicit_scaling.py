"""Control-grid scaling of explicit fine-grid P1 and exact-composition layers."""

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

from phase6_train_image_to_latent import _minimum_area_ratio
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ExactAlternatingMonotoneComposition, MultiscaleMonotoneGridLayer
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


def _latents(side: int, batch: int, device: torch.device, dtype: torch.dtype) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    coarse = max(3, (side - 1) // 8 + 1)
    result = []
    for level in (coarse, side):
        global_logits = (0.08 * torch.randn(batch, level - 1, device=device, dtype=dtype)).requires_grad_()
        line_logits = (0.08 * torch.randn(batch, level, level - 1, device=device, dtype=dtype)).requires_grad_()
        result.append((global_logits, line_logits))
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("A", "AB2"), required=True)
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    if args.side < 2 or args.image_side < 2 or args.batch < 1 or args.repeat < 1:
        raise ValueError("invalid dimension, batch, or repeat")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    setup_start = time.perf_counter()
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1), height=args.image_side, width=args.image_side
    )
    table.prepare(device=device, dtype=dtype)
    axes = ("vertical",) if args.method == "A" else ("vertical", "horizontal")
    decoder = MultiscaleMonotoneGridLayer(args.side) if args.method == "A" else ExactAlternatingMonotoneComposition(args.side, table, axes)
    latents = tuple(_latents(args.side, args.batch, device, dtype) for _ in axes)
    line = torch.linspace(0.0, 1.0, args.image_side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    moving = (0.3 * torch.sin(8 * math.pi * xx + 3 * math.pi * yy) + 0.2 * torch.cos(13 * math.pi * yy - 2 * math.pi * xx))[None, None].expand(args.batch, -1, -1, -1)
    target = moving.detach()
    _sync(device)
    setup_seconds = time.perf_counter() - setup_start
    setup_rss = _rss_bytes()

    def decode() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if args.method == "A":
            control = decoder(latents[0])
            return table.interpolate(control.reshape(args.batch, -1, 2)), (control,)
        result = decoder(latents)
        return result.dense, result.controls

    for _ in range(2):
        dense, _ = decode()
        warped = F.grid_sample(moving, 2 * dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
        (warped - target).square().mean().backward()
        for level in latents:
            for global_logits, line_logits in level:
                global_logits.grad = None
                line_logits.grad = None
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    decode_times = []
    warp_loss_times = []
    backward_times = []
    sampled_peak_rss = setup_rss
    for _ in range(args.repeat):
        start = time.perf_counter()
        dense, controls = decode()
        _sync(device)
        after_decode = time.perf_counter()
        warped = F.grid_sample(moving, 2 * dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
        loss = (warped - target).square().mean()
        _sync(device)
        after_warp = time.perf_counter()
        loss.backward()
        _sync(device)
        after_backward = time.perf_counter()
        decode_times.append(after_decode - start)
        warp_loss_times.append(after_warp - after_decode)
        backward_times.append(after_backward - after_warp)
        current_rss = _rss_bytes()
        if current_rss is not None:
            sampled_peak_rss = max(sampled_peak_rss or 0, current_rss)
        for level in latents:
            for global_logits, line_logits in level:
                global_logits.grad = None
                line_logits.grad = None
    with torch.no_grad():
        minimum_ratios = [_minimum_area_ratio(control) for control in controls]
    print(json.dumps({
        "method": args.method,
        "representation": "original_grid_P1" if args.method == "A" else "exact_PL_composition",
        "side": args.side,
        "control_vertices": args.side**2,
        "control_faces_per_layer": 2 * (args.side - 1)**2,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "batch": args.batch,
        "layers": len(axes),
        "latent_sides": [(args.side - 1) // 8 + 1, args.side],
        "dtype": args.dtype,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "setup_seconds": setup_seconds,
        "median_decode_query_seconds": statistics.median(decode_times),
        "median_warp_loss_seconds": statistics.median(warp_loss_times),
        "median_backward_seconds": statistics.median(backward_times),
        "decode_query_seconds": decode_times,
        "warp_loss_seconds": warp_loss_times,
        "backward_seconds": backward_times,
        "minimum_layer_signed_area_ratios": minimum_ratios,
        "last_image_loss": loss.item(),
        "sampled_peak_process_rss_bytes": sampled_peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
