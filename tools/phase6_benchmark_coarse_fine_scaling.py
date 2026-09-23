"""Forward/VJP and memory scaling of exact coarse-fine PL composition."""

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

from qcopt.neural_bijection.dense import CoarseFineConvexQuadComposition, certify_convex_quad_output


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss() -> int | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        return None


def _parameters(decoder, batch: int, device: torch.device, dtype: torch.dtype):
    root = torch.nn.Parameter(0.08 * torch.randn(batch, 1, 1, 2, device=device, dtype=dtype))
    levels = tuple(
        (
            torch.nn.Parameter(0.08 * torch.randn(batch, n, n - 1, device=device, dtype=dtype)),
            torch.nn.Parameter(0.08 * torch.randn(batch, n - 1, n, device=device, dtype=dtype)),
            torch.nn.Parameter(0.08 * torch.randn(batch, n - 1, n - 1, 2, device=device, dtype=dtype)),
        )
        for n in decoder.latent_sides
    )
    return root, levels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-side", type=int, default=17)
    parser.add_argument("--fine-side", type=int, required=True)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    if args.repeat < 1 or args.batch < 1:
        raise ValueError("repeat and batch must be positive")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    started = time.perf_counter()
    layer = CoarseFineConvexQuadComposition(args.coarse_side, args.fine_side, args.image_side)
    layer.prepare(device=device, dtype=dtype)
    coarse = _parameters(layer.coarse, args.batch, device, dtype)
    fine = _parameters(layer.fine, args.batch, device, dtype)
    parameters = [coarse[0], *(x for level in coarse[1] for x in level), fine[0], *(x for level in fine[1] for x in level)]
    axis = torch.linspace(0, 1, args.image_side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    moving = (0.3 * torch.sin(8 * math.pi * xx + 3 * math.pi * yy) + 0.2 * torch.cos(13 * math.pi * yy - 2 * math.pi * xx))[None, None].expand(args.batch, -1, -1, -1)
    setup_seconds = time.perf_counter() - started
    setup_rss = _rss()

    def step():
        result = layer(coarse[0], coarse[1], fine[0], fine[1])
        warped = F.grid_sample(moving, 2 * result.dense - 1, mode="bilinear", padding_mode="border", align_corners=True)
        loss = (warped - moving).square().mean()
        return result, loss

    for _ in range(2):
        _, loss = step()
        loss.backward()
        for parameter in parameters:
            parameter.grad = None
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, backward_times = [], []
    peak_rss = setup_rss
    for _ in range(args.repeat):
        began = time.perf_counter()
        result, loss = step()
        _sync(device)
        middle = time.perf_counter()
        loss.backward()
        _sync(device)
        ended = time.perf_counter()
        forward_times.append(middle - began)
        backward_times.append(ended - middle)
        if (current := _rss()) is not None:
            peak_rss = max(peak_rss or 0, current)
        for parameter in parameters:
            parameter.grad = None
    print(json.dumps({
        "method": "CF2_exact_composition",
        "representation": "exact_PL_composition_not_original_grid_P1",
        "coarse_side": args.coarse_side,
        "fine_side": args.fine_side,
        "coarse_control_vertices": args.coarse_side**2,
        "fine_control_vertices": args.fine_side**2,
        "coarse_control_faces": 2 * (args.coarse_side - 1)**2,
        "fine_control_faces": 2 * (args.fine_side - 1)**2,
        "image_queries": args.image_side**2,
        "batch": args.batch,
        "latent_values": sum(parameter.numel() for parameter in parameters),
        "dtype": args.dtype,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "setup_seconds": setup_seconds,
        "forward_seconds": forward_times,
        "backward_seconds": backward_times,
        "median_forward_seconds": statistics.median(forward_times),
        "median_backward_seconds": statistics.median(backward_times),
        "minimum_factor_signed_area_ratios": [certify_convex_quad_output(control) for control in result.controls],
        "last_image_loss": loss.item(),
        "sampled_peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
