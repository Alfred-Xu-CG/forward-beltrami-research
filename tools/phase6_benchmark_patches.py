"""Benchmark exact local-patch PL composition at dense control/query scales."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import LocalPatchComposition, LocalPatchMonotoneLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _min_area_ratio(control: torch.Tensor) -> float:
    a = control[:, :-1, :-1]
    b = control[:, :-1, 1:]
    c = control[:, 1:, 1:]
    d = control[:, 1:, :-1]
    lower = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (c[..., 0] - a[..., 0])
    upper = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1]) - (
        c[..., 1] - a[..., 1]
    ) * (d[..., 0] - a[..., 0])
    return min(lower.min().item(), upper.min().item()) * (control.shape[1] - 1) ** 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--patch-cells", type=int, default=16)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--layers", type=int, choices=(2, 4), default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    if args.side < args.patch_cells + 1 or args.patch_cells < 2:
        raise ValueError("invalid patch size")
    if args.patch_cells % 2:
        raise ValueError("patch_cells must be even for shifted partitions")
    if args.batch < 1 or args.repeat < 1:
        raise ValueError("batch and repeat must be positive")
    torch.manual_seed(20260923)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    layers = []
    for layer_id in range(args.layers):
        offset = args.patch_cells // 2 if layer_id % 2 else 0
        axis = "horizontal" if layer_id % 2 else "vertical"
        layers.append(
            LocalPatchMonotoneLayer(
                args.side,
                args.patch_cells,
                offset_x=offset,
                offset_y=offset,
                axis=axis,
            ).to(device)
        )
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side,
        width=args.image_side,
    )
    table.prepare(device=device, dtype=dtype)
    model = LocalPatchComposition(layers, table)
    latents = tuple(
        (0.3 * torch.randn(
            args.batch,
            layer.patch_count,
            args.patch_cells - 1,
            args.patch_cells,
            device=device,
            dtype=dtype,
        )).requires_grad_()
        for layer in layers
    )
    line = torch.linspace(0.0, 1.0, args.image_side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    target = torch.stack((xx + 0.03 * bump, yy + 0.05 * bump), dim=-1)[None]

    for _ in range(2):
        result = model(latents)
        (result.dense - target).square().mean().backward()
        for latent in latents:
            latent.grad = None
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times = []
    backward_times = []
    for _ in range(args.repeat):
        begin = time.perf_counter()
        result = model(latents)
        loss = (result.dense - target).square().mean()
        _sync(device)
        middle = time.perf_counter()
        loss.backward()
        _sync(device)
        end = time.perf_counter()
        forward_times.append(middle - begin)
        backward_times.append(end - middle)
        for latent in latents:
            latent.grad = None
    payload = {
        "route": "B",
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "patch_cells": args.patch_cells,
        "patch_counts": [layer.patch_count for layer in layers],
        "latent_values_per_sample": sum(
            layer.patch_count * (args.patch_cells - 1) * args.patch_cells for layer in layers
        ),
        "layers": args.layers,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "batch": args.batch,
        "dtype": args.dtype,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "forward_seconds": forward_times,
        "backward_seconds": backward_times,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "minimum_layer_signed_area_ratios": [_min_area_ratio(control) for control in result.controls],
        "sampled_composition_map_rmse": (result.dense - target).square().mean().sqrt().item(),
        "representation": "exact_PL_composition_not_P1_on_original_mesh",
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
