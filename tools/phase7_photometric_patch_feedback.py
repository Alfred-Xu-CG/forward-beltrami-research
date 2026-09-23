"""Evaluate image-derived local flow hints through safe residual patch passes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    ResidualStaggeredPatchP1Layer, local_photometric_logits,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian
from phase7_train_random_patch_images import make_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--patch-cells", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=194381)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    width = args.patch_cells / (args.side - 1)
    dataset = tuple(t.to(device) for t in make_dataset(
        args.test_count, args.image_side, args.side, args.seed,
        amplitude_min=0.003, amplitude_max=0.006, patch_width=width,
    ))
    decoder = ResidualStaggeredPatchP1Layer(
        args.side, args.patch_cells, cycles=args.cycles,
        minimum_jacobian=0.05,
    ).to(device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float32)
    axis = torch.arange(args.side, device=device, dtype=torch.float32) / (args.side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    accumulated = [{"image": 0.0, "map": 0.0, "support": 0.0,
                    "support_n": 0, "min_j": float("inf")} for _ in range(args.iterations + 1)]
    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for start in range(0, args.test_count, args.batch):
            fixed, moving, target, support = (
                t[start:start + args.batch] for t in dataset
            )
            current = identity.expand(fixed.shape[0], -1, -1, -1)
            for iteration in range(args.iterations + 1):
                query = table.interpolate(current.reshape(fixed.shape[0], -1, 2))
                warped = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                                       padding_mode="border", align_corners=True)
                difference = (current - target).square().sum(dim=-1)
                item = accumulated[iteration]
                item["image"] += float((warped - fixed).square().mean()) * fixed.shape[0]
                item["map"] += float(difference.mean(dim=(-1, -2)).sum())
                mask = support > 0
                item["support"] += float(difference[mask].sum())
                item["support_n"] += int(mask.sum())
                item["min_j"] = min(item["min_j"], minimum_jacobian(current))
                if iteration < args.iterations:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    hint = local_photometric_logits(
                        fixed, moving, current, window=args.window,
                        ridge=args.ridge, raw_span=0.5 * args.patch_cells,
                    )
                    current = decoder(current, args.gain * hint)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - tick)
    results = [{
        "iterations": index,
        "image_mse": item["image"] / args.test_count,
        "control_vertex_vector_rmse": math.sqrt(item["map"] / args.test_count),
        "support_vertex_vector_rmse": math.sqrt(
            item["support"] / item["support_n"]
        ),
        "minimum_jacobian": item["min_j"],
    } for index, item in enumerate(accumulated)]
    print(json.dumps({
        "method": "phase7_photometric_patch_feedback",
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "image_side": args.image_side,
        "image_queries": args.image_side ** 2,
        "patch_cells": args.patch_cells,
        "cycles_per_iteration": args.cycles,
        "test_count": args.test_count,
        "seed": args.seed,
        "batch": args.batch,
        "window": args.window,
        "ridge": args.ridge,
        "gain": args.gain,
        "results": results,
        "median_one_feedback_iteration_seconds": statistics.median(times),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "dtype": "float32",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
