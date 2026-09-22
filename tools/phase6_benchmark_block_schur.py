"""Benchmark exact all-edge block-Schur layer against complete sparse direct."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time

import numpy as np
import psutil
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ExactBlockSchurTutteLayer
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--patch-cells", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if args.batch < 1 or args.repeat < 1:
        raise ValueError("batch and repeat must be positive")
    torch.manual_seed(20260923)
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    layer = ExactBlockSchurTutteLayer(mesh, args.patch_cells)
    reference = MatrixFreeSymmetricTutteLayer(mesh)
    midpoint = mesh.vertices[reference.active_edges].mean(axis=1)
    x, y = midpoint[:, 0], midpoint[:, 1]
    coarse = 0.4 * np.sin(2 * math.pi * x) * np.sin(2 * math.pi * y)
    localized = 0.6 * np.exp(-((x - 0.42) ** 2 + (y - 0.61) ** 2) / 0.015)
    fine = 0.15 * np.sin(16 * math.pi * x) * np.sin(16 * math.pi * y)
    base_logits = -0.8 + coarse + localized + fine
    inputs = torch.tensor(np.stack([base_logits + 0.05 * b for b in range(args.batch)]), dtype=torch.float64, requires_grad=True)
    vertices = torch.tensor(np.array(mesh.vertices, copy=True), dtype=torch.float64)
    bump = torch.sin(2 * math.pi * vertices[:, 0]) * torch.sin(2 * math.pi * vertices[:, 1])
    target = torch.stack((vertices[:, 0] + 0.03 * bump, vertices[:, 1] + 0.05 * bump), dim=-1)
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    forward_times = []
    backward_times = []
    records = []
    peak_rss = initial_rss
    for _ in range(args.repeat):
        begin = time.perf_counter()
        mapped = layer(inputs)
        loss = (mapped - target).square().mean()
        middle = time.perf_counter()
        peak_rss = max(peak_rss, process.memory_info().rss)
        loss.backward()
        end = time.perf_counter()
        peak_rss = max(peak_rss, process.memory_info().rss)
        forward_times.append(middle - begin)
        backward_times.append(end - middle)
        records.append([s.__dict__ for s in layer.last_forward_stats])
        inputs.grad = None
    oracle_time = None
    oracle_error = None
    oracle_residual = None
    if args.oracle:
        start = time.perf_counter()
        direct, oracle_residual = layer.independent_full_solve(base_logits)
        oracle_time = time.perf_counter() - start
        oracle_error = float(np.max(np.abs(direct - mapped.detach().numpy()[0])))
        peak_rss = max(peak_rss, process.memory_info().rss)
    print(json.dumps({
        "route": "C",
        "method": "exact_patch_interior_Schur",
        "control_side": args.side,
        "control_vertices": mesh.n_vertices,
        "control_faces": mesh.n_faces,
        "active_edges": layer.n_conductances,
        "patch_cells": args.patch_cells,
        "patch_count": len(layer._blocks),
        "batch": args.batch,
        "dtype": "float64",
        "device": "cpu",
        "device_name": platform.processor(),
        "torch_version": torch.__version__,
        "scipy_version": __import__("scipy").__version__,
        "forward_seconds": forward_times,
        "backward_seconds": backward_times,
        "forward_stage_records": records,
        "adjoint_seconds": layer.last_adjoint_seconds,
        "initial_process_rss_bytes": initial_rss,
        "maximum_observed_process_rss_bytes": peak_rss,
        "untrained_map_rmse": (mapped - target).square().mean().sqrt().item(),
        "independent_full_direct_seconds": oracle_time,
        "independent_full_direct_max_absolute_error": oracle_error,
        "independent_full_direct_relative_residual": oracle_residual,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
