"""Dense fine-grid positive-edge Woodbury forward/VJP benchmark."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time

import numpy as np
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import SparseEdgeWoodburyTutteLayer
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


def regular_selected_edges(side: int, cells_per_axis: int) -> np.ndarray:
    """Pick one diagonal fine edge at each cell of a coarse spatial lattice."""
    mesh = structured_rectangle(side - 1, side - 1)
    reference = MatrixFreeSymmetricTutteLayer(mesh)
    lookup = {tuple(edge): i for i, edge in enumerate(reference.active_edges)}
    indices = []
    for cy in range(cells_per_axis):
        for cx in range(cells_per_axis):
            x = min(side - 2, int((cx + 0.5) * (side - 1) / cells_per_axis))
            y = min(side - 2, int((cy + 0.5) * (side - 1) / cells_per_axis))
            first = y * side + x
            indices.append(lookup[(first, first + side + 1)])
    return np.asarray(indices, dtype=np.int64)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--cells-per-axis", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if args.side < 3 or args.cells_per_axis < 1 or args.cells_per_axis > args.side - 1:
        raise ValueError("invalid grid or selected-edge lattice")
    torch.manual_seed(20260923)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    selected = regular_selected_edges(args.side, args.cells_per_axis)
    layer = SparseEdgeWoodburyTutteLayer(mesh, selected).to(device=device, dtype=dtype)
    logits = (-1.0 + 0.3 * torch.randn(args.batch, layer.rank, device=device, dtype=dtype)).requires_grad_()
    coordinate = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=dtype, device=device)
    bump = torch.sin(2 * math.pi * coordinate[:, 0]) * torch.sin(2 * math.pi * coordinate[:, 1])
    target = torch.stack((coordinate[:, 0] + 0.03 * bump, coordinate[:, 1] + 0.05 * bump), dim=-1)
    for _ in range(2):
        output = layer(logits)
        (output - target).square().mean().backward()
        logits.grad = None
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times = []
    backward_times = []
    for _ in range(args.repeat):
        began = time.perf_counter()
        output = layer(logits)
        loss = (output - target).square().mean()
        _sync(device)
        middle = time.perf_counter()
        loss.backward()
        _sync(device)
        ended = time.perf_counter()
        forward_times.append(middle - began)
        backward_times.append(ended - middle)
        logits.grad = None
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), device=device)
    triangles = output[:, faces]
    first = triangles[:, :, 1] - triangles[:, :, 0]
    second = triangles[:, :, 2] - triangles[:, :, 0]
    areas = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    map_difference = (output - target).square().mean().sqrt().item()
    oracle_error = None
    oracle_residual = None
    if args.oracle:
        delta = (layer.minimum_increment + torch.nn.functional.softplus(logits.detach()[0])).cpu().numpy()
        oracle, oracle_residual = layer.independent_sparse_solve(delta)
        oracle_error = float(np.max(np.abs(output.detach()[0].cpu().numpy() - oracle)))
    print(json.dumps({
        "route": "C",
        "method": "fine_grid_positive_edge_woodbury",
        "control_side": args.side,
        "control_vertices": mesh.n_vertices,
        "control_faces": mesh.n_faces,
        "active_edges": layer.n_active_edges,
        "interior_vertices": layer.n_interior,
        "updated_edges": layer.rank,
        "batch": args.batch,
        "dtype": args.dtype,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "scipy_version": __import__("scipy").__version__,
        "setup_seconds": layer.setup_seconds,
        "factor_nnz_L_plus_U": layer.factor_nnz,
        "stored_solution_columns_bytes": layer.precomputed_bytes,
        "forward_seconds": forward_times,
        "backward_seconds": backward_times,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "minimum_signed_area_ratio": float(areas.min().item() * (args.side - 1) ** 2),
        "untrained_map_rmse": map_difference,
        "independent_sparse_max_absolute_error": oracle_error,
        "independent_sparse_relative_residual": oracle_residual,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
