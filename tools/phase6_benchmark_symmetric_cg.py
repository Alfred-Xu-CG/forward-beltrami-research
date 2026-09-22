"""Same fine-grid edge-update task with the existing all-edge CG decoder."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time

import numpy as np
import torch

from phase6_benchmark_edge_woodbury import regular_selected_edges
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--cells-per-axis", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args()
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    layer = MatrixFreeSymmetricTutteLayer(mesh, relative_tolerance=args.rtol, max_iterations=args.max_iterations).to(device)
    selected = regular_selected_edges(args.side, args.cells_per_axis)
    selected_tensor = torch.as_tensor(selected, dtype=torch.int64, device=device)
    latent = -1.0 + 0.3 * torch.randn(args.batch, len(selected), device=device, dtype=dtype)
    increments = torch.nn.functional.softplus(latent) + 1.0e-4
    conductance = torch.ones((args.batch, layer.n_conductances), dtype=dtype, device=device)
    conductance[:, selected_tensor] += increments
    logits = torch.log(torch.expm1(conductance - layer.minimum_conductance)).detach().requires_grad_()
    boundary_np = np.array(mesh.vertices[layer.boundary_vertices], copy=True)
    boundary = torch.tensor(boundary_np, dtype=dtype, device=device)
    coordinate = torch.tensor(np.array(mesh.vertices, copy=True), dtype=dtype, device=device)
    bump = torch.sin(2 * math.pi * coordinate[:, 0]) * torch.sin(2 * math.pi * coordinate[:, 1])
    target = torch.stack((coordinate[:, 0] + 0.03 * bump, coordinate[:, 1] + 0.05 * bump), dim=-1)
    forward_times = []
    backward_times = []
    forward_iterations = []
    backward_iterations = []
    for _ in range(args.repeat):
        began = time.perf_counter()
        mapped = layer(logits, boundary)
        loss = (mapped - target).square().mean()
        _sync(device)
        middle = time.perf_counter()
        forward_iterations.append(layer.last_forward_stats.iterations)
        loss.backward()
        _sync(device)
        ended = time.perf_counter()
        backward_iterations.append(layer.last_adjoint_stats.iterations)
        forward_times.append(middle - began)
        backward_times.append(ended - middle)
        logits.grad = None
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), device=device)
    triangles = mapped[:, faces]
    first = triangles[:, :, 1] - triangles[:, :, 0]
    second = triangles[:, :, 2] - triangles[:, :, 0]
    areas = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    print(json.dumps({
        "route": "C",
        "method": "existing_symmetric_matrix_free_CG",
        "control_side": args.side,
        "control_vertices": mesh.n_vertices,
        "control_faces": mesh.n_faces,
        "active_edges": layer.n_conductances,
        "updated_edges": len(selected),
        "batch": args.batch,
        "dtype": args.dtype,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "rtol": args.rtol,
        "max_iterations": args.max_iterations,
        "forward_seconds": forward_times,
        "backward_seconds": backward_times,
        "forward_iterations": forward_iterations,
        "backward_iterations": backward_iterations,
        "last_forward_relative_residual": layer.last_forward_stats.relative_residual,
        "last_backward_relative_residual": layer.last_adjoint_stats.relative_residual,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "minimum_signed_area_ratio": float(areas.min().item() * (args.side - 1) ** 2),
        "untrained_map_rmse": (mapped - target).square().mean().sqrt().item(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
