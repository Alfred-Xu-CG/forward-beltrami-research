"""Image-to-latent training through full-edge Schur or sparse-edge Woodbury."""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from phase6_benchmark_edge_woodbury import regular_selected_edges
from phase6_train_image_to_latent import _minimum_area_ratio, _synthetic_pair
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ExactBlockSchurTutteLayer, SparseEdgeWoodburyTutteLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


class MultiscaleEdgeImageEncoder(torch.nn.Module):
    """Sample coarse and fine image features at selected fine-edge midpoints."""

    def __init__(self, side: int, edge_midpoints: np.ndarray, width: int = 8) -> None:
        super().__init__()
        self.side = side
        self.coarse_side = min(side, max(3, (side - 1) // 8 + 1))
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(2, width, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.GELU(),
        )
        self.coarse_head = torch.nn.Conv2d(width, 1, 1)
        self.fine_head = torch.nn.Conv2d(width, 1, 1)
        grid = torch.tensor(2.0 * np.array(edge_midpoints, copy=True) - 1.0, dtype=torch.float32).reshape(1, 1, -1, 2)
        self.register_buffer("edge_grid", grid, persistent=False)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        feature = self.body(F.interpolate(pair, size=(self.side, self.side), mode="bilinear", align_corners=True))
        coarse_feature = F.interpolate(feature, size=(self.coarse_side, self.coarse_side), mode="bilinear", align_corners=True)
        grid = self.edge_grid.expand(pair.shape[0], -1, -1, -1)
        coarse = F.grid_sample(self.coarse_head(coarse_feature), grid, mode="bilinear", align_corners=True)[:, 0, 0]
        fine = F.grid_sample(self.fine_head(feature), grid, mode="bilinear", align_corners=True)[:, 0, 0]
        return -0.8 + coarse + fine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("schur", "woodbury"), required=True)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--patch-cells", type=int, default=16)
    parser.add_argument("--woodbury-cells-per-axis", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--target-kind", choices=("smooth", "high_frequency"), default="high_frequency")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.solver == "schur" and args.device != "cpu":
        raise ValueError("the current exact Schur implementation supports CPU only")
    torch.manual_seed(20260923)
    device = torch.device(args.device)
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    reference = MatrixFreeSymmetricTutteLayer(mesh)
    selected = None
    if args.solver == "schur":
        solver = ExactBlockSchurTutteLayer(mesh, args.patch_cells)
        active_edges = reference.active_edges
    else:
        selected = regular_selected_edges(args.side, args.woodbury_cells_per_axis)
        solver = SparseEdgeWoodburyTutteLayer(mesh, selected).to(device=device, dtype=torch.float32)
        active_edges = reference.active_edges[selected]
    encoder = MultiscaleEdgeImageEncoder(args.side, mesh.vertices[active_edges].mean(axis=1)).to(device)
    fixed, moving, true_map = _synthetic_pair(args.image_side, args.batch, device, args.target_kind)
    pair = torch.cat((fixed, moving), dim=1)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=args.image_side, width=args.image_side)
    table.prepare(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)

    def evaluate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = encoder(pair)
        control = solver(logits.double() if args.solver == "schur" else logits).float()
        queries = table.interpolate(control)
        warped = F.grid_sample(moving, 2.0 * queries - 1.0, mode="bilinear", padding_mode="border", align_corners=True)
        return (warped - fixed).square().mean(), queries, control

    with torch.no_grad():
        initial_loss = evaluate()[0].item()
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    maximum_rss = initial_rss
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    first_head_gradient_norms = None
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        began = time.perf_counter()
        loss, _, _ = evaluate()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        middle = time.perf_counter()
        loss.backward()
        if first_head_gradient_norms is None:
            first_head_gradient_norms = {
                "coarse": float(torch.linalg.vector_norm(torch.cat([parameter.grad.reshape(-1) for parameter in encoder.coarse_head.parameters()])).item()),
                "fine": float(torch.linalg.vector_norm(torch.cat([parameter.grad.reshape(-1) for parameter in encoder.fine_head.parameters()])).item()),
            }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        after_backward = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ended = time.perf_counter()
        maximum_rss = max(maximum_rss, process.memory_info().rss)
        records.append({"loss_before_update": loss.item(), "forward_seconds": middle - began, "backward_seconds": after_backward - middle, "optimizer_seconds": ended - after_backward})
    with torch.no_grad():
        final_loss, final_query, final_control = evaluate()
        map_rmse = (final_query - true_map).square().mean().sqrt().item()
        area_ratio = _minimum_area_ratio(final_control.reshape(args.batch, args.side, args.side, 2))
    print(json.dumps({
        "route": "C",
        "method": "all_edge_exact_block_schur" if args.solver == "schur" else "selected_edge_woodbury",
        "target_kind": args.target_kind,
        "control_side": args.side,
        "control_vertices": mesh.n_vertices,
        "control_faces": mesh.n_faces,
        "active_edges": reference.n_conductances,
        "learned_edges": len(active_edges),
        "latent_sides": [encoder.coarse_side, args.side],
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "batch": args.batch,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "solver_precision": "float64" if args.solver == "schur" else "float32",
        "torch_version": torch.__version__,
        "initial_image_mse": initial_loss,
        "final_image_mse": final_loss.item(),
        "final_query_map_rmse": map_rmse,
        "minimum_signed_area_ratio": area_ratio,
        "initial_process_rss_bytes": initial_rss,
        "maximum_observed_process_rss_bytes": maximum_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "solver_setup_seconds": solver.setup_seconds if args.solver == "woodbury" else None,
        "last_solver_relative_residual": solver.last_forward_stats[0].relative_residual if args.solver == "schur" else None,
        "step_records": records,
        "first_head_gradient_norms": first_head_gradient_norms,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
