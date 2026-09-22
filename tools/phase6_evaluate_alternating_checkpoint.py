"""Exact a.e. Jacobian and Beltrami evaluation of a trained PL composition."""

from __future__ import annotations

import argparse
import json
import math
import platform

import torch

from qcopt.neural_bijection.dense import DenseMonotoneGridLayer, evaluate_structured_p1_with_jacobian


def target_and_jacobian(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = points[..., 0], points[..., 1]
    low = torch.sin(2 * math.pi * x) * torch.sin(2 * math.pi * y)
    high = torch.sin(16 * math.pi * x) * torch.sin(16 * math.pi * y)
    low_dx = 2 * math.pi * torch.cos(2 * math.pi * x) * torch.sin(2 * math.pi * y)
    low_dy = 2 * math.pi * torch.sin(2 * math.pi * x) * torch.cos(2 * math.pi * y)
    high_dx = 16 * math.pi * torch.cos(16 * math.pi * x) * torch.sin(16 * math.pi * y)
    high_dy = 16 * math.pi * torch.sin(16 * math.pi * x) * torch.cos(16 * math.pi * y)
    mapped = torch.stack((x + 0.03 * low + 0.003 * high, y + 0.05 * low + 0.003 * high), dim=-1)
    ux = 1 + 0.03 * low_dx + 0.003 * high_dx
    uy = 0.03 * low_dy + 0.003 * high_dy
    vx = 0.05 * low_dx + 0.003 * high_dx
    vy = 1 + 0.05 * low_dy + 0.003 * high_dy
    jacobian = torch.stack((torch.stack((ux, uy), dim=-1), torch.stack((vx, vy), dim=-1)), dim=-2)
    return mapped, jacobian


def beltrami(jacobian: torch.Tensor) -> torch.Tensor:
    ux = jacobian[..., 0, 0]
    uy = jacobian[..., 0, 1]
    vx = jacobian[..., 1, 0]
    vy = jacobian[..., 1, 1]
    fz = torch.complex(ux + vy, vx - uy)
    fbar = torch.complex(ux - vy, vx + uy)
    return fbar / fz


def determinant(jacobian: torch.Tensor) -> torch.Tensor:
    return jacobian[..., 0, 0] * jacobian[..., 1, 1] - jacobian[..., 0, 1] * jacobian[..., 1, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--query-side", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    state = torch.load(args.state, map_location="cpu", weights_only=True)
    if state["method"] != "alternating":
        raise ValueError("state must contain an alternating composition")
    side = int(state["side"])
    layers = int(state["layers"])
    device = torch.device(args.device)
    dtype = torch.float64
    parameters = [tensor.to(device=device, dtype=dtype) for tensor in state["parameters"]]
    controls = []
    for index in range(layers):
        axis = "vertical" if index % 2 == 0 else "horizontal"
        decoder = DenseMonotoneGridLayer(side, axis=axis)
        controls.append(decoder(parameters[index], parameters[layers + index]))
    coordinate = (torch.arange(args.query_side, device=device, dtype=dtype) + 0.5) / args.query_side
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    points = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)
    source_points = points
    jacobian = torch.eye(2, device=device, dtype=dtype)[None, None].expand(1, points.shape[1], 2, 2)
    for control in controls:
        points, current_jacobian = evaluate_structured_p1_with_jacobian(control, points)
        jacobian = torch.matmul(current_jacobian, jacobian)
    true_points, true_jacobian = target_and_jacobian(source_points)
    coefficient = beltrami(jacobian)
    true_coefficient = beltrami(true_jacobian)
    det = determinant(jacobian)
    true_det = determinant(true_jacobian)
    grid_line = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(grid_line, grid_line, indexing="ij")
    original_vertices = torch.stack((gx, gy), dim=-1).reshape(1, -1, 2)
    sampled_vertices = original_vertices
    for control in controls:
        sampled_vertices, _ = evaluate_structured_p1_with_jacobian(control, sampled_vertices)
    sampled = sampled_vertices.reshape(side, side, 2)
    a = sampled[:-1, :-1]
    b = sampled[:-1, 1:]
    c = sampled[1:, 1:]
    d = sampled[1:, :-1]
    lower = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    upper = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1]) - (c[..., 1] - a[..., 1]) * (d[..., 0] - a[..., 0])
    output = {
        "method": "alternating_checkpoint_exact_a.e._PL_jacobian",
        "state": args.state,
        "control_side": side,
        "control_vertices": side**2,
        "control_faces_per_layer": 2 * (side - 1) ** 2,
        "layers": layers,
        "query_side": args.query_side,
        "query_count": args.query_side**2,
        "query_location": "pixel_centers_not_vertices",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "calculation_dtype": "float64_from_float32_checkpoint",
        "query_map_rmse": (points - true_points).square().mean().sqrt().item(),
        "query_map_max_pointwise_error": torch.linalg.vector_norm(points - true_points, dim=-1).max().item(),
        "beltrami_area_sample_rmse": (coefficient - true_coefficient).abs().square().mean().sqrt().item(),
        "maximum_predicted_beltrami_modulus": coefficient.abs().max().item(),
        "maximum_target_beltrami_modulus": true_coefficient.abs().max().item(),
        "minimum_exact_composition_jacobian_determinant_sample": det.min().item(),
        "minimum_target_jacobian_determinant_sample": true_det.min().item(),
        "minimum_exported_original_grid_P1_signed_area_ratio": min(lower.min().item(), upper.min().item()) * (side - 1) ** 2,
        "negative_exported_original_grid_P1_faces": int((lower <= 0).sum().item() + (upper <= 0).sum().item()),
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
