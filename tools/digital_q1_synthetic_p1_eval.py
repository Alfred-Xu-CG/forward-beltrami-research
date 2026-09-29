"""Evaluate fixed-diagonal P1 deployment of a trained safe Q1 vertex network."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from qcopt.neural_bijection.dense.q1_image_sampling import q1_map_at_pixel_centers
from tools.digital_q1_network_teacher import load_checkpoint
from tools.digital_q1_p1_layer import FactorizedP1Map, evaluate_factorized_p1
from tools.digital_q1_synthetic_slide_holdout import (
    _batch, analytic_map, make_coefficients, point_grid,
)


@torch.no_grad()
def evaluate(*, textures: torch.Tensor, test_texture_indices: list[int],
             weights: Path, side: int, test_count: int, batch: int,
             device: str) -> dict:
    target_device = torch.device(device)
    textures = textures.to(target_device, dtype=torch.float32)
    image_side = int(textures.shape[-1])
    pixel_points = point_grid(image_side, centers=True, device=target_device)
    vertex_points = point_grid(side, centers=False, device=target_device)
    coefficients = make_coefficients(test_count, 291003).to(target_device)
    model = load_checkpoint(weights, device=device)
    model.eval()
    if model.decoder.final_side != side:
        raise ValueError("checkpoint and requested control side differ")
    cases = []
    full_p1_times = []
    minimum_p1_triangle = float("inf")
    nonpositive_p1_triangles = 0
    for start in range(0, test_count, batch):
        indices = torch.arange(start, min(start + batch, test_count), device=target_device)
        fixed, moving, _ = _batch(textures[test_texture_indices], coefficients,
                                  indices, pixel_points, vertex_points)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        timed = time.perf_counter()
        predicted, _, _ = model(fixed, moving)
        size = indices.numel()
        q = pixel_points.expand(size, -1, -1, -1)
        p1 = evaluate_factorized_p1(
            FactorizedP1Map(
                residual_vertices=predicted,
                post_affine_matrix=torch.eye(2, device=target_device)[None].expand(size, -1, -1),
                post_affine_offset=torch.zeros((size, 2), device=target_device),
            ),
            q.reshape(size, -1, 2),
        ).reshape(size, image_side, image_side, 2)
        p1_image = F.grid_sample(moving, 2 * p1 - 1, mode="bilinear",
                                 padding_mode="border", align_corners=False)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        full_p1_times.append((time.perf_counter() - timed) / size)
        q1 = q1_map_at_pixel_centers(predicted, image_side, image_side)
        truth = analytic_map(pixel_points, coefficients[indices])
        corners = q1_corner_determinants(predicted)
        p1_areas = corners[..., (1, 3)]
        nonpositive_p1_triangles += int((p1_areas <= 0).sum())
        minimum_p1_triangle = min(minimum_p1_triangle, float(p1_areas.min()))
        q1_image = F.grid_sample(moving, 2 * q1 - 1, mode="bilinear",
                                 padding_mode="border", align_corners=False)
        for local in range(size):
            cases.append({
                "sample_index": int(indices[local]),
                "texture_index": test_texture_indices[int(indices[local]) % len(test_texture_indices)],
                "p1_query_map_rmse": float((p1[local] - truth[local]).square().sum(-1).mean().sqrt()),
                "q1_query_map_rmse": float((q1[local] - truth[local]).square().sum(-1).mean().sqrt()),
                "p1_q1_query_difference_rmse": float((p1[local] - q1[local]).square().sum(-1).mean().sqrt()),
                "p1_image_mse": float((p1_image[local] - fixed[local]).square().mean()),
                "q1_image_mse": float((q1_image[local] - fixed[local]).square().mean()),
            })
    means = {
        key + "_mean": statistics.mean(item[key] for item in cases)
        for key in ("p1_query_map_rmse", "q1_query_map_rmse",
                    "p1_q1_query_difference_rmse", "p1_image_mse", "q1_image_mse")
    }
    return {
        "test_texture_indices": test_texture_indices,
        "test_count": test_count,
        "image_side": image_side,
        "control_side": side,
        "fixed_sw_ne_p1_triangles_per_map": 2 * (side - 1) ** 2,
        "nonpositive_p1_triangles_all_test_maps_float32": nonpositive_p1_triangles,
        "minimum_unnormalized_p1_triangle_determinant_float32": minimum_p1_triangle,
        "pixel_query_fractions_on_sw_ne_diagonal": .5 if image_side == 512 and side == 257 else None,
        "full_p1_inference_seconds_per_image_median": statistics.median(full_p1_times),
        **means,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textures", type=Path, required=True)
    parser.add_argument("--test-texture-index", type=int, action="append", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.textures) as archive:
        textures = torch.from_numpy(archive["images"].copy())
    report = evaluate(textures=textures, test_texture_indices=args.test_texture_index,
                      weights=args.weights, side=args.side, test_count=args.test_count,
                      batch=args.batch, device=args.device)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
