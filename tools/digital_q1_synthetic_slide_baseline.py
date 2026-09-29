"""Image-only per-pair latent optimization baseline on the same safe Q1 decoder.

Uses exactly the held-out synthetic slide set from the image-network probe.
The analytic map is used only for post-optimization evaluation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.forward_q1_pyramid import HybridPatchSeedVertexQ1Pyramid
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.q1_image_sampling import warp_moving_at_q1_map
from tools.digital_q1_synthetic_slide_holdout import (
    _batch, make_coefficients, point_grid,
)


def optimize_baseline(*, textures: torch.Tensor, test_texture_indices: list[int],
                      side: int, steps: int, batch: int, test_count: int,
                      device: str, learning_rate: float,
                      regularization: float = 0.0,
                      coefficient_seed: int = 291003,
                      output_example: Path | None = None) -> dict:
    if min(side, steps, batch, test_count) < 1 or not test_texture_indices:
        raise ValueError("invalid baseline dimensions")
    if any(index < 0 or index >= textures.shape[0] for index in test_texture_indices):
        raise ValueError("texture indices must be canonical nonnegative indices")
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")
    target_device = torch.device(device)
    textures = textures.to(target_device, dtype=torch.float32)
    test_textures = textures[test_texture_indices]
    image_side = int(textures.shape[-1])
    pixel_points = point_grid(image_side, centers=True, device=target_device)
    vertex_points = point_grid(side, centers=False, device=target_device)
    test_coeff = make_coefficients(test_count, coefficient_seed).to(target_device)
    decoder = HybridPatchSeedVertexQ1Pyramid(17, side).to(target_device)
    cases = []
    step_times = []
    example = None
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    started_all = time.perf_counter()
    for start in range(0, test_count, batch):
        indices = torch.arange(start, min(start + batch, test_count), device=target_device)
        fixed, moving, target = _batch(test_textures, test_coeff, indices,
                                       pixel_points, vertex_points)
        size = indices.numel()
        seed = [torch.nn.Parameter(torch.zeros((size, 15, 15, 2), device=target_device))
                for _ in range(decoder.seed_passes)]
        levels = [torch.nn.Parameter(torch.zeros((size, level - 2, level - 2, 2),
                                                 device=target_device))
                  for level in decoder.level_sides]
        optimizer = torch.optim.Adam((*seed, *levels), lr=learning_rate)
        with torch.no_grad():
            initial_image_mse = ((moving - fixed).square().mean(dim=(1, 2, 3))).tolist()
        for _ in range(steps):
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            predicted = decoder(seed, levels)
            warped = warp_moving_at_q1_map(moving, predicted,
                                           height=image_side, width=image_side)
            loss = (warped - fixed).square().mean() + regularization * (
                predicted - vertex_points
            ).square().sum(-1).mean()
            loss.backward()
            gradients = [x.grad for x in (*seed, *levels)]
            if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
                raise FloatingPointError("nonfinite per-pair latent gradient")
            optimizer.step()
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            step_times.append(time.perf_counter() - started)
        with torch.no_grad():
            prediction = decoder(seed, levels)
            validity = validate_q1_map(
                prediction, vertex_points.expand(size, -1, -1, -1)
            )
            if validity["nonpositive_corners"] or validity["boundary_max_error"]:
                raise AssertionError("optimized Q1 output invalid")
            warped = warp_moving_at_q1_map(moving, prediction,
                                           height=image_side, width=image_side)
            for local in range(size):
                cases.append({
                    "sample_index": int(indices[local]),
                    "texture_index": test_texture_indices[int(indices[local]) % len(test_texture_indices)],
                    "map_rmse": float((prediction[local] - target[local]).square().sum(-1).mean().sqrt()),
                    "identity_rmse": float((vertex_points[0] - target[local]).square().sum(-1).mean().sqrt()),
                    "initial_image_mse": float(initial_image_mse[local]),
                    "optimized_image_mse": float((warped[local] - fixed[local]).square().mean()),
                })
            if example is None:
                example = prediction[0:1].cpu().numpy().copy()
    elapsed = time.perf_counter() - started_all
    saved_certificate = None
    if output_example is not None:
        assert example is not None
        output_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_example, vertices=example,
                            boundary_reference=vertex_points.cpu().numpy())
        saved_certificate = certify_q1_binary_map(output_example)
    return {
        "question": "how does no-network image-only optimization compare on identical held-out synthetic pairs?",
        "method": "per-pair Adam over F2 seed and all F1 level latent fields",
        "test_texture_indices": test_texture_indices,
        "control_side": side,
        "image_side": image_side,
        "steps_per_batch": steps,
        "batch": batch,
        "test_count": test_count,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "coefficient_seed": coefficient_seed,
        "mean_map_rmse": statistics.mean(x["map_rmse"] for x in cases),
        "mean_identity_rmse": statistics.mean(x["identity_rmse"] for x in cases),
        "mean_initial_image_mse": statistics.mean(x["initial_image_mse"] for x in cases),
        "mean_optimized_image_mse": statistics.mean(x["optimized_image_mse"] for x in cases),
        "median_complete_step_seconds": statistics.median(step_times),
        "total_optimization_seconds": elapsed,
        "seconds_per_pair": elapsed / test_count,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda" else None
        ),
        "saved_example_certificate": saved_certificate,
        "test_cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textures", type=Path, required=True)
    parser.add_argument("--test-texture-index", type=int, action="append", required=True)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=.04)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--coefficient-seed", type=int, default=291003)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-example", type=Path)
    args = parser.parse_args()
    with np.load(args.textures) as archive:
        textures = torch.from_numpy(archive["images"].copy())
        sources = archive["sources"].tolist() if "sources" in archive else None
    report = optimize_baseline(
        textures=textures, test_texture_indices=args.test_texture_index,
        side=args.side, steps=args.steps, batch=args.batch, test_count=args.test_count,
        device=args.device, learning_rate=args.learning_rate,
        regularization=args.regularization, coefficient_seed=args.coefficient_seed,
        output_example=args.output_example,
    )
    report["texture_sources"] = sources
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "test_cases"}, indent=2))


if __name__ == "__main__":
    main()
