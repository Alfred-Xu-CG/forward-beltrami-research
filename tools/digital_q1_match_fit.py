"""Fit the safe Q1 residual to image-generated SuperGlue correspondences.

This is pair-specific sparse-match optimization, not a trained image encoder.
The input JSON is produced without anatomical labels by the feature matcher.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense import HybridPatchSeedVertexQ1Pyramid
from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_q1_dhr_distill import identity_vertices


def q1_map_at_points(vertices: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Evaluate a BxNxNx2 Q1 map at BxKx2 source unit coordinates."""
    if vertices.ndim != 4 or vertices.shape[-1] != 2 or (
        points.ndim != 3 or points.shape[0] != vertices.shape[0] or points.shape[-1] != 2
    ):
        raise ValueError("expected BxNxNx2 vertices and BxKx2 points")
    grid = 2 * points[:, :, None, :] - 1
    return F.grid_sample(vertices.permute(0, 3, 1, 2), grid,
                         mode="bilinear", padding_mode="border",
                         align_corners=True)[:, :, :, 0].permute(0, 2, 1)


def fit_correspondences(
    source_points: torch.Tensor, target_points: torch.Tensor, *,
    final_side: int = 257, steps: int = 200, learning_rate: float = .04,
    seed_value: int = 290929, device: str = "cuda:0",
) -> tuple[torch.Tensor, dict]:
    """Train seed and first two F1 levels; select by held-back image matches."""
    if source_points.shape != target_points.shape or source_points.ndim != 2 or (
        source_points.shape[1] != 2 or len(source_points) < 16 or
        not bool(torch.isfinite(source_points).all()) or
        not bool(torch.isfinite(target_points).all()) or
        steps < 1 or learning_rate <= 0
    ):
        raise ValueError("finite Nx2 matches (N>=16) and positive fit budget required")
    n = len(source_points)
    permutation = np.random.default_rng(seed_value).permutation(n)
    train_count = int(.8 * n)
    train_index = torch.from_numpy(permutation[:train_count].copy())
    val_index = torch.from_numpy(permutation[train_count:].copy())
    target_device = torch.device(device)
    src = source_points.to(device=target_device, dtype=torch.float32)[None]
    dst = target_points.to(device=target_device, dtype=torch.float32)[None]
    train_index = train_index.to(target_device)
    val_index = val_index.to(target_device)
    decoder = HybridPatchSeedVertexQ1Pyramid(17, final_side, patch_cells=4).to(target_device)
    seed = tuple(torch.nn.Parameter(torch.zeros((1, 15, 15, 2), device=target_device))
                 for _ in range(decoder.seed_passes))
    levels = []
    trainable: list[torch.Tensor] = list(seed)
    for index, side in enumerate(decoder.level_sides):
        field = torch.zeros((1, side - 2, side - 2, 2), device=target_device)
        if index < 2:
            field = torch.nn.Parameter(field)
            trainable.append(field)
        levels.append(field)
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    identity = identity_vertices(final_side, device=target_device)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    with torch.no_grad():
        zero_map = decoder(seed, levels)
        zero_val = (q1_map_at_points(zero_map, src[:, val_index]) -
                    dst[:, val_index]).square().sum(-1).mean().sqrt()
        zero_train = (q1_map_at_points(zero_map, src[:, train_index]) -
                      dst[:, train_index]).square().sum(-1).mean().sqrt()
        best_map = zero_map.detach().clone()
        best_val = float(zero_val)
        best_train = float(zero_train)
    best_step = 0
    durations = []
    finite_steps = 0
    first_loss = last_loss = None
    for step in range(1, steps + 1):
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        mapped = decoder(seed, levels)
        predicted_train = q1_map_at_points(mapped, src[:, train_index])
        data_loss = (predicted_train - dst[:, train_index]).square().sum(-1).mean()
        loss = data_loss + .05 * (mapped - identity).square().mean()
        with torch.no_grad():
            val_rmse = float((q1_map_at_points(mapped, src[:, val_index]) -
                              dst[:, val_index]).square().sum(-1).mean().sqrt())
            if val_rmse < best_val:
                best_val = val_rmse
                best_train = float(data_loss.sqrt())
                best_map = mapped.detach().clone()
                best_step = step
        loss.backward()
        if not all(parameter.grad is not None and
                   bool(torch.isfinite(parameter.grad).all()) for parameter in trainable):
            raise FloatingPointError(f"nonfinite or missing latent gradient at step {step}")
        finite_steps += 1
        first_loss = float(loss.detach()) if first_loss is None else first_loss
        last_loss = float(loss.detach())
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        durations.append(time.perf_counter() - started)
    validity = validate_q1_map(best_map, identity)
    return best_map, {
        "mode": "per_pair_safe_Q1_fit_to_image_generated_sparse_matches",
        "control_side": final_side,
        "match_count": n,
        "train_matches": train_count,
        "heldback_matches": n - train_count,
        "split_seed": seed_value,
        "steps": steps,
        "learning_rate": learning_rate,
        "trainable_levels": ["17_F2_seed", *[f"{side}_F1" for side in decoder.level_sides[:2]]],
        "zero_step_train_match_vector_rmse_unit": float(zero_train),
        "zero_step_heldback_match_vector_rmse_unit": float(zero_val),
        "best_step": best_step,
        "best_train_match_vector_rmse_unit": best_train,
        "best_heldback_match_vector_rmse_unit": best_val,
        "first_regularized_train_loss": first_loss,
        "last_regularized_train_loss": last_loss,
        "median_complete_step_seconds": statistics.median(durations),
        "finite_gradient_steps": finite_steps,
        "residual_nonpositive_corners": validity["nonpositive_corners"],
        "residual_boundary_max_error": validity["boundary_max_error"],
        "cuda_peak_allocated_bytes": (None if target_device.type != "cuda" else
                                      torch.cuda.max_memory_allocated(target_device)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--affine-map", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--final-side", type=int, default=257)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=.04)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    match_data = json.loads(args.matches.read_text(encoding="utf-8"))
    if match_data.get("status") != "ok" or match_data.get("ransac_inliers", 0) < 16:
        raise ValueError("fit requires >=16 accepted image-generated matches")
    source = torch.as_tensor(match_data["source_points_unit"], dtype=torch.float32)
    target = torch.as_tensor(match_data["target_points_unit"], dtype=torch.float32)
    with np.load(args.affine_map) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    if matrix.shape != (2, 2) or offset.shape != (2,) or (
        not np.isfinite(matrix).all() or not np.isfinite(offset).all()
    ):
        raise ValueError("finite 2x2 affine and 2-vector required")
    a, b, c, d = (Fraction.from_float(float(value)) for value in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("affine factor must have exactly positive determinant")
    mapped, result = fit_correspondences(
        source, target, final_side=args.final_side, steps=args.steps,
        learning_rate=args.learning_rate, device=args.device,
    )
    reference = identity_vertices(args.final_side, device=torch.device("cpu")).numpy()
    np.savez_compressed(args.output_map, vertices=mapped.cpu().numpy(),
                        boundary_reference=reference,
                        post_affine_matrix=matrix, post_affine_offset=offset)
    certificate = certify_q1_binary_map(args.output_map)
    result.update({
        "image_matches": str(args.matches),
        "fixed_affine_archive": str(args.affine_map),
        "saved_map": str(args.output_map),
        "saved_binary_residual_valid": bool(certificate["valid"]),
        "saved_binary_nonpositive_corners": int(certificate["nonpositive_corners"]),
        "stored_affine_det_positive_exact": True,
        "saved_factorization_valid": bool(certificate["valid"]),
        "external_feature_extraction_seconds": match_data.get(
            "total_decode_align_model_match_seconds"),
    })
    args.output_report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
