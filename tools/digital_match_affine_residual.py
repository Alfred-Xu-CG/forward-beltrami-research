"""Image-only affine correction composed after a certified residual P1 map.

Correspondences are SuperGlue/RANSAC results in the prewarped moving canvas.
No anatomical landmark or full-DHR field is loaded. This is a diagnostic of
global-init error, not a claim that affine refinement is a new neural layer.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_birl_landmark_score import p1_at_queries


def ridge_correction(source: np.ndarray, target: np.ndarray, *,
                     ridge: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Fit C(y)=My+t with an identity-centered ridge at unit-coordinate scale."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("paired Nx2 coordinates required")
    if len(source) < 16 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("at least 16 finite matches required")
    center = source.mean(axis=0)
    x = np.column_stack((source - center, np.ones(len(source))))
    residual = target - source
    penalty = np.diag((ridge, ridge, 0.0))
    coefficient = np.linalg.solve(x.T @ x + penalty, x.T @ residual)
    matrix = np.eye(2) + coefficient[:2].T
    offset = coefficient[2] - (matrix - np.eye(2)) @ center
    return matrix, offset


def refine(map_path: Path, match_path: Path, output: Path, *,
           ridge: float = 1.0, max_spectral_change: float = .2,
           max_shift: float = .1) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    with np.load(map_path) as archive:
        vertices = np.asarray(archive["vertices"], dtype=np.float32)
        reference = np.asarray(archive["boundary_reference"], dtype=np.float32)
        affine = np.asarray(archive["post_affine_matrix"], dtype=np.float64)
        shift = np.asarray(archive["post_affine_offset"], dtype=np.float64)
    with np.load(match_path) as matches:
        source = np.asarray(matches["source_fixed_unit"], dtype=np.float64)
        target = np.asarray(matches["target_aligned_unit"], dtype=np.float64)
    if affine.shape != (2, 2) or shift.shape != (2,) or not (
        np.isfinite(affine).all() and np.isfinite(shift).all()
    ):
        raise ValueError("finite input affine required")
    a, b, c, d = (Fraction.from_float(float(x)) for x in affine.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive input affine required")
    mapped_source = p1_at_queries(vertices[0].astype(np.float64), source)
    correction, correction_offset = ridge_correction(
        mapped_source, target, ridge=ridge)
    spectral = float(np.linalg.norm(correction - np.eye(2), ord=2))
    correction_det = float(np.linalg.det(correction))
    admissible = (correction_det > 0 and spectral <= max_spectral_change and
                  np.linalg.norm(correction_offset) <= max_shift)
    if not admissible:
        correction, correction_offset = np.eye(2), np.zeros(2)
    new_affine = (affine @ correction).astype(np.float32)
    new_shift = (affine @ correction_offset + shift).astype(np.float32)
    a, b, c, d = (Fraction.from_float(float(x)) for x in new_affine.flat)
    if not (np.isfinite(new_affine).all() and np.isfinite(new_shift).all()
            and a * d - b * c > 0):
        raise ValueError("rounded composite affine is not positive and finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=vertices, boundary_reference=reference,
                        post_affine_matrix=new_affine,
                        post_affine_offset=new_shift)
    binary = certify_q1_binary_map(output)
    if not binary["valid"]:
        raise RuntimeError("composed binary32 map not certified")
    residual_before = float(np.mean(np.linalg.norm(
        512 * (mapped_source - target), axis=1)))
    residual_after = float(np.mean(np.linalg.norm(
        512 * (mapped_source @ correction.T + correction_offset - target), axis=1)))
    report = {
        "method": "match-only identity-regularized affine correction after safe P1",
        "input_map": str(map_path), "matches": str(match_path),
        "output": str(output), "match_count": len(source),
        "ridge": ridge, "max_spectral_change": max_spectral_change,
        "max_shift": max_shift, "estimated_spectral_change": spectral,
        "estimated_correction_det": correction_det,
        "correction_admissible": bool(admissible),
        "applied_correction_matrix": correction.tolist(),
        "applied_correction_offset": correction_offset.tolist(),
        "input_machine_match_mean_px": residual_before,
        "output_machine_match_mean_px": residual_after,
        "saved_binary_certificate": binary,
        "anatomical_landmarks_or_DHR_full_field_used": False,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.)
    args = parser.parse_args()
    result = refine(args.map, args.matches, args.output, ridge=args.ridge)
    print(json.dumps({key: result[key] for key in (
        "match_count", "correction_admissible",
        "estimated_spectral_change", "input_machine_match_mean_px",
        "output_machine_match_mean_px",
    )}))


if __name__ == "__main__":
    main()
