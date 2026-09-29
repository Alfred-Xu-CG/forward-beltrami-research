"""Read-only fixed-boundary lower bound for saved ACROBAT pseudo-teacher fits.

The bound concerns a given DHR pseudo-target and a stored positive affine. It
does not bound anatomical landmark accuracy or establish teacher validity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _apply_affine(points: np.ndarray, matrix: np.ndarray,
                  offset: np.ndarray) -> np.ndarray:
    return np.stack((
        matrix[0, 0] * points[..., 0] + matrix[0, 1] * points[..., 1] + offset[0],
        matrix[1, 0] * points[..., 0] + matrix[1, 1] * points[..., 1] + offset[1],
    ), axis=-1)


def boundary_floor(reference: np.ndarray, matrix: np.ndarray,
                   offset: np.ndarray, teacher: np.ndarray,
                   candidate: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    teacher = np.asarray(teacher, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (reference.ndim != 3 or reference.shape[-1] != 2
            or reference.shape[0] != reference.shape[1] or reference.shape[0] < 3
            or teacher.shape != reference.shape or candidate.shape != reference.shape
            or matrix.shape != (2, 2) or offset.shape != (2,)
            or not all(np.isfinite(x).all() for x in
                       (reference, matrix, offset, teacher, candidate))):
        raise ValueError("finite matching square vertex tables and 2D affine required")
    side = reference.shape[0]
    boundary = np.zeros((side, side), dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    if not np.array_equal(candidate[boundary], reference[boundary]):
        raise ValueError("candidate residual boundary differs from fixed reference")
    if matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0] <= 0:
        raise ValueError("stored post-affine must preserve orientation")
    fixed_boundary_error = _apply_affine(reference, matrix, offset) - teacher
    full_error = _apply_affine(candidate, matrix, offset) - teacher
    denominator = side * side
    boundary_squared = float(np.sum(np.sum(fixed_boundary_error[boundary] ** 2,
                                           axis=-1)) / denominator)
    full_squared = float(np.sum(np.sum(full_error ** 2, axis=-1)) / denominator)
    if full_squared + 1e-14 < boundary_squared:
        raise ArithmeticError("candidate violates its boundary-only lower bound")
    return {
        "side": side,
        "vertices": denominator,
        "boundary_vertices": int(boundary.sum()),
        "boundary_floor_vector_rmse": boundary_squared ** .5,
        "candidate_full_vector_rmse": full_squared ** .5,
        "boundary_fraction_of_squared_error": (
            boundary_squared / full_squared if full_squared > 0 else 0.0),
        "boundary_floor_over_full_rmse": (
            (boundary_squared / full_squared) ** .5 if full_squared > 0 else 0.0),
        "mean_boundary_vector_error": float(np.linalg.norm(
            fixed_boundary_error[boundary], axis=-1).mean()),
        "max_boundary_vector_error": float(np.linalg.norm(
            fixed_boundary_error[boundary], axis=-1).max()),
    }


def analyze(root: Path, model_output: Path, cases: list[int], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if len(cases) != len(set(cases)) or not cases:
        raise ValueError("distinct nonempty case list required")
    rows = []
    for case in cases:
        with np.load(model_output / f"{case}_actual_safe_q1.npz",
                     allow_pickle=False) as archive:
            reference = archive["boundary_reference"][0]
            candidate = archive["vertices"][0]
            matrix = archive["post_affine_matrix"]
            offset = archive["post_affine_offset"]
        with np.load(root / f"{case}_DHR_physical_full_teacher_affine.npz",
                     allow_pickle=False) as archive:
            teacher = archive["raw_teacher_vertices"][0]
        rows.append({"case": case, **boundary_floor(
            reference, matrix, offset, teacher, candidate)})
    report = {
        "question": "unavoidable saved-affine pseudo-target error from fixed residual boundary",
        "metric": "Euclidean-vector RMSE over all vertices; boundary term only for lower bound",
        "scope": "DHR pseudo-target and saved affine only, not anatomical truth",
        "cases": rows,
        "mean_case_boundary_floor_vector_rmse": float(np.mean([
            row["boundary_floor_vector_rmse"] for row in rows])),
        "mean_case_boundary_fraction_of_squared_error": float(np.mean([
            row["boundary_fraction_of_squared_error"] for row in rows])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.model_output, args.cases, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
