"""Measure DHR pseudo-teacher detail beyond its nested 129² P1 subgrid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.digital_boundary_error_floor import boundary_floor


def p1_refine(coarse: np.ndarray) -> np.ndarray:
    coarse = np.asarray(coarse, dtype=np.float64)
    if (coarse.ndim != 3 or coarse.shape[0] != coarse.shape[1]
            or coarse.shape[0] < 2 or coarse.shape[-1] != 2
            or not np.isfinite(coarse).all()):
        raise ValueError("finite square coarse vertex table required")
    side = 2 * coarse.shape[0] - 1
    fine = np.empty((side, side, 2), dtype=np.float64)
    fine[::2, ::2] = coarse
    fine[::2, 1::2] = (coarse[:, :-1] + coarse[:, 1:]) / 2
    fine[1::2, ::2] = (coarse[:-1] + coarse[1:]) / 2
    # Each old square is split along its SW--NE diagonal.
    fine[1::2, 1::2] = (coarse[:-1, :-1] + coarse[1:, 1:]) / 2
    return fine


def analyze(root: Path, model_output: Path, cases: list[int], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("distinct nonempty case IDs required")
    rows = []
    for case in cases:
        with np.load(root / f"{case}_DHR_physical_full_teacher_affine.npz",
                     allow_pickle=False) as archive:
            teacher = np.asarray(archive["raw_teacher_vertices"][0], dtype=np.float64)
        with np.load(model_output / f"{case}_actual_safe_q1.npz",
                     allow_pickle=False) as archive:
            reference = archive["boundary_reference"][0]
            candidate = archive["vertices"][0]
            matrix = archive["post_affine_matrix"]
            offset = archive["post_affine_offset"]
        if teacher.shape != (257, 257, 2):
            raise ValueError("saved 257² teacher required")
        detail = teacher - p1_refine(teacher[::2, ::2])
        fine_vector_rmse = float(np.sqrt(np.mean(np.sum(detail ** 2, axis=-1))))
        fit = boundary_floor(reference, matrix, offset, teacher, candidate)
        full_vector_rmse = fit["candidate_full_vector_rmse"]
        rows.append({
            "case": case,
            "teacher_fine_detail_vector_rmse": fine_vector_rmse,
            "candidate_full_vector_rmse": full_vector_rmse,
            "fine_detail_over_candidate_error": fine_vector_rmse / full_vector_rmse,
            "maximum_teacher_fine_detail_vector": float(
                np.linalg.norm(detail, axis=-1).max()),
        })
    report = {
        "question": "DHR teacher detail not represented by its 129² even-index P1 table",
        "construction": "257² teacher minus exact SW-NE P1 refinement of its 129² even-index subgrid",
        "metric": "Euclidean-vector RMSE over all 257² vertices",
        "scope": "diagnostic, not an optimal 129² approximation bound or anatomical truth",
        "cases": rows,
        "mean_case_teacher_fine_detail_vector_rmse": float(np.mean([
            row["teacher_fine_detail_vector_rmse"] for row in rows])),
        "mean_case_fine_detail_over_candidate_error": float(np.mean([
            row["fine_detail_over_candidate_error"] for row in rows])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.model_output, args.cases, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))


if __name__ == "__main__":
    main()
