"""Measure teacher deformation energy in the cell-readout's near-null mode.

This is a linearized readout diagnostic, not an anatomical score or an
optimization of the safe decoder. It reads saved image-only teacher/baseline
maps and never reads human landmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def one_dimensional_readout() -> np.ndarray:
    indices = np.arange(1, 256, dtype=np.float64)
    a = (indices + .5) / 257.
    matrix = np.zeros((255, 256), dtype=np.float64)
    rows = np.arange(255)
    matrix[rows, rows] = a
    matrix[rows, rows + 1] = 1. - a
    return matrix


def near_null_energy(field: np.ndarray, mode: np.ndarray) -> float:
    """Orthogonal projection on mode along at least one spatial axis."""
    left = mode @ field
    right = field @ mode
    overlap = float(left @ mode)
    projection = (np.outer(mode, left) + np.outer(right, mode)
                  - overlap * np.outer(mode, mode))
    return float(np.square(projection).sum())


def audit(training_report: Path, teachers: Path, frozen: Path) -> dict:
    saved = json.loads(training_report.read_text(encoding="utf-8"))
    matrix = one_dimensional_readout()
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    mode = left[:, -1]
    rows = []
    for case in (row["case"] for row in saved["validation_final"]):
        with np.load(teachers / f"{case}_multilevel25_safe257.npz") as data:
            teacher = data["vertices"].astype(np.float64)
            affine = (data["post_affine_matrix"], data["post_affine_offset"])
        with np.load(frozen / f"{case}_old_safe257.npz") as data:
            base = data["vertices"].astype(np.float64)
            if not (np.array_equal(affine[0], data["post_affine_matrix"])
                    and np.array_equal(affine[1], data["post_affine_offset"])):
                raise ValueError(f"affine mismatch for {case}")
        displacement = (teacher - base)[0, 1:-1, 1:-1]
        energy = sum(float(np.square(displacement[..., component]).sum())
                     for component in range(2))
        near = sum(near_null_energy(displacement[..., component], mode)
                   for component in range(2))
        rows.append({"case": case, "relative_near_null_energy": near / energy,
                     "interior_displacement_rms": float(np.sqrt(
                         energy / (255 * 255)))})
    return {"question": "Does the bilinear head's near-null mode carry the "
            "held-out pseudo-teacher displacement?",
            "scope": "18 ACROBAT validation cases; teacher minus saved frozen start; "
            "linearized vertex-displacement projection only",
            "one_dimensional_smallest_binary64_singular": float(singular_values[-1]),
            "one_dimensional_next_singular": float(singular_values[-2]),
            "cases": rows,
            "mean_relative_near_null_energy": float(np.mean(
                [row["relative_near_null_energy"] for row in rows])),
            "maximum_relative_near_null_energy": float(np.max(
                [row["relative_near_null_energy"] for row in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.training_report, args.teachers, args.frozen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "mean_relative_near_null_energy", "maximum_relative_near_null_energy")}))


if __name__ == "__main__":
    main()
