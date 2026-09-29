"""Audit DHR pseudo-field compatibility with an image-only affine and fixed-boundary P1 head.

No anatomical landmarks are read. Full-DHR fields are used strictly as
training-only pseudo-label candidates, never as inference input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def boundary_indices(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask.reshape(-1)


def audit(root: Path, case_ids: list[int]) -> dict:
    rows = []
    for case in case_ids:
        aff = root / f"{case}_directSG_affine.npz"
        teacher = root / f"{case}_DHR_physical_full_teacher_affine.npz"
        if not aff.exists() or not teacher.exists():
            rows.append({"case": case, "status": "missing_input"})
            continue
        with np.load(aff) as a, np.load(teacher) as t:
            matrix = np.asarray(a["post_affine_matrix"], dtype=np.float64)
            offset = np.asarray(a["post_affine_offset"], dtype=np.float64)
            raw = np.asarray(t["raw_teacher_vertices"], dtype=np.float64)
        if raw.shape != (1, 257, 257, 2):
            raise ValueError(f"unexpected teacher shape for {case}: {raw.shape}")
        side = 257
        axis = np.linspace(0., 1., side)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        q = np.stack((xx, yy), -1).reshape(-1, 2)
        raw = raw.reshape(-1, 2)
        u = np.linalg.solve(matrix, (raw - offset).T).T
        boundary = boundary_indices(side)
        basis = np.column_stack((q[boundary], np.ones(boundary.sum())))
        coeff = np.linalg.lstsq(basis, u[boundary], rcond=None)[0]
        correction = coeff[:2].T
        shift = coeff[2]
        affine_fit = q @ correction.T + shift
        v = np.linalg.solve(correction, (u - shift).T).T
        displacement = u - q
        residual = v - q
        rows.append({
            "case": case, "status": "ok",
            "direct_affine_det": float(np.linalg.det(matrix)),
            "boundary_fit_affine_det": float(np.linalg.det(correction)),
            "unfactored_target_boundary_rmse_unit": float(
                np.sqrt(np.mean(np.sum(displacement[boundary] ** 2, axis=1)))),
            "unfactored_target_all_rmse_unit": float(
                np.sqrt(np.mean(np.sum(displacement ** 2, axis=1)))),
            "boundary_affine_correction_rmse_unit": float(
                np.sqrt(np.mean(np.sum((u[boundary] - affine_fit[boundary]) ** 2, axis=1)))),
            "factorized_target_boundary_rmse_unit": float(
                np.sqrt(np.mean(np.sum(residual[boundary] ** 2, axis=1)))),
            "factorized_target_all_rmse_unit": float(
                np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
            "factorized_target_extrema": [float(v.min()), float(v.max())],
            "boundary_correction_matrix": correction.tolist(),
            "boundary_correction_offset": shift.tolist(),
        })
    good = [r for r in rows if r["status"] == "ok"]
    metrics = (
        "unfactored_target_boundary_rmse_unit",
        "unfactored_target_all_rmse_unit",
        "boundary_affine_correction_rmse_unit",
        "factorized_target_boundary_rmse_unit",
        "factorized_target_all_rmse_unit",
    )
    return {
        "method": "image-only affine versus DHR pseudo-field compatibility audit",
        "case_count": len(case_ids), "success_count": len(good),
        "summary": {m: {
            "median": float(np.median([r[m] for r in good])),
            "mean": float(np.mean([r[m] for r in good])),
            "max": float(np.max([r[m] for r in good])),
        } for m in metrics},
        "positive_boundary_correction_count": sum(
            r["boundary_fit_affine_det"] > 0 for r in good),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = audit(args.root, selection["combined_train_ids"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "case_count", "success_count", "summary",
        "positive_boundary_correction_count",
    )}))


if __name__ == "__main__":
    main()
