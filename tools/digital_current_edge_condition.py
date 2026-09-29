"""Measure the current-image F1/F2 centered-edge basis on saved residual maps.

This reads map vertices only. External output-side affines are deliberately not
included because the recurrent proposal was applied before that post-affine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def edge_condition(vertices: np.ndarray) -> dict:
    vertices = np.asarray(vertices, dtype=np.float64)
    if (vertices.ndim != 3 or vertices.shape[0] != vertices.shape[1]
            or vertices.shape[0] < 3 or vertices.shape[-1] != 2
            or not np.isfinite(vertices).all()):
        raise ValueError("finite square HxHx2 residual table with H>=3 required")
    side = vertices.shape[0]
    h = 1 / (side - 1)
    horizontal = (vertices[1:-1, 2:] - vertices[1:-1, :-2]) / (2 * h)
    vertical = (vertices[2:, 1:-1] - vertices[:-2, 1:-1]) / (2 * h)
    matrices = np.stack((horizontal, vertical), axis=-1)
    a, b = matrices[..., 0, 0], matrices[..., 0, 1]
    c, d = matrices[..., 1, 0], matrices[..., 1, 1]
    determinants = a * d - b * c
    # Stable closed-form 2x2 singular values, avoiding a BLAS/LAPACK runtime
    # collision when this NumPy diagnostic runs after PyTorch on Windows.
    frobenius_squared = a * a + b * b + c * c + d * d
    absolute_det = np.abs(determinants)
    discriminant = np.maximum(frobenius_squared - 2 * absolute_det, 0) * (
        frobenius_squared + 2 * absolute_det
    )
    sigma_max = np.sqrt((frobenius_squared + np.sqrt(discriminant)) / 2)
    sigma_min = absolute_det / sigma_max
    condition = sigma_max / sigma_min
    deviations = np.linalg.norm(matrices - np.eye(2), axis=(-2, -1))
    horizontal_length = np.linalg.norm(horizontal, axis=-1)
    vertical_length = np.linalg.norm(vertical, axis=-1)
    horizontal_angle = np.degrees(np.arctan2(horizontal[..., 1], horizontal[..., 0]))
    vertical_angle = np.degrees(np.arctan2(-vertical[..., 0], vertical[..., 1]))
    result = {
        "control_side": side,
        "interior_vertices": int((side - 2) ** 2),
        "nonpositive_centered_edge_determinants": int(np.count_nonzero(determinants <= 0)),
        "sigma_min_min": float(sigma_min.min()),
        "sigma_min_median": float(np.median(sigma_min)),
        "sigma_max_median": float(np.median(sigma_max)),
        "sigma_max_max": float(sigma_max.max()),
        "condition_median": float(np.median(condition)),
        "condition_p95": float(np.quantile(condition, .95)),
        "condition_p99": float(np.quantile(condition, .99)),
        "condition_max": float(condition.max()),
        "condition_gt_2_fraction": float(np.mean(condition > 2)),
        "condition_gt_10_fraction": float(np.mean(condition > 10)),
        "frobenius_distance_from_identity_median": float(np.median(deviations)),
        "frobenius_distance_from_identity_p95": float(np.quantile(deviations, .95)),
        "horizontal_length_over_h_p05_p50_p95": [
            float(x) for x in np.quantile(horizontal_length, [.05, .5, .95])],
        "vertical_length_over_h_p05_p50_p95": [
            float(x) for x in np.quantile(vertical_length, [.05, .5, .95])],
        "horizontal_absolute_angle_deg_p95": float(np.quantile(np.abs(horizontal_angle), .95)),
        "vertical_absolute_angle_deg_p95": float(np.quantile(np.abs(vertical_angle), .95)),
    }
    if not all(np.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ArithmeticError("nonfinite centered-edge condition statistic")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True,
                        help="label=directory under root; may repeat")
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output must be new")
    report = {
        "question": "conditioning of the deformed centered-edge proposal basis",
        "formula": "E=[(right-left)/2,(down-up)/2]; singular values and cond(E/h)",
        "frame": "saved residual vertex table, before stored external post-affine",
        "arms": {},
        "scope": "final saved states only; not all recurrent intermediate states or registration accuracy",
    }
    for spec in args.arm:
        if "=" not in spec:
            raise ValueError("arm must be label=directory")
        name, dirname = spec.split("=", 1)
        if name in report["arms"] or not name:
            raise ValueError("unique nonempty arm labels required")
        rows = []
        for case in args.cases:
            path = args.root / dirname / f"{case}_actual_safe_q1.npz"
            with np.load(path, allow_pickle=False) as archive:
                vertices = np.asarray(archive["vertices"])
            if vertices.shape[0] != 1:
                raise ValueError("saved B=1 map required")
            row = edge_condition(vertices[0])
            row.update({"case": case, "map": str(path)})
            rows.append(row)
        report["arms"][name] = {
            "cases": rows,
            "mean_case_condition_p99": float(np.mean([r["condition_p99"] for r in rows])),
            "mean_case_condition_gt_2_fraction": float(np.mean([
                r["condition_gt_2_fraction"] for r in rows])),
            "maximum_case_condition": float(max(r["condition_max"] for r in rows)),
            "minimum_case_sigma_min": float(min(r["sigma_min_min"] for r in rows)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in item.items() if key != "cases"}
                      for name, item in report["arms"].items()}))


if __name__ == "__main__":
    main()
