"""Independent shared-edge audit for anisotropic P1 cotangent conditions.

This script intentionally does not call the production FEM or mesh code.  It
constructs two triangles sharing one edge, computes the off-diagonal stiffness
entry from barycentric gradients, and compares it with the metric-cotangent
formula.  The output is a compact JSON receipt suitable for the Phase III
research log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _det1_tensor(rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(-np.pi, np.pi))
    lam = float(np.exp(rng.uniform(-2.0, 2.0)))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([lam, 1.0 / lam]) @ R.T


def _spd_tensor(rng: np.random.Generator) -> np.ndarray:
    """General SPD tensor, included to expose determinant-weight effects."""
    angle = float(rng.uniform(-np.pi, np.pi))
    eigs = np.exp(rng.uniform(-2.0, 2.0, size=2))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag(eigs) @ R.T


def _area2(tri: np.ndarray) -> float:
    return float(np.cross(tri[1] - tri[0], tri[2] - tri[0]))


def _gradients(tri: np.ndarray) -> np.ndarray:
    area2 = _area2(tri)
    if area2 <= 1.0e-10:
        raise ValueError("degenerate triangle")
    out = np.empty((3, 2), dtype=float)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        out[i] = np.array(
            [tri[j, 1] - tri[k, 1], tri[k, 0] - tri[j, 0]], dtype=float
        ) / area2
    return out


def _metric_angle(opposite: np.ndarray, a: np.ndarray, b: np.ndarray, A: np.ndarray) -> float:
    vals, vecs = np.linalg.eigh(A)
    Ainvhalf = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    x = Ainvhalf @ (a - opposite)
    y = Ainvhalf @ (b - opposite)
    cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _sample_quad(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # p0--p1 is the shared horizontal edge; p2 is above and p3 below it.
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([rng.uniform(-0.8, 1.8), rng.uniform(0.08, 2.8)])
    p3 = np.array([rng.uniform(-0.8, 1.8), -rng.uniform(0.08, 2.8)])
    return np.vstack([p0, p1, p2]), np.vstack([p1, p0, p3])


def _one(tri1: np.ndarray, tri2: np.ndarray, A1: np.ndarray, A2: np.ndarray) -> dict[str, float]:
    g1, g2 = _gradients(tri1), _gradients(tri2)
    # In both oriented triangles the shared edge is local edge (0, 1).
    # _area2 is twice the Euclidean area, whereas the element matrix uses |T|.
    direct = float(0.5 * abs(_area2(tri1)) * g1[0] @ A1 @ g1[1])
    direct += float(0.5 * abs(_area2(tri2)) * g2[0] @ A2 @ g2[1])
    th1 = _metric_angle(tri1[2], tri1[0], tri1[1], A1)
    th2 = _metric_angle(tri2[2], tri2[0], tri2[1], A2)
    weighted = float(np.sqrt(np.linalg.det(A1)) / 2.0 * 1.0 / np.tan(th1))
    weighted += float(np.sqrt(np.linalg.det(A2)) / 2.0 * 1.0 / np.tan(th2))
    # K_01 = - weighted.  Return the unscaled sum too for a clear sign test.
    return {
        "direct_k01": direct,
        "formula_k01": -weighted,
        "theta1": th1,
        "theta2": th2,
        "weighted_cot_sum": 2.0 * weighted,
        "unweighted_angle_sum": th1 + th2,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    max_formula_error = 0.0
    same_condition_mismatches = 0
    hetero_condition_mismatches = 0
    general_hetero_condition_mismatches = 0
    local_obtuse_but_edge_good = 0
    hetero_unweighted_misclassifications = 0
    general_hetero_unweighted_misclassifications = 0
    same_samples = 0
    hetero_samples = 0
    general_hetero_samples = 0
    example = None
    for _ in range(args.samples):
        tri1, tri2 = _sample_quad(rng)
        same = _det1_tensor(rng)
        hetero = _det1_tensor(rng)
        general = _spd_tensor(rng)
        for A1, A2, kind in (
            (same, same, "same"),
            (same, hetero, "hetero_qc"),
            (same, general, "hetero_general"),
        ):
            try:
                row = _one(tri1, tri2, A1, A2)
            except ValueError:
                continue
            max_formula_error = max(max_formula_error, abs(row["direct_k01"] - row["formula_k01"]))
            good = row["direct_k01"] <= 1.0e-10
            if kind == "same":
                same_samples += 1
                condition_good = row["unweighted_angle_sum"] <= np.pi + 1.0e-10
                same_condition_mismatches += int(good != condition_good)
                local_good = row["theta1"] <= np.pi / 2 + 1.0e-10 and row["theta2"] <= np.pi / 2 + 1.0e-10
                if condition_good and not local_good:
                    local_obtuse_but_edge_good += 1
                    if example is None:
                        example = row
            else:
                if kind == "hetero_general":
                    general_hetero_samples += 1
                else:
                    hetero_samples += 1
                condition_good = row["weighted_cot_sum"] >= -2.0e-10
                if kind == "hetero_general":
                    general_hetero_condition_mismatches += int(good != condition_good)
                else:
                    hetero_condition_mismatches += int(good != condition_good)
                naive = row["unweighted_angle_sum"] <= np.pi + 1.0e-10
                if kind == "hetero_general":
                    general_hetero_unweighted_misclassifications += int(naive != condition_good)
                else:
                    hetero_unweighted_misclassifications += int(naive != condition_good)
    result = {
        "seed": args.seed,
        "samples_requested": args.samples,
        "same_tensor_samples": same_samples,
        "heterogeneous_tensor_samples": hetero_samples,
        "general_heterogeneous_tensor_samples": general_hetero_samples,
        "max_abs_direct_minus_cotangent_stiffness": max_formula_error,
        "same_metric_delaunay_sign_mismatches": same_condition_mismatches,
        "heterogeneous_weighted_cot_sign_mismatches": hetero_condition_mismatches,
        "general_heterogeneous_weighted_cot_sign_mismatches": general_hetero_condition_mismatches,
        "local_obtuse_but_edge_good_count": local_obtuse_but_edge_good,
        "heterogeneous_unweighted_angle_misclassifications": hetero_unweighted_misclassifications,
        "general_heterogeneous_unweighted_angle_misclassifications": general_hetero_unweighted_misclassifications,
        "local_obtuse_example": example,
        "formula": "K01 = -0.5*(sqrt(det(A1))*cot(theta1)+sqrt(det(A2))*cot(theta2))",
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
