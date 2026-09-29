"""Read-only landmark diagnostic of the fixed-diagonal P1 view of a Q1 archive.

The same stored vertex table and positive output affine are interpreted with
piecewise-affine triangles rather than bilinear Q1 cell interpolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.digital_dhr_field_eval import _landmarks
from tools.digital_q1_real_eval import (
    load_effective_vertices, pixel_to_unit, unit_to_pixel,
)


def _cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def p1_map_at_unit_queries(vertices: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Evaluate the SW-NE fixed-diagonal triangle map at K normalized points."""
    vertices = np.asarray(vertices, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[-1] != 2 or (
        vertices.shape[0] != vertices.shape[1] or vertices.shape[0] < 2 or
        queries.ndim != 2 or queries.shape[1] != 2 or
        not np.isfinite(vertices).all() or not np.isfinite(queries).all() or
        np.any((queries < 0) | (queries > 1))
    ):
        raise ValueError("finite square map and Kx2 in-domain unit queries required")
    cells = vertices.shape[0] - 1
    scaled = queries * cells
    indices = np.minimum(np.floor(scaled).astype(np.int64), cells - 1)
    x_index, y_index = indices[:, 0], indices[:, 1]
    s, t = scaled[:, 0] - x_index, scaled[:, 1] - y_index
    a = vertices[y_index, x_index]
    b = vertices[y_index, x_index + 1]
    c = vertices[y_index + 1, x_index + 1]
    d = vertices[y_index + 1, x_index]
    lower = a + s[:, None] * (b - a) + t[:, None] * (c - b)
    upper = a + s[:, None] * (c - d) + t[:, None] * (d - a)
    return np.where((t <= s)[:, None], lower, upper)


def prepare_p1(vertices: np.ndarray) -> dict[str, np.ndarray]:
    """Construct source/target triangle arrays and target bounding boxes."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[0] != vertices.shape[1] or (
        vertices.shape[-1] != 2 or vertices.shape[0] < 2 or
        not np.isfinite(vertices).all()
    ):
        raise ValueError("finite square NxNx2 vertex table required")
    cells = vertices.shape[0] - 1
    row, col = np.indices((cells, cells))
    a0 = np.stack((col, row), axis=-1).astype(np.float64) / cells
    b0 = a0 + np.array([1 / cells, 0.])
    c0 = a0 + np.array([1 / cells, 1 / cells])
    d0 = a0 + np.array([0., 1 / cells])
    a, b = vertices[:-1, :-1], vertices[:-1, 1:]
    c, d = vertices[1:, 1:], vertices[1:, :-1]
    lower_source = np.stack((a0, b0, c0), axis=-2).reshape(-1, 3, 2)
    upper_source = np.stack((a0, c0, d0), axis=-2).reshape(-1, 3, 2)
    lower_target = np.stack((a, b, c), axis=-2).reshape(-1, 3, 2)
    upper_target = np.stack((a, c, d), axis=-2).reshape(-1, 3, 2)
    source = np.concatenate((lower_source, upper_source))
    target = np.concatenate((lower_target, upper_target))
    determinant = _cross(target[:, 1] - target[:, 0],
                         target[:, 2] - target[:, 0])
    if not np.all(determinant > 0):
        raise ValueError("P1 triangle with nonpositive orientation")
    return {
        "source": source,
        "target": target,
        "lower_bounds": target.min(axis=1),
        "upper_bounds": target.max(axis=1),
        "determinant": determinant,
    }


def invert_p1_at_unit_point(geometry: dict[str, np.ndarray],
                            target_point: np.ndarray,
                            *, tolerance: float = 1e-9) -> list[np.ndarray]:
    """Return distinct fixed-unit preimages under the prepared P1 map."""
    point = np.asarray(target_point, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("finite 2-vector target required")
    candidate = np.flatnonzero(np.all(
        (geometry["lower_bounds"] - tolerance <= point) &
        (point <= geometry["upper_bounds"] + tolerance), axis=1,
    ))
    if len(candidate) == 0:
        return []
    triangle = geometry["target"][candidate]
    first = triangle[:, 1] - triangle[:, 0]
    second = triangle[:, 2] - triangle[:, 0]
    remainder = point - triangle[:, 0]
    denominator = geometry["determinant"][candidate]
    lambda_1 = _cross(remainder, second) / denominator
    lambda_2 = _cross(first, remainder) / denominator
    lambda_0 = 1 - lambda_1 - lambda_2
    inside = (np.minimum.reduce((lambda_0, lambda_1, lambda_2)) >= -tolerance)
    valid = candidate[inside]
    if len(valid) == 0:
        return []
    source = geometry["source"][valid]
    weights = np.stack((lambda_0[inside], lambda_1[inside], lambda_2[inside]), axis=-1)
    solutions = np.einsum("nt,ntk->nk", weights, source)
    distinct: list[np.ndarray] = []
    for solution in solutions:
        if not any(np.linalg.norm(solution - prior) <= 1e-8 for prior in distinct):
            distinct.append(solution)
    return distinct


def evaluate(
    map_path: Path, moving_image: Path, fixed_image: Path,
    moving_landmarks: Path, fixed_landmarks: Path, *,
    landmark_id_policy: str = "require_equal",
) -> dict:
    vertices, saved_certificate = load_effective_vertices(map_path)
    if not saved_certificate["composite_representation_valid"]:
        raise ValueError("stored Q1/affine representation lacks a valid certificate")
    geometry = prepare_p1(vertices)
    with Image.open(moving_image) as image:
        moving_size = image.size
    with Image.open(fixed_image) as image:
        fixed_size = image.size
    moving = _landmarks(moving_landmarks, moving_size)
    fixed = _landmarks(fixed_landmarks, fixed_size)
    if landmark_id_policy not in {"require_equal", "intersection"}:
        raise ValueError("invalid landmark-id policy")
    unmatched_fixed = sorted(fixed.keys() - moving.keys())
    unmatched_moving = sorted(moving.keys() - fixed.keys())
    if landmark_id_policy == "require_equal" and (unmatched_fixed or unmatched_moving):
        raise ValueError("landmark IDs disagree")
    names = sorted(fixed.keys() & moving.keys())
    if not names:
        raise ValueError("no corresponding landmarks")
    rows = []
    for name in names:
        query = pixel_to_unit(moving[name], moving_size)
        roots = invert_p1_at_unit_point(geometry, query)
        row = {"id": name, "inverse_root_count": len(roots)}
        if len(roots) == 1:
            predicted = unit_to_pixel(roots[0], fixed_size)
            row["predicted_fixed_xy"] = predicted.tolist()
            row["tre_px"] = float(np.linalg.norm(predicted - fixed[name]))
        rows.append(row)
    complete = all(row["inverse_root_count"] == 1 for row in rows)
    values = np.asarray([row["tre_px"] for row in rows if "tre_px" in row])
    return {
        "interpolation": "P1 SW-NE fixed diagonal on saved Q1 vertex table",
        "saved_map": str(map_path),
        "saved_factorization_valid": saved_certificate["composite_representation_valid"],
        "stored_residual_nonpositive_Q1_corners": saved_certificate[
            "saved_binary_nonpositive_corners"],
        "all_P1_triangle_signs_positive_from_Q1_certificate": True,
        "control_side": int(vertices.shape[0]),
        "triangle_count": int(len(geometry["target"])),
        "moving_size_xy": list(moving_size),
        "fixed_size_xy": list(fixed_size),
        "landmark_id_policy": landmark_id_policy,
        "unmatched_fixed_ids": unmatched_fixed,
        "unmatched_moving_ids": unmatched_moving,
        "landmark_count": len(rows),
        "unique_inverse_count": sum(row["inverse_root_count"] == 1 for row in rows),
        "complete_inverse_coverage": complete,
        "mean_tre_px": float(values.mean()) if complete else None,
        "median_tre_px": float(np.median(values)) if complete else None,
        "p95_tre_px": float(np.percentile(values, 95)) if complete else None,
        "max_tre_px": float(values.max()) if complete else None,
        "landmarks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    for name in ("moving_image", "fixed_image", "moving_landmarks", "fixed_landmarks"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--landmark-id-policy", choices=("require_equal", "intersection"),
                        default="require_equal")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.map, args.moving_image, args.fixed_image,
                      args.moving_landmarks, args.fixed_landmarks,
                      landmark_id_policy=args.landmark_id_policy)
    encoded = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
