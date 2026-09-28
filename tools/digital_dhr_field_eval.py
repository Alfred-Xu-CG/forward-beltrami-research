"""Evaluate a saved DeeperHistReg backward field without running registration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def canvas_xy(xy: np.ndarray, scale: float, pad_yx: list[list[int]]) -> np.ndarray:
    """Original pixel centers to the padded, resampled canvas (align_corners=False)."""
    xy = np.asarray(xy, dtype=np.float64)
    return (xy + 0.5) * scale - 0.5 + np.array([pad_yx[1][0], pad_yx[0][0]])


def original_xy(canvas: np.ndarray, scale: float, pad_yx: list[list[int]]) -> np.ndarray:
    canvas = np.asarray(canvas, dtype=np.float64)
    return (canvas - np.array([pad_yx[1][0], pad_yx[0][0]]) + 0.5) / scale - 0.5


def q1_corner_determinants(displacement: np.ndarray) -> np.ndarray:
    """Determinants at top-left, top-right, bottom-left, bottom-right."""
    if displacement.ndim != 3 or displacement.shape[0] != 2:
        raise ValueError("expected x/y component planes with shape (2,H,W)")
    h, w = displacement.shape[1:]
    if h < 2 or w < 2 or not np.all(np.isfinite(displacement)):
        raise ValueError("field must be finite and at least 2 by 2")
    yy, xx = np.indices((h, w), dtype=np.float64)
    fx = xx + displacement[0]
    fy = yy + displacement[1]
    ax = fx[:-1, 1:] - fx[:-1, :-1]
    ay = fy[:-1, 1:] - fy[:-1, :-1]
    bx = fx[1:, :-1] - fx[:-1, :-1]
    by = fy[1:, :-1] - fy[:-1, :-1]
    cx = fx[1:, 1:] - fx[:-1, 1:] - fx[1:, :-1] + fx[:-1, :-1]
    cy = fy[1:, 1:] - fy[:-1, 1:] - fy[1:, :-1] + fy[:-1, :-1]
    cross = lambda x1, y1, x2, y2: x1 * y2 - y1 * x2
    return np.stack((
        cross(ax, ay, bx, by),
        cross(ax, ay, bx + cx, by + cy),
        cross(ax + cx, ay + cy, bx, by),
        cross(ax + cx, ay + cy, bx + cx, by + cy),
    ))


def _map_nodes(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = field.shape[1:]
    yy, xx = np.indices((h, w), dtype=np.float64)
    return xx + field[0], yy + field[1]


def _cell_bounds(fx: np.ndarray, fy: np.ndarray) -> tuple[np.ndarray, ...]:
    corners_x = (fx[:-1, :-1], fx[:-1, 1:], fx[1:, :-1], fx[1:, 1:])
    corners_y = (fy[:-1, :-1], fy[:-1, 1:], fy[1:, :-1], fy[1:, 1:])
    return (np.minimum.reduce(corners_x), np.maximum.reduce(corners_x),
            np.minimum.reduce(corners_y), np.maximum.reduce(corners_y))


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def invert_q1_at_point(
    field: np.ndarray,
    moving_canvas_xy: np.ndarray,
    *,
    geometry: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]] | None = None,
    tolerance: float = 1e-6,
) -> list[dict]:
    """Enumerate all isolated in-cell inverse roots of a bilinear saved field."""
    if geometry is None:
        fx, fy = _map_nodes(field)
        bounds = _cell_bounds(fx, fy)
    else:
        fx, fy, bounds = geometry
    point = np.asarray(moving_canvas_xy, dtype=np.float64)
    minx, maxx, miny, maxy = bounds
    candidates = np.argwhere((minx - tolerance <= point[0]) & (point[0] <= maxx + tolerance)
                             & (miny - tolerance <= point[1]) & (point[1] <= maxy + tolerance))
    roots: list[dict] = []
    for row, col in candidates:
        p00 = np.array([fx[row, col], fy[row, col]])
        p10 = np.array([fx[row, col + 1], fy[row, col + 1]])
        p01 = np.array([fx[row + 1, col], fy[row + 1, col]])
        p11 = np.array([fx[row + 1, col + 1], fy[row + 1, col + 1]])
        a, b = p10 - p00, p01 - p00
        c = p11 - p10 - p01 + p00
        d = point - p00
        # cross(d-a*s,b+c*s)=0; solve for horizontal local coordinate s.
        coefficients = [-_cross(a, c), _cross(d, c) - _cross(a, b), _cross(d, b)]
        scale = max(1.0, *(abs(v) for v in coefficients))
        if abs(coefficients[0]) <= 1e-13 * scale:
            if abs(coefficients[1]) <= 1e-13 * scale:
                continue
            s_candidates = [-coefficients[2] / coefficients[1]]
        else:
            s_candidates = [z.real for z in np.roots(coefficients) if abs(z.imag) < 1e-10]
        for s in s_candidates:
            if not -tolerance <= s <= 1 + tolerance:
                continue
            v = b + c * s
            denom = float(v @ v)
            if denom <= 1e-20:
                continue
            t = float((d - a * s) @ v / denom)
            if not -tolerance <= t <= 1 + tolerance:
                continue
            reconstructed = p00 + a * s + b * t + c * s * t
            residual = float(np.linalg.norm(reconstructed - point))
            if residual > tolerance:
                continue
            fixed_xy = [float(col + s), float(row + t)]
            if any(np.linalg.norm(np.array(root["fixed_xy"]) - fixed_xy) < 1e-6 for root in roots):
                continue
            local_det = _cross(a + c * t, b + c * s)
            roots.append({"fixed_xy": fixed_xy, "residual_px": residual,
                          "local_jacobian_det": local_det,
                          "cell_xy": [int(col), int(row)]})
    return roots


def _landmarks(path: Path, size_xy: tuple[int, int]) -> dict[str, np.ndarray]:
    points = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"X", "Y"}.issubset(reader.fieldnames):
            raise ValueError(f"landmarks need X,Y columns: {path}")
        ids = [name for name in reader.fieldnames if name not in {"X", "Y"}]
        if len(ids) != 1:
            raise ValueError("landmarks need exactly one ID column")
        for record in reader:
            name = record[ids[0]].strip()
            xy = np.array([float(record["X"]), float(record["Y"])])
            if not name or name in points or not np.all(np.isfinite(xy)):
                raise ValueError(f"invalid or duplicate landmark ID {name}")
            if not (0 <= xy[0] < size_xy[0] and 0 <= xy[1] < size_xy[1]):
                raise ValueError(f"landmark {name} outside image")
            points[name] = xy
    return points


def evaluate(field_path: Path, params_path: Path, config_path: Path,
             moving_image: Path, fixed_image: Path,
             moving_landmarks: Path, fixed_landmarks: Path) -> dict:
    import SimpleITK as sitk

    field = sitk.GetArrayFromImage(sitk.ReadImage(str(field_path))).astype(np.float64)
    params = json.loads(params_path.read_text())
    config = json.loads(config_path.read_text())
    if config["loading_params"]["source_resample_ratio"] != params["source_resample_ratio"]:
        raise ValueError("source resample ratio disagrees")
    if config["loading_params"]["target_resample_ratio"] != params["target_resample_ratio"]:
        raise ValueError("target resample ratio disagrees")
    if params["initial_resample_ratio"] != 1:
        raise ValueError("extra preprocessing resample requires separate coordinate accounting")
    with Image.open(moving_image) as image:
        moving_size = image.size
    with Image.open(fixed_image) as image:
        fixed_size = image.size
    moving = _landmarks(moving_landmarks, moving_size)
    fixed = _landmarks(fixed_landmarks, fixed_size)
    if moving.keys() != fixed.keys():
        raise ValueError("landmark ID sets disagree")
    q1 = q1_corner_determinants(field)
    fx, fy = _map_nodes(field)
    geometry = fx, fy, _cell_bounds(fx, fy)
    scale_m = float(params["source_resample_ratio"])
    scale_f = float(params["target_resample_ratio"])
    rows = []
    for name in sorted(moving):
        source_canvas = canvas_xy(moving[name], scale_m, params["pad_1"])
        roots = invert_q1_at_point(field, source_canvas, geometry=geometry)
        row = {"id": name, "inverse_root_count": len(roots),
               "inverse_roots": roots}
        if len(roots) == 1:
            mapped_original = original_xy(roots[0]["fixed_xy"], scale_f, params["pad_2"])
            row["mapped_fixed_xy"] = mapped_original.tolist()
            row["tre_px"] = float(np.linalg.norm(mapped_original - fixed[name]))
            row["inside_fixed"] = bool(0 <= mapped_original[0] < fixed_size[0]
                                       and 0 <= mapped_original[1] < fixed_size[1])
        rows.append(row)
    unique = [row for row in rows if row["inverse_root_count"] == 1]
    distances = np.array([row["tre_px"] for row in unique])
    residuals = np.array([row["inverse_roots"][0]["residual_px"] for row in unique])
    local_dets = np.array([row["inverse_roots"][0]["local_jacobian_det"] for row in unique])
    result = {
        "field_shape_component_yx": list(field.shape),
        "canvas_size_xy": [int(field.shape[2]), int(field.shape[1])],
        "cell_count": int(q1.shape[1] * q1.shape[2]),
        "q1_corner_order": ["top-left", "top-right", "bottom-left", "bottom-right"],
        "q1_nonpositive_by_corner": [int(np.count_nonzero(x <= 0)) for x in q1],
        "q1_nonpositive_any_cell": int(np.count_nonzero(np.any(q1 <= 0, axis=0))),
        "q1_min_by_corner": [float(np.min(x)) for x in q1],
        "landmark_count": len(rows),
        "unique_inverse_count": len(unique),
        "zero_inverse_count": sum(row["inverse_root_count"] == 0 for row in rows),
        "multiple_inverse_count": sum(row["inverse_root_count"] > 1 for row in rows),
        "inside_fixed_count": sum(row.get("inside_fixed", False) for row in unique),
        "max_inverse_residual_canvas_px": float(residuals.max()) if len(residuals) else None,
        "min_landmark_local_jacobian_det": float(local_dets.min()) if len(local_dets) else None,
        "tre_unit": "original supplied JPEG pixels",
        "mean_tre_px": float(distances.mean()) if len(unique) == len(rows) else None,
        "median_tre_px": float(np.median(distances)) if len(unique) == len(rows) else None,
        "p95_tre_px": float(np.percentile(distances, 95)) if len(unique) == len(rows) else None,
        "max_tre_px": float(distances.max()) if len(unique) == len(rows) else None,
        "landmarks": rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("field", "params", "config", "moving_image", "fixed_image",
                 "moving_landmarks", "fixed_landmarks"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.field, args.params, args.config, args.moving_image, args.fixed_image,
                      args.moving_landmarks, args.fixed_landmarks)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
