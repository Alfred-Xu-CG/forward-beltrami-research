"""Read-only landmark scoring of already saved BIRL development predictions.

No image/encoder/optimizer is loaded here. The predicted map goes from the
fixed 512-square canvas to the moving 512-square canvas. Paired original JPEG
landmarks are transformed by saved resize/pad metadata and scored in moving
original-image pixel coordinates. This is not the official ANHIR protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.digital_dhr_field_eval import _landmarks
from tools.digital_q1_real_eval import load_effective_vertices


def original_pixel_to_canvas_unit(xy: np.ndarray, layout: dict, side: int) -> np.ndarray:
    scale = np.asarray(layout["effective_original_to_canvas_scale_xy"], dtype=np.float64)
    padding = np.asarray(layout["padding_xy"], dtype=np.float64)
    if np.any(scale <= 0):
        raise ValueError("positive resize scales required")
    return ((np.asarray(xy, dtype=np.float64) + .5) * scale + padding) / side


def canvas_unit_to_original_pixel(unit: np.ndarray, layout: dict, side: int) -> np.ndarray:
    scale = np.asarray(layout["effective_original_to_canvas_scale_xy"], dtype=np.float64)
    padding = np.asarray(layout["padding_xy"], dtype=np.float64)
    if np.any(scale <= 0):
        raise ValueError("positive resize scales required")
    return (np.asarray(unit, dtype=np.float64) * side - padding) / scale - .5


def q1_at_queries(vertices: np.ndarray, unit_xy: np.ndarray) -> np.ndarray:
    """Evaluate a saved Q1 vertex table at K unit-square fixed queries."""
    vertices = np.asarray(vertices, dtype=np.float64)
    queries = np.asarray(unit_xy, dtype=np.float64)
    if (vertices.ndim != 3 or vertices.shape[0] != vertices.shape[1]
            or vertices.shape[-1] != 2 or vertices.shape[0] < 2
            or queries.ndim != 2 or queries.shape[-1] != 2
            or not np.isfinite(vertices).all() or not np.isfinite(queries).all()
            or np.any((queries < 0) | (queries > 1))):
        raise ValueError("finite square map and Kx2 unit queries required")
    cells = vertices.shape[0] - 1
    scaled = queries * cells
    ij = np.minimum(np.floor(scaled).astype(np.int64), cells - 1)
    x, y = ij[:, 0], ij[:, 1]
    s, t = scaled[:, 0] - x, scaled[:, 1] - y
    a, b = vertices[y, x], vertices[y, x + 1]
    c, d = vertices[y + 1, x + 1], vertices[y + 1, x]
    return ((1 - s) * (1 - t))[:, None] * a + (s * (1 - t))[:, None] * b + (
        s * t)[:, None] * c + ((1 - s) * t)[:, None] * d


def p1_at_queries(vertices: np.ndarray, unit_xy: np.ndarray) -> np.ndarray:
    """Evaluate the same table with the fixed top-left to bottom-right diagonal."""
    vertices = np.asarray(vertices, dtype=np.float64)
    queries = np.asarray(unit_xy, dtype=np.float64)
    if (vertices.ndim != 3 or vertices.shape[0] != vertices.shape[1]
            or vertices.shape[-1] != 2 or vertices.shape[0] < 2
            or queries.ndim != 2 or queries.shape[-1] != 2
            or not np.isfinite(vertices).all() or not np.isfinite(queries).all()
            or np.any((queries < 0) | (queries > 1))):
        raise ValueError("finite square map and Kx2 unit queries required")
    cells = vertices.shape[0] - 1
    scaled = queries * cells
    ij = np.minimum(np.floor(scaled).astype(np.int64), cells - 1)
    x, y = ij[:, 0], ij[:, 1]
    s, t = scaled[:, 0] - x, scaled[:, 1] - y
    a, b = vertices[y, x], vertices[y, x + 1]
    c, d = vertices[y + 1, x + 1], vertices[y + 1, x]
    lower = ((1 - s)[:, None] * a + (s - t)[:, None] * b + t[:, None] * c)
    upper = ((1 - t)[:, None] * a + s[:, None] * c + (t - s)[:, None] * d)
    return np.where((t <= s)[:, None], lower, upper)


def score_pair(layout_path: Path, fixed_landmarks: Path, moving_landmarks: Path,
               map_paths: dict[str, Path], initial_affine: Path,
               full_field: Path, full_params: Path, output: Path, *,
               include_full_dhr: bool = True,
               interpolation: str = "q1") -> dict:
    if output.exists():
        raise FileExistsError("scoring report must not overwrite a prior run")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    side = int(layout["side"])
    if side != 512:
        raise ValueError("this diagnostic expects a 512-square canvas")
    # The registration outputs are read and checked before landmark CSVs.
    maps = {}
    for name, path in map_paths.items():
        vertices, certificate = load_effective_vertices(path)
        if not certificate["composite_representation_valid"]:
            raise ValueError(f"invalid saved factorization: {name}")
        maps[name] = vertices
    with np.load(initial_affine) as data:
        matrix = np.asarray(data["post_affine_matrix"], dtype=np.float64)
        offset = np.asarray(data["post_affine_offset"], dtype=np.float64)
    if (matrix.shape != (2, 2) or offset.shape != (2,)
            or matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0] <= 0):
        raise ValueError("positive 2D affine required")
    if include_full_dhr:
        import SimpleITK as sitk
        field = sitk.GetArrayFromImage(sitk.ReadImage(str(full_field)))
        params = json.loads(full_params.read_text(encoding="utf-8"))
    with Image.open(layout["fixed"]["source"]) as image:
        fixed_size = image.size
    with Image.open(layout["moving"]["source"]) as image:
        moving_size = image.size
    fixed = _landmarks(fixed_landmarks, fixed_size)
    moving = _landmarks(moving_landmarks, moving_size)
    names = sorted(fixed.keys() & moving.keys())
    if not names:
        raise ValueError("no shared landmark IDs")
    fixed_original = np.stack([fixed[name] for name in names])
    moving_original = np.stack([moving[name] for name in names])
    query = original_pixel_to_canvas_unit(fixed_original, layout["fixed"], side)
    if np.any((query < 0) | (query > 1)):
        raise ValueError("landmark outside fixed canvas")
    if interpolation not in ("q1", "p1"):
        raise ValueError("interpolation must be q1 or p1")
    evaluator = q1_at_queries if interpolation == "q1" else p1_at_queries
    predicted = {name: evaluator(vertices, query) for name, vertices in maps.items()}
    predicted["initial_affine"] = np.stack((
        query[:, 0] * matrix[0, 0] + query[:, 1] * matrix[0, 1] + offset[0],
        query[:, 0] * matrix[1, 0] + query[:, 1] * matrix[1, 1] + offset[1],
    ), axis=-1)
    predicted["canvas_identity"] = query
    if include_full_dhr:
        import torch
        from tools.digital_compare_appearance import dhr_map_at_unit_queries
        full_query = torch.from_numpy(query.astype(np.float32)).reshape(1, 1, -1, 2)
        predicted["full_DHR"] = dhr_map_at_unit_queries(
            field, params, fixed_size=(side, side), moving_size=(side, side),
            query=full_query,
        )[0, 0].cpu().numpy().astype(np.float64)
    results = {}
    for name, canvas_unit in predicted.items():
        original_xy = canvas_unit_to_original_pixel(canvas_unit, layout["moving"], side)
        distances = np.linalg.norm(original_xy - moving_original, axis=1)
        results[name] = {
            "mean_tre_native_moving_px": float(distances.mean()),
            "median_tre_native_moving_px": float(np.median(distances)),
            "p95_tre_native_moving_px": float(np.percentile(distances, 95)),
            "max_tre_native_moving_px": float(distances.max()),
            "outside_moving_image": int(np.count_nonzero(
                (original_xy[:, 0] < -.5) | (original_xy[:, 0] > moving_size[0] - .5)
                | (original_xy[:, 1] < -.5) | (original_xy[:, 1] > moving_size[1] - .5)
            )),
            "per_landmark_tre_px": dict(zip(names, distances.tolist(), strict=True)),
        }
    report = {
        "mode": "read_only_BIRL_5pc_development_landmark_diagnostic",
        "interpolation": interpolation,
        "full_DHR_evaluated": include_full_dhr,
        "scale_assumption": layout["scale_assumption"], "side": side,
        "landmark_count": len(names),
        "fixed_only_ids": sorted(fixed.keys() - moving.keys()),
        "moving_only_ids": sorted(moving.keys() - fixed.keys()),
        "fixed_size_xy": list(fixed_size), "moving_size_xy": list(moving_size),
        "metric": "Euclidean error in original moving JPEG pixels after saved canvas inverse",
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("layout", "fixed_landmarks", "moving_landmarks",
                 "initial_affine", "full_field", "full_params", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ("baseline_actual", "baseline_blank", "augmented_actual",
                 "augmented_blank"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--safe-hybrid", type=Path,
                        help="predeclared image-only full-DHR-field teacher fit")
    args = parser.parse_args()
    maps = {name: getattr(args, name) for name in
            ("baseline_actual", "baseline_blank", "augmented_actual", "augmented_blank")}
    if args.safe_hybrid is not None:
        maps["safe_hybrid"] = args.safe_hybrid
    result = score_pair(args.layout, args.fixed_landmarks, args.moving_landmarks,
                        maps, args.initial_affine, args.full_field,
                        args.full_params, args.output)
    compact = {name: value["mean_tre_native_moving_px"]
               for name, value in result["results"].items()}
    print(json.dumps({"landmark_count": result["landmark_count"], "means": compact}))


if __name__ == "__main__":
    main()
