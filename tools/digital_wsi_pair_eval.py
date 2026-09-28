"""Independent landmark evaluator for a saved moving-to-fixed affine map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_landmarks(path: str | Path, image_size_xy: tuple[int, int]) -> dict[str, np.ndarray]:
    width, height = image_size_xy
    points: dict[str, np.ndarray] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"X", "Y"}.issubset(reader.fieldnames):
            raise ValueError(f"landmark CSV must have X,Y columns: {path}")
        id_columns = [name for name in reader.fieldnames if name not in {"X", "Y"}]
        if len(id_columns) != 1:
            raise ValueError(f"landmark CSV must have one ID column: {path}")
        id_column = id_columns[0]
        for row in reader:
            landmark_id = row[id_column].strip()
            if not landmark_id:
                raise ValueError(f"empty landmark ID in {path}")
            if landmark_id in points:
                raise ValueError(f"duplicate landmark ID {landmark_id} in {path}")
            xy = np.array([float(row["X"]), float(row["Y"])], dtype=np.float64)
            if not np.all(np.isfinite(xy)):
                raise ValueError(f"nonfinite landmark coordinate for ID {landmark_id}")
            if not (0 <= xy[0] < width and 0 <= xy[1] < height):
                raise ValueError(f"landmark ID {landmark_id} outside {width}x{height} image")
            points[landmark_id] = xy
    if not points:
        raise ValueError(f"no landmarks in {path}")
    return points


def _validated_transform(path: str | Path) -> tuple[dict, np.ndarray]:
    transform = json.loads(Path(path).read_text(encoding="utf-8"))
    if transform.get("direction") != "moving_to_fixed":
        raise ValueError("transform direction must be moving_to_fixed")
    if transform.get("coordinate_frame") != "top_left_origin_pixel_xy":
        raise ValueError("transform must use top-left-origin pixel xy coordinates")
    matrix = np.asarray(transform["matrix_moving_to_fixed_xy"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("affine matrix must be finite 3x3")
    if not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("affine matrix must have homogeneous bottom row [0,0,1]")
    determinant = float(np.linalg.det(matrix[:2, :2]))
    if determinant <= 0:
        raise ValueError("affine linear determinant must be positive")
    if not np.isclose(determinant, transform["det_linear"], rtol=1e-12, atol=1e-12):
        raise ValueError("recorded affine determinant disagrees with matrix")
    for role in ("moving", "fixed"):
        with Image.open(transform[f"{role}_image"]) as image:
            actual_size = list(image.size)
        if actual_size != transform[f"{role}_size_xy"]:
            raise ValueError(f"{role} image dimensions changed since transform creation")
    return transform, matrix


def _pair_key(path: str | Path) -> str:
    stem = Path(path).stem
    for prefix in ("Images_", "Landmarks_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def evaluate_transform(
    transform_path: str | Path,
    moving_landmarks: str | Path,
    fixed_landmarks: str | Path,
) -> dict:
    transform, matrix = _validated_transform(transform_path)
    if _pair_key(moving_landmarks) != _pair_key(transform["moving_image"]):
        raise ValueError("moving landmark file role does not match moving image")
    if _pair_key(fixed_landmarks) != _pair_key(transform["fixed_image"]):
        raise ValueError("fixed landmark file role does not match fixed image")
    moving = read_landmarks(moving_landmarks, tuple(transform["moving_size_xy"]))
    fixed = read_landmarks(fixed_landmarks, tuple(transform["fixed_size_xy"]))
    if moving.keys() != fixed.keys():
        raise ValueError("moving and fixed landmark ID sets differ")
    ids = sorted(moving)
    source_xy = np.stack([moving[landmark_id] for landmark_id in ids])
    target_xy = np.stack([fixed[landmark_id] for landmark_id in ids])
    mapped_xy = np.empty_like(source_xy)
    mapped_xy[:, 0] = matrix[0, 0] * source_xy[:, 0] + matrix[0, 1] * source_xy[:, 1] + matrix[0, 2]
    mapped_xy[:, 1] = matrix[1, 0] * source_xy[:, 0] + matrix[1, 1] * source_xy[:, 1] + matrix[1, 2]
    delta = mapped_xy - target_xy
    distances = np.hypot(delta[:, 0], delta[:, 1])
    width, height = transform["fixed_size_xy"]
    inside = (mapped_xy[:, 0] >= 0) & (mapped_xy[:, 0] < width) & (mapped_xy[:, 1] >= 0) & (mapped_xy[:, 1] < height)
    return {
        "method": transform["method"],
        "direction": transform["direction"],
        "unit": "pixel",
        "n_landmarks": len(ids),
        "mapped_inside_fixed_count": int(inside.sum()),
        "mean_tre_px": float(distances.mean()),
        "median_tre_px": float(np.median(distances)),
        "p95_tre_px": float(np.percentile(distances, 95)),
        "max_tre_px": float(distances.max()),
        "det_linear": float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]),
        "moving_size_xy": transform["moving_size_xy"],
        "fixed_size_xy": transform["fixed_size_xy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--moving-landmarks", required=True)
    parser.add_argument("--fixed-landmarks", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_transform(args.transform, args.moving_landmarks, args.fixed_landmarks), indent=2))


if __name__ == "__main__":
    main()
