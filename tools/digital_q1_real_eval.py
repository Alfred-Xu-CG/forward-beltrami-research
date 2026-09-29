"""Read-only landmark diagnostic of a saved fixed->moving Q1 vertex map.

The registration code never calls this file. Input landmarks are interpreted
as continuous coordinates of pixel centers in each supplied JPEG. The map was
fit after independently resizing each entire JPEG to a unit square.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_dhr_field_eval import (
    _cell_bounds,
    _landmarks,
    _map_nodes,
    invert_q1_at_point,
)


def pixel_to_unit(xy: np.ndarray, size_xy: tuple[int, int]) -> np.ndarray:
    """Continuous JPEG pixel-center coordinate to the unit-square frame."""
    return (np.asarray(xy, dtype=np.float64) + .5) / np.asarray(size_xy, dtype=np.float64)


def unit_to_pixel(xy: np.ndarray, size_xy: tuple[int, int]) -> np.ndarray:
    return np.asarray(xy, dtype=np.float64) * np.asarray(size_xy, dtype=np.float64) - .5


def vertex_map_as_displacement(vertices: np.ndarray) -> np.ndarray:
    """Express a unit-square Q1 map as pixel-index displacement for the inverter."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[-1] != 2 or vertices.shape[0] != vertices.shape[1]:
        raise ValueError("expected square vertex table (N,N,2)")
    side = vertices.shape[0]
    if side < 2 or not np.all(np.isfinite(vertices)):
        raise ValueError("vertex table must be finite and at least 2 by 2")
    row, column = np.indices((side, side), dtype=np.float64)
    return np.stack((vertices[..., 0] * (side - 1) - column,
                     vertices[..., 1] * (side - 1) - row))


def prepare_vertex_map(vertices: np.ndarray) -> tuple[np.ndarray, tuple]:
    field = vertex_map_as_displacement(vertices)
    fx, fy = _map_nodes(field)
    return field, (fx, fy, _cell_bounds(fx, fy))


def invert_vertex_map_at_unit_point(
    vertices: np.ndarray, moving_unit_xy: np.ndarray, *,
    prepared: tuple[np.ndarray, tuple] | None = None,
) -> list[dict]:
    if prepared is None:
        prepared = prepare_vertex_map(vertices)
    field, geometry = prepared
    side = field.shape[-1]
    roots = invert_q1_at_point(
        field, np.asarray(moving_unit_xy, dtype=np.float64) * (side - 1),
        geometry=geometry, tolerance=1e-6,
    )
    return [{"fixed_unit_xy": (np.asarray(root["fixed_xy"]) / (side - 1)).tolist(),
             "residual_unit": root["residual_px"] / (side - 1),
             "local_jacobian_det": root["local_jacobian_det"],
             "cell_xy": root["cell_xy"]} for root in roots]


def load_effective_vertices(map_path: Path) -> tuple[np.ndarray, dict]:
    """Interpret a saved Q1 residual, optionally followed by exact affine data.

    The returned float64 table is only for numerical landmark inversion. A
    composite certificate refers to the stored residual table plus affine
    parameters interpreted as exact binary rationals, not to this newly
    rounded materialization of their composition.
    """
    certificate = certify_q1_binary_map(map_path)
    with np.load(map_path) as archive:
        vertices = archive["vertices"]
        affine = archive["post_affine_matrix"] if "post_affine_matrix" in archive else None
        offset = archive["post_affine_offset"] if "post_affine_offset" in archive else None
    if vertices.ndim != 4 or vertices.shape[0] != 1:
        raise ValueError("expected a saved B=1 Q1 map")
    if (affine is None) != (offset is None):
        raise ValueError("affine matrix and offset must be stored together")
    report = {
        "map_representation": "Q1 vertex table" if affine is None
                              else "positive-affine postcomposition of Q1 residual",
        "saved_binary_residual_certificate_valid": bool(certificate["valid"]),
        "saved_binary_nonpositive_corners": int(certificate["nonpositive_corners"]),
        "stored_affine_det_positive_exact": None,
    }
    effective = vertices[0].astype(np.float64)
    if affine is not None:
        if affine.shape != (2, 2) or offset.shape != (2,) or (
            not np.all(np.isfinite(affine)) or not np.all(np.isfinite(offset))
        ):
            raise ValueError("post-affine data must be finite 2x2 and 2-vector")
        a, b, c, d = (Fraction.from_float(float(value)) for value in affine.flat)
        positive = a * d - b * c > 0
        report["stored_affine_det_positive_exact"] = bool(positive)
        # A 2x2 point transform needs no BLAS; componentwise arithmetic also
        # avoids conflicting OpenMP runtimes in some Windows NumPy environments.
        transformed_x = (effective[..., 0] * float(affine[0, 0])
                         + effective[..., 1] * float(affine[0, 1])
                         + float(offset[0]))
        transformed_y = (effective[..., 0] * float(affine[1, 0])
                         + effective[..., 1] * float(affine[1, 1])
                         + float(offset[1]))
        effective = np.stack((transformed_x, transformed_y), axis=-1)
    report["composite_representation_valid"] = bool(
        certificate["valid"] and (affine is None or report["stored_affine_det_positive_exact"])
    )
    return effective, report


def evaluate(
    map_path: Path, moving_image: Path, fixed_image: Path,
    moving_landmarks: Path, fixed_landmarks: Path,
    *, landmark_id_policy: str = "require_equal",
) -> dict:
    vertices, certificate = load_effective_vertices(map_path)
    prepared = prepare_vertex_map(vertices)
    with Image.open(moving_image) as image:
        moving_size = image.size
    with Image.open(fixed_image) as image:
        fixed_size = image.size
    moving = _landmarks(moving_landmarks, moving_size)
    fixed = _landmarks(fixed_landmarks, fixed_size)
    if landmark_id_policy not in {"require_equal", "intersection"}:
        raise ValueError("landmark_id_policy must be require_equal or intersection")
    unmatched_fixed_ids = sorted(fixed.keys() - moving.keys())
    unmatched_moving_ids = sorted(moving.keys() - fixed.keys())
    if landmark_id_policy == "require_equal" and (unmatched_fixed_ids or unmatched_moving_ids):
        raise ValueError("moving and fixed landmark IDs disagree")
    matched_ids = sorted(fixed.keys() & moving.keys())
    if not matched_ids:
        raise ValueError("no matched landmarks")
    rows = []
    for name in matched_ids:
        moving_unit = pixel_to_unit(moving[name], moving_size)
        roots = invert_vertex_map_at_unit_point(vertices, moving_unit, prepared=prepared)
        row = {"id": name, "inverse_root_count": len(roots), "inverse_roots": roots}
        if len(roots) == 1:
            prediction = unit_to_pixel(roots[0]["fixed_unit_xy"], fixed_size)
            row["predicted_fixed_xy"] = prediction.tolist()
            row["tre_px"] = float(np.linalg.norm(prediction - fixed[name]))
            row["inside_fixed"] = bool(
                0 <= prediction[0] < fixed_size[0] and 0 <= prediction[1] < fixed_size[1]
            )
        rows.append(row)
    unique = [row for row in rows if row["inverse_root_count"] == 1]
    distances = np.asarray([row["tre_px"] for row in unique])
    complete = len(unique) == len(rows)
    return {
        "map_path": str(map_path),
        "map_side": int(vertices.shape[0]),
        "saved_binary_certificate_valid": certificate["composite_representation_valid"],
        "saved_binary_nonpositive_corners": certificate["saved_binary_nonpositive_corners"],
        "map_representation": certificate["map_representation"],
        "stored_affine_det_positive_exact": certificate["stored_affine_det_positive_exact"],
        "moving_size_xy": list(moving_size),
        "fixed_size_xy": list(fixed_size),
        "coordinate_convention": "unit=(original JPEG pixel-center coordinate+0.5)/image size",
        "landmark_id_policy": landmark_id_policy,
        "fixed_landmark_file_count": len(fixed),
        "moving_landmark_file_count": len(moving),
        "unmatched_fixed_ids": unmatched_fixed_ids,
        "unmatched_moving_ids": unmatched_moving_ids,
        "landmark_count": len(rows),
        "unique_inverse_count": len(unique),
        "zero_inverse_count": sum(row["inverse_root_count"] == 0 for row in rows),
        "multiple_inverse_count": sum(row["inverse_root_count"] > 1 for row in rows),
        "inside_fixed_count": sum(row.get("inside_fixed", False) for row in unique),
        "max_inverse_residual_unit": max(
            (root["residual_unit"] for row in unique for root in row["inverse_roots"]),
            default=None,
        ),
        "tre_unit": "original supplied fixed JPEG pixels",
        "mean_tre_px": float(distances.mean()) if complete else None,
        "median_tre_px": float(np.median(distances)) if complete else None,
        "p95_tre_px": float(np.percentile(distances, 95)) if complete else None,
        "max_tre_px": float(distances.max()) if complete else None,
        "landmarks": rows,
    }


def evaluate_unit_identity(
    moving_image: Path, fixed_image: Path,
    moving_landmarks: Path, fixed_landmarks: Path, *,
    landmark_id_policy: str = "require_equal",
) -> dict:
    """Fair identity for independently normalized whole-JPEG rectangles."""
    with Image.open(moving_image) as image:
        moving_size = image.size
    with Image.open(fixed_image) as image:
        fixed_size = image.size
    moving = _landmarks(moving_landmarks, moving_size)
    fixed = _landmarks(fixed_landmarks, fixed_size)
    if landmark_id_policy not in {"require_equal", "intersection"}:
        raise ValueError("landmark_id_policy must be require_equal or intersection")
    unmatched_fixed_ids = sorted(fixed.keys() - moving.keys())
    unmatched_moving_ids = sorted(moving.keys() - fixed.keys())
    if landmark_id_policy == "require_equal" and (unmatched_fixed_ids or unmatched_moving_ids):
        raise ValueError("moving and fixed landmark IDs disagree")
    matched_ids = sorted(fixed.keys() & moving.keys())
    if not matched_ids:
        raise ValueError("no matched landmarks")
    distances = []
    inside = 0
    for name in matched_ids:
        prediction = unit_to_pixel(pixel_to_unit(moving[name], moving_size), fixed_size)
        distances.append(float(np.linalg.norm(prediction - fixed[name])))
        inside += int(0 <= prediction[0] < fixed_size[0] and
                      0 <= prediction[1] < fixed_size[1])
    values = np.asarray(distances)
    return {
        "map_representation": "unit-square identity",
        "moving_size_xy": list(moving_size),
        "fixed_size_xy": list(fixed_size),
        "coordinate_convention": "unit=(original JPEG pixel-center coordinate+0.5)/image size",
        "landmark_id_policy": landmark_id_policy,
        "fixed_landmark_file_count": len(fixed),
        "moving_landmark_file_count": len(moving),
        "unmatched_fixed_ids": unmatched_fixed_ids,
        "unmatched_moving_ids": unmatched_moving_ids,
        "landmark_count": len(matched_ids),
        "inside_fixed_count": inside,
        "tre_unit": "original supplied fixed JPEG pixels",
        "mean_tre_px": float(values.mean()),
        "median_tre_px": float(np.median(values)),
        "p95_tre_px": float(np.percentile(values, 95)),
        "max_tre_px": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--map", type=Path)
    choice.add_argument("--unit-identity", action="store_true")
    for name in ("moving_image", "fixed_image", "moving_landmarks", "fixed_landmarks"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--landmark-id-policy", choices=("require_equal", "intersection"),
                        default="require_equal")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.unit_identity:
        result = evaluate_unit_identity(
            args.moving_image, args.fixed_image,
            args.moving_landmarks, args.fixed_landmarks,
            landmark_id_policy=args.landmark_id_policy,
        )
    else:
        result = evaluate(
            args.map, args.moving_image, args.fixed_image,
            args.moving_landmarks, args.fixed_landmarks,
            landmark_id_policy=args.landmark_id_policy,
        )
    encoded = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
