"""Measure non-affine geometry in unchanged saved lung registration maps.

No annotations, image proxies, or model selection are involved. All pixel
figures use a 512-wide normalized canvas, not the original stain JPEG size.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from tools.digital_lung_all20_score import STAIN_NAME
from tools.digital_q1_real_eval import load_effective_vertices


def project_affine(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the least-squares affine projection and orthogonal residual."""
    if vertices.ndim != 3 or vertices.shape[-1] != 2 or (
        vertices.shape[0] != vertices.shape[1] or vertices.shape[0] < 3
    ):
        raise ValueError("expected a square H x H x 2 vertex table")
    n = vertices.shape[0]
    x, y = np.meshgrid(np.linspace(0., 1., n), np.linspace(0., 1., n))
    centered_x, centered_y = x - .5, y - .5
    target = np.asarray(vertices, dtype=np.float64)
    # On a Cartesian grid, 1, x-.5 and y-.5 are exactly orthogonal under
    # vertex counting measure. This avoids a large dense least-squares call.
    intercept = np.mean(target, axis=(0, 1))
    slope_x = np.sum(centered_x[..., None] * target, axis=(0, 1)) / (
        n * np.sum(centered_x[0] ** 2))
    slope_y = np.sum(centered_y[..., None] * target, axis=(0, 1)) / (
        n * np.sum(centered_y[:, 0] ** 2))
    projected = (intercept + centered_x[..., None] * slope_x +
                 centered_y[..., None] * slope_y)
    return projected, target - projected


def rms_canvas_px(field: np.ndarray) -> float:
    return float(512. * np.sqrt(np.mean(np.sum(field * field, axis=-1))))


def run(predictions: Path) -> dict:
    rows = []
    for fixed, moving in itertools.permutations(STAIN_NAME, 2):
        name = f"{fixed}_to_{moving}"
        maps = {}
        for method, suffix in (("frozen", "frozen"),
                               ("retained", "dynamic")):
            vertices, certificate = load_effective_vertices(
                predictions / f"{name}_{suffix}_safe257.npz")
            if not certificate["composite_representation_valid"]:
                raise ValueError(f"invalid saved map: {name}, {method}")
            maps[method] = vertices
        if maps["frozen"].shape != (257, 257, 2) or (
            maps["retained"].shape != (257, 257, 2)
        ):
            raise ValueError(f"wrong control grid: {name}")
        _, frozen_nonaffine = project_affine(maps["frozen"])
        _, retained_nonaffine = project_affine(maps["retained"])
        increment = maps["retained"] - maps["frozen"]
        increment_affine, increment_nonaffine = project_affine(increment)
        # An affine difference field is fitted as a function of the reference
        # coordinates; the fitted field itself includes translation.
        total_energy = float(np.sum(increment * increment))
        nonaffine_energy = float(np.sum(increment_nonaffine * increment_nonaffine))
        affine_energy = float(np.sum(increment_affine * increment_affine))
        if not np.isclose(total_energy, affine_energy + nonaffine_energy,
                          rtol=1e-9, atol=1e-8):
            raise ArithmeticError(f"affine projection not orthogonal: {name}")
        rows.append({
            "direction": name,
            "frozen_best_affine_residual_rms_canvas_px": rms_canvas_px(
                frozen_nonaffine),
            "retained_best_affine_residual_rms_canvas_px": rms_canvas_px(
                retained_nonaffine),
            "retained_minus_frozen_total_rms_canvas_px": rms_canvas_px(
                increment),
            "retained_minus_frozen_affine_rms_canvas_px": rms_canvas_px(
                increment_affine),
            "retained_minus_frozen_nonaffine_rms_canvas_px": rms_canvas_px(
                increment_nonaffine),
            "retained_minus_frozen_nonaffine_energy_fraction": (
                nonaffine_energy / total_energy if total_energy else 0.),
        })
    keys = tuple(key for key in rows[0] if key != "direction")
    return {
        "question": "does the retained dense map use non-affine spatial geometry?",
        "cohort": "20 correlated directions from one previously viewed lung specimen",
        "grid": "257x257 control vertices, 131072 P1 triangles",
        "metric_unit": "512 normalized-canvas pixels, not native JPEG pixels",
        "method": "float64 least-squares projection onto span{1,x,y} at all vertices",
        "manual_landmarks_read": False,
        "direction_count": len(rows),
        "equal_direction_mean": {
            key: float(np.mean([row[key] for row in rows])) for key in keys
        },
        "median_nonaffine_fraction": float(np.median([
            row["retained_minus_frozen_nonaffine_energy_fraction"]
            for row in rows])),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.predictions)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "direction_count": result["direction_count"],
        "equal_direction_mean": result["equal_direction_mean"],
        "median_nonaffine_fraction": result["median_nonaffine_fraction"],
    }))


if __name__ == "__main__":
    main()
