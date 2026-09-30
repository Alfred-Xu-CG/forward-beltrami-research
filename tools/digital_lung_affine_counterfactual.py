"""Ask whether saved learned lung gains survive an affine-only post-map.

The positive-affine fit uses all saved vertices and no manual landmarks. The
same already viewed specimen supplies annotations only for evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from tools.digital_birl_landmark_score import (
    canvas_unit_to_original_pixel, original_pixel_to_canvas_unit, p1_at_queries,
)
from tools.digital_lung_all20_score import STAIN_NAME
from tools.digital_lung_lesion3_score import PREFIX, scaled_landmarks
from tools.digital_q1_real_eval import load_effective_vertices


def fit_affine_postmap(source: np.ndarray,
                       target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 2:
        raise ValueError("matching H x W x 2 vertex tables required")
    design = np.column_stack((source.reshape(-1, 2),
                              np.ones(source.shape[0] * source.shape[1])))
    coefficients, _, rank, _ = np.linalg.lstsq(
        design, target.reshape(-1, 2), rcond=None)
    if rank != 3:
        raise ValueError("rank-deficient source table")
    matrix = coefficients[:2].T
    offset = coefficients[2]
    determinant = float(np.linalg.det(matrix))
    residual = design @ coefficients - target.reshape(-1, 2)
    rms = float(512. * np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    return matrix, offset, rms


def run(canvas: Path, annotations: Path, predictions: Path) -> dict:
    layouts, landmarks = {}, {}
    for stain, original in STAIN_NAME.items():
        layout_file = (canvas / "cc10_layout.json" if stain == "he" else
                       canvas / f"{stain}_layout.json")
        key = "fixed" if stain == "he" else "moving"
        layouts[stain] = json.loads(layout_file.read_text(encoding="utf-8"))[key]
        landmarks[stain] = scaled_landmarks(
            annotations / f"{PREFIX}{original}-les3.csv")

    rows = []
    for fixed, moving in itertools.permutations(STAIN_NAME, 2):
        name = f"{fixed}_to_{moving}"
        source, target = landmarks[fixed], landmarks[moving]
        ids = sorted(source.keys() & target.keys(), key=int)
        if len(ids) != 80:
            raise ValueError(f"unexpected landmark count: {name}")
        source_px = np.stack([source[item] for item in ids])
        target_px = np.stack([target[item] for item in ids])
        queries = original_pixel_to_canvas_unit(source_px, layouts[fixed], 512)
        tables = {}
        for method in ("frozen", "dynamic"):
            path = predictions / f"{name}_{method}_safe257.npz"
            _, cert = load_effective_vertices(path)
            if not cert["composite_representation_valid"]:
                raise ValueError(f"invalid saved map: {name} {method}")
            with np.load(path) as archive:
                table = np.asarray(archive["vertices"][0], dtype=np.float64)
                external_matrix = np.asarray(
                    archive["post_affine_matrix"], dtype=np.float64)
                external_offset = np.asarray(
                    archive["post_affine_offset"], dtype=np.float64)
            tables[method] = (table, external_matrix, external_offset)
        frozen, common_matrix, common_offset = tables["frozen"]
        dynamic, dynamic_matrix, dynamic_offset = tables["dynamic"]
        if not (frozen.shape == dynamic.shape == (257, 257, 2) and
                np.array_equal(common_matrix, dynamic_matrix) and
                np.array_equal(common_offset, dynamic_offset) and
                np.linalg.det(common_matrix) > 0):
            raise ValueError(f"non-common frame: {name}")
        affine_matrix, affine_offset, fit_rms = fit_affine_postmap(frozen, dynamic)
        affine_determinant = float(np.linalg.det(affine_matrix))
        if not np.isfinite(affine_determinant) or affine_determinant <= 0:
            raise ValueError(f"affine counterfactual is not orientation preserving: {name}")
        frozen_query = p1_at_queries(frozen, queries)
        dynamic_query = p1_at_queries(dynamic, queries)
        counterfactual_query = frozen_query @ affine_matrix.T + affine_offset
        def score(aligned: np.ndarray) -> np.ndarray:
            mapped = canvas_unit_to_original_pixel(
                aligned @ common_matrix.T + common_offset,
                layouts[moving], 512)
            return np.linalg.norm(mapped - target_px, axis=1)
        frozen_error = score(frozen_query)
        affine_error = score(counterfactual_query)
        dynamic_error = score(dynamic_query)
        rows.append({
            "direction": name,
            "landmark_count": len(ids),
            "affine_fit_source_is_all_saved_frozen_vertices_not_landmarks": True,
            "extra_affine_determinant": affine_determinant,
            "extra_affine_fit_rms_canvas_px": fit_rms,
            "frozen_mean_TRE_native_moving_px": float(np.mean(frozen_error)),
            "affine_counterfactual_mean_TRE_native_moving_px": float(np.mean(affine_error)),
            "full_dynamic_mean_TRE_native_moving_px": float(np.mean(dynamic_error)),
            "full_minus_affine_counterfactual_mean_TRE_native_moving_px": float(
                np.mean(dynamic_error) - np.mean(affine_error)),
        })
    columns = ("frozen_mean_TRE_native_moving_px",
               "affine_counterfactual_mean_TRE_native_moving_px",
               "full_dynamic_mean_TRE_native_moving_px")
    return {
        "question": "does the retained non-affine increment improve anatomy beyond its best affine post-map?",
        "cohort": "20 correlated directions of one previously viewed lung specimen",
        "landmarks_used_for_affine_fit_or_model_selection": False,
        "counterfactual_topology": "certified frozen P1 composed with positive fitted affine and common positive external affine",
        "flattened_rounded_counterfactual_vertex_table_certified": False,
        "direction_count": len(rows),
        "equal_direction_means": {key: float(np.mean([row[key] for row in rows]))
                                  for key in columns},
        "full_beats_affine_counterfactual_count": sum(
            row["full_minus_affine_counterfactual_mean_TRE_native_moving_px"] < 0
            for row in rows),
        "counterfactual_beats_frozen_count": sum(
            row["affine_counterfactual_mean_TRE_native_moving_px"] <
            row["frozen_mean_TRE_native_moving_px"] for row in rows),
        "minimum_extra_affine_determinant": min(
            row["extra_affine_determinant"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "annotations", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.canvas, args.annotations, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("rows", "question", "cohort")}))


if __name__ == "__main__":
    main()
