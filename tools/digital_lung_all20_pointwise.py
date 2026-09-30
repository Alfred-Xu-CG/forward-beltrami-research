"""Check whether all-20 mean TRE gains hide landmark-level regressions.

All 20 directions come from one already viewed specimen. This post-hoc
evaluation reads landmarks only after the saved maps have been frozen.
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


def run(canvas: Path, annotations: Path, predictions: Path) -> dict:
    layouts = {}
    for stain in STAIN_NAME:
        path = canvas / ("cc10_layout.json" if stain == "he" else
                         f"{stain}_layout.json")
        layouts[stain] = json.loads(path.read_text(encoding="utf-8"))[
            "fixed" if stain == "he" else "moving"]
    landmarks = {
        stain: scaled_landmarks(annotations / f"{PREFIX}{original}-les3.csv")
        for stain, original in STAIN_NAME.items()
    }
    all_errors = {key: [] for key in ("affine", "frozen", "matched")}
    directions = []
    for fixed, moving in itertools.permutations(STAIN_NAME, 2):
        name = f"{fixed}_to_{moving}"
        ids = sorted(landmarks[fixed].keys() & landmarks[moving].keys(), key=int)
        if len(ids) != 80:
            raise ValueError(f"expected 80 landmarks: {name}")
        source = np.stack([landmarks[fixed][item] for item in ids])
        target = np.stack([landmarks[moving][item] for item in ids])
        query = original_pixel_to_canvas_unit(source, layouts[fixed], 512)
        errors = {}
        with np.load(predictions / f"{name}_affine.npz") as archive:
            matrix = archive["post_affine_matrix"].astype(np.float64)
            offset = archive["post_affine_offset"].astype(np.float64)
        if np.linalg.det(matrix) <= 0:
            raise ValueError(f"nonpositive affine: {name}")
        aligned = query @ matrix.T + offset
        errors["affine"] = np.linalg.norm(
            canvas_unit_to_original_pixel(aligned, layouts[moving], 512) - target,
            axis=1)
        for key, suffix in (("frozen", "frozen"), ("matched", "dynamic")):
            vertices, certificate = load_effective_vertices(
                predictions / f"{name}_{suffix}_safe257.npz")
            if not certificate["composite_representation_valid"]:
                raise ValueError(f"invalid saved map: {name} {key}")
            pixel = canvas_unit_to_original_pixel(
                p1_at_queries(vertices, query), layouts[moving], 512)
            errors[key] = np.linalg.norm(pixel - target, axis=1)
        for key in all_errors:
            all_errors[key].append(errors[key])
        directions.append({
            "direction": name, "landmark_count": len(ids),
            **{f"{key}_mean_px": float(errors[key].mean())
               for key in all_errors},
            "matched_better_than_affine_points": int((errors["matched"]
                                                       < errors["affine"]).sum()),
            "matched_better_than_frozen_points": int((errors["matched"]
                                                       < errors["frozen"]).sum()),
            "matched_minus_affine_median_px": float(np.median(
                errors["matched"] - errors["affine"])),
            "matched_minus_frozen_median_px": float(np.median(
                errors["matched"] - errors["frozen"])),
        })
    concatenated = {key: np.concatenate(values)
                    for key, values in all_errors.items()}
    matched = concatenated["matched"]
    affine = concatenated["affine"]
    frozen = concatenated["frozen"]
    return {
        "question": "Are mean TRE gains driven only by a few large outliers?",
        "cohort": "20 ordered directions, 80 repeated landmark IDs, one known specimen",
        "observation_count": len(matched),
        "pooled_mean_px": {key: float(values.mean())
                           for key, values in concatenated.items()},
        "pooled_median_px": {key: float(np.median(values))
                             for key, values in concatenated.items()},
        "pooled_90th_percentile_px": {key: float(np.quantile(values, .9))
                                      for key, values in concatenated.items()},
        "matched_better_than_affine_point_count": int((matched < affine).sum()),
        "matched_better_than_frozen_point_count": int((matched < frozen).sum()),
        "matched_minus_affine_median_px": float(np.median(matched - affine)),
        "matched_minus_frozen_median_px": float(np.median(matched - frozen)),
        "directions": directions,
        "correlated_observations_not_independent_patients": True,
        "anatomy_labels_used_only_for_posthoc_evaluation": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "annotations", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.canvas, args.annotations, args.predictions)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "pooled_mean_px", "pooled_median_px", "pooled_90th_percentile_px",
        "matched_better_than_affine_point_count",
        "matched_better_than_frozen_point_count")}))


if __name__ == "__main__":
    main()
