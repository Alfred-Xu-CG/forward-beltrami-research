"""Post-hoc boundary-distance anatomy diagnostic on all 20 saved lung maps.

Reads manual annotations only after maps have been frozen and scored. The
output is a mechanism diagnostic on one known specimen, never a model input.
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


def diagnose(canvas: Path, annotations: Path, predictions: Path) -> dict:
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
    rows = []
    for fixed, moving in itertools.permutations(STAIN_NAME, 2):
        name = f"{fixed}_to_{moving}"
        names = sorted(landmarks[fixed].keys() & landmarks[moving].keys(), key=int)
        if len(names) != 80:
            raise ValueError(f"expected 80 landmarks: {name}")
        source_px = np.stack([landmarks[fixed][item] for item in names])
        target_px = np.stack([landmarks[moving][item] for item in names])
        query = original_pixel_to_canvas_unit(source_px, layouts[fixed], 512)
        if np.any(query < 0) or np.any(query > 1) or not np.isfinite(query).all():
            raise ValueError(f"landmark outside canvas: {name}")
        boundary_distance = np.minimum(query, 1. - query).min(axis=1)
        affine_path = predictions / f"{name}_affine.npz"
        with np.load(affine_path) as archive:
            matrix = archive["post_affine_matrix"].astype(np.float64)
            offset = archive["post_affine_offset"].astype(np.float64)
        if np.linalg.det(matrix) <= 0:
            raise ValueError(f"nonpositive affine: {name}")
        errors = {}
        affine_pixel = canvas_unit_to_original_pixel(
            query @ matrix.T + offset, layouts[moving], 512)
        errors["affine"] = np.linalg.norm(affine_pixel - target_px, axis=1)
        for key, suffix in (("frozen", "frozen"), ("matched", "dynamic")):
            vertices, certificate = load_effective_vertices(
                predictions / f"{name}_{suffix}_safe257.npz")
            if not certificate["composite_representation_valid"]:
                raise ValueError(f"invalid saved {key} map: {name}")
            pixel = canvas_unit_to_original_pixel(
                p1_at_queries(vertices, query), layouts[moving], 512)
            errors[key] = np.linalg.norm(pixel - target_px, axis=1)
        rows.append({"name": name, "distance": boundary_distance,
                     "error": errors})
    all_distances = np.concatenate([row["distance"] for row in rows])
    quartiles = np.quantile(all_distances, (0., .25, .5, .75, 1.))
    pooled = []
    for index in range(4):
        selected = [(row, np.searchsorted(quartiles[1:-1], row["distance"],
                                           side="right") == index) for row in rows]
        pooled.append({
            "quartile": index + 1,
            "distance_range_unit": [float(quartiles[index]),
                                    float(quartiles[index + 1])],
            "landmark_direction_observations": int(sum(mask.sum() for _, mask in selected)),
            **{f"{key}_mean_native_px": float(np.mean(np.concatenate([
                row["error"][key][mask] for row, mask in selected])))
               for key in ("affine", "frozen", "matched")},
        })
    median = quartiles[2]
    by_direction = []
    for row in rows:
        near = row["distance"] < median
        far = ~near
        if not near.any() or not far.any():
            raise ValueError(f"empty near/far group: {row['name']}")
        by_direction.append({
            "name": row["name"], "near_count": int(near.sum()),
            "far_count": int(far.sum()),
            **{f"{key}_{region}_mean_px": float(np.mean(row["error"][key][mask]))
               for key in ("affine", "frozen", "matched")
               for region, mask in (("near", near), ("far", far))},
        })
    return {
        "question": "Do matched-map residual anatomy errors concentrate near the fixed canvas boundary?",
        "cohort": "20 correlated directions, 80 landmark positions on one known specimen",
        "distance_definition": "minimum of x,1-x,y,1-y in fixed 512-canvas unit coordinates",
        "quantile_edges_unit": quartiles.tolist(),
        "pooled_quartiles": pooled, "directions": by_direction,
        "matched_near_error_exceeds_far_count": sum(
            row["matched_near_mean_px"] > row["matched_far_mean_px"]
            for row in by_direction),
        "annotations_only_posthoc": True,
        "maps_or_models_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "annotations", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = diagnose(args.canvas, args.annotations, args.predictions)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"quartiles": result["pooled_quartiles"],
                      "near_worse_directions": result[
                          "matched_near_error_exceeds_far_count"]}))


if __name__ == "__main__":
    main()
