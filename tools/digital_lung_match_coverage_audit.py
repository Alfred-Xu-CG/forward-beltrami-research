"""Read-only spatial audit of archived machine matches versus anatomy gains.

All twenty directions belong to one previously viewed specimen. Manual
annotations enter evaluation only; this script does not choose or train maps.
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


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(len(values), dtype=np.float64)
    return out


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


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
        if len(ids) != 80 or len(source) != 80 or len(target) != 80:
            raise ValueError(f"unexpected annotations: {name}")
        source_px = np.stack([source[item] for item in ids])
        target_px = np.stack([target[item] for item in ids])
        query = original_pixel_to_canvas_unit(source_px, layouts[fixed], 512)
        with np.load(predictions / f"{name}_aligned_matches.npz") as archive:
            match_source = np.asarray(archive["source_fixed_unit"], dtype=np.float64)
            match_matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float64)
            match_offset = np.asarray(archive["post_affine_offset"], dtype=np.float64)
        if (match_source.ndim != 2 or match_source.shape[1:] != (2,) or
            not len(match_source) or not np.isfinite(match_source).all() or
            np.any((match_source < 0) | (match_source > 1)) or
            not np.isfinite(query).all() or np.any((query < 0) | (query > 1))):
            raise ValueError(f"empty/invalid match set: {name}")
        nearest = 512. * np.sqrt(np.min(np.sum(
            (query[:, None, :] - match_source[None, :, :]) ** 2,
            axis=-1), axis=1))
        errors = {}
        for method in ("frozen", "dynamic"):
            with np.load(predictions / f"{name}_{method}_safe257.npz") as archive:
                matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float64)
                offset = np.asarray(archive["post_affine_offset"], dtype=np.float64)
                vertices = np.asarray(archive["vertices"][0], dtype=np.float64)
            if not (np.array_equal(matrix, match_matrix) and
                    np.array_equal(offset, match_offset)):
                raise ValueError(f"different external affine: {name}")
            aligned = p1_at_queries(vertices, query)
            mapped_px = canvas_unit_to_original_pixel(
                aligned @ matrix.T + offset, layouts[moving], 512)
            errors[method] = np.linalg.norm(mapped_px - target_px, axis=1)
        gain = errors["frozen"] - errors["dynamic"]
        median_distance = float(np.median(nearest))
        near = nearest <= median_distance
        rows.append({
            "direction": name,
            "match_count": int(len(match_source)),
            "landmark_count": int(len(ids)),
            "median_nearest_match_canvas_px": median_distance,
            "mean_gain_native_moving_px": float(np.mean(gain)),
            "near_half_mean_gain_native_moving_px": float(np.mean(gain[near])),
            "far_half_mean_gain_native_moving_px": float(np.mean(gain[~near])),
            "distance_gain_spearman": _correlation(_ranks(nearest), _ranks(gain)),
            "points": [{
                "id": item,
                "nearest_match_canvas_px": float(distance),
                "frozen_TRE_native_moving_px": float(frozen),
                "dynamic_TRE_native_moving_px": float(dynamic),
                "frozen_minus_dynamic_gain_native_moving_px": float(delta),
            } for item, distance, frozen, dynamic, delta in zip(
                ids, nearest, errors["frozen"], errors["dynamic"], gain)],
        })
    distance = np.array([point["nearest_match_canvas_px"]
                         for row in rows for point in row["points"]])
    gain = np.array([point["frozen_minus_dynamic_gain_native_moving_px"]
                     for row in rows for point in row["points"]])
    quantiles = np.quantile(distance, [0, .25, .5, .75, 1.])
    quartile = np.minimum(np.searchsorted(quantiles[1:-1], distance,
                                         side="left"), 3)
    summary = {
        "distance_quantiles_canvas_px": quantiles.tolist(),
        "pooled_quartile_counts": [int(np.sum(quartile == k)) for k in range(4)],
        "pooled_quartile_mean_gain_native_moving_px": [
            float(np.mean(gain[quartile == k])) for k in range(4)],
        "equal_direction_mean_near_gain_native_moving_px": float(np.mean([
            row["near_half_mean_gain_native_moving_px"] for row in rows])),
        "equal_direction_mean_far_gain_native_moving_px": float(np.mean([
            row["far_half_mean_gain_native_moving_px"] for row in rows])),
        "near_gain_exceeds_far_direction_count": sum(
            row["near_half_mean_gain_native_moving_px"] >
            row["far_half_mean_gain_native_moving_px"] for row in rows),
        "equal_direction_mean_spearman": float(np.mean([
            row["distance_gain_spearman"] for row in rows])),
        "equal_direction_mean_TRE_gain_native_moving_px": float(np.mean([
            row["mean_gain_native_moving_px"] for row in rows])),
    }
    return {
        "question": "is anatomical gain localized near selected fixed-side machine matches?",
        "cohort": "20 correlated directions of one previously viewed lung specimen",
        "manual_landmarks_used_only_for_retrospective_evaluation": True,
        "not_causal_or_independent_patient_evidence": True,
        "direction_count": len(rows), "summary": summary, "rows": rows,
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
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
