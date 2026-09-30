"""Score all previously predicted lung-lesion-3 stain directions once.

This separate annotation-reading step refuses absent maps; the prediction step
does not import it or access landmarks. All 20 directions are retained.
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
from tools.digital_lung_lesion3_score import PREFIX, scaled_landmarks, statistics
from tools.digital_q1_real_eval import load_effective_vertices


STAIN_NAME = {"he": "He", "cc10": "Cc10-5", "cd31": "CD31-3",
              "ki67": "Ki67-7", "prospc": "proSPC-4"}


def score(canvas: Path, annotations: Path, predictions: Path) -> dict:
    pred_report = json.loads((predictions / "predictions.json").read_text())
    expected = {f"{a}_to_{b}" for a, b in itertools.permutations(STAIN_NAME, 2)}
    if {row["name"] for row in pred_report["rows"]} != expected or (
        len(pred_report["rows"]) != 20
    ):
        raise ValueError("prediction cohort is not exactly all 20 directions")
    layouts = {}
    for stain in STAIN_NAME:
        if stain == "he":
            layouts[stain] = json.loads((canvas / "cc10_layout.json").read_text())["fixed"]
        else:
            layouts[stain] = json.loads((canvas / f"{stain}_layout.json").read_text())["moving"]
    landmarks = {stain: scaled_landmarks(
        annotations / f"{PREFIX}{original}-les3.csv")
        for stain, original in STAIN_NAME.items()}
    rows = []
    for fixed, moving in itertools.permutations(STAIN_NAME, 2):
        name = f"{fixed}_to_{moving}"
        affine_path = predictions / f"{name}_affine.npz"
        frozen_path = predictions / f"{name}_frozen_safe257.npz"
        dynamic_path = predictions / f"{name}_dynamic_safe257.npz"
        for path in (affine_path, frozen_path, dynamic_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        names = sorted(landmarks[fixed].keys() & landmarks[moving].keys(), key=int)
        if len(names) != 80 or len(landmarks[fixed]) != 80 or len(landmarks[moving]) != 80:
            raise ValueError(f"not 80 corresponding landmarks for {name}")
        source_px = np.stack([landmarks[fixed][item] for item in names])
        target_px = np.stack([landmarks[moving][item] for item in names])
        query = original_pixel_to_canvas_unit(source_px, layouts[fixed], 512)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > 1):
            raise ValueError(f"landmark outside fixed canvas for {name}")
        with np.load(affine_path) as archive:
            matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float64)
            offset = np.asarray(archive["post_affine_offset"], dtype=np.float64)
        if float(np.linalg.det(matrix)) <= 0:
            raise ValueError(f"nonpositive affine for {name}")
        frozen, frozen_cert = load_effective_vertices(frozen_path)
        dynamic, dynamic_cert = load_effective_vertices(dynamic_path)
        if not frozen_cert["composite_representation_valid"] or not (
            dynamic_cert["composite_representation_valid"]
        ):
            raise ValueError(f"invalid saved map for {name}")
        for path in (frozen_path, dynamic_path):
            with np.load(path) as archive:
                if not (np.array_equal(archive["post_affine_matrix"], matrix) and
                        np.array_equal(archive["post_affine_offset"], offset)):
                    raise ValueError(f"saved map affine mismatch for {name}")
        affine_px = canvas_unit_to_original_pixel(
            query @ matrix.T + offset, layouts[moving], 512)
        frozen_px = canvas_unit_to_original_pixel(
            p1_at_queries(frozen, query), layouts[moving], 512)
        dynamic_px = canvas_unit_to_original_pixel(
            p1_at_queries(dynamic, query), layouts[moving], 512)
        frozen_error = frozen_px - target_px
        dynamic_increment = dynamic_px - frozen_px
        frozen_dist = np.linalg.norm(frozen_error, axis=1)
        dynamic_dist = np.linalg.norm(dynamic_px - target_px, axis=1)
        rows.append({
            "name": name, "fixed": fixed, "moving": moving,
            "landmarks": len(names),
            "affine": statistics(affine_px, target_px, names),
            "frozen": statistics(frozen_px, target_px, names),
            "dynamic": statistics(dynamic_px, target_px, names),
            "posthoc_landmark_direction_diagnostic": {
                "fraction_points_dynamic_better_than_frozen": float(np.mean(
                    dynamic_dist < frozen_dist)),
                "rms_dynamic_increment_px": float(np.sqrt(np.mean(np.sum(
                    dynamic_increment * dynamic_increment, axis=1)))),
                "mean_frozen_error_dot_dynamic_increment_px2": float(np.mean(
                    np.sum(frozen_error * dynamic_increment, axis=1))),
            },
            "frozen_certificate": frozen_cert,
            "dynamic_certificate": dynamic_cert,
        })
    return {
        "protocol": "all 20 ordered directions, one known specimen, maps saved before labels scored",
        "metric": "mean target error in moving native 5%-JPEG pixels, 80 matched landmarks per direction",
        "rows": rows,
        "dynamic_better_than_frozen_count": sum(
            row["dynamic"]["mean_px"] < row["frozen"]["mean_px"] for row in rows),
        "dynamic_better_than_affine_count": sum(
            row["dynamic"]["mean_px"] < row["affine"]["mean_px"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "annotations", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = score(args.canvas, args.annotations, args.predictions)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(result["rows"]),
                      "better_frozen": result["dynamic_better_than_frozen_count"],
                      "better_affine": result["dynamic_better_than_affine_count"]}))


if __name__ == "__main__":
    main()
