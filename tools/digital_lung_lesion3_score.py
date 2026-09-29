"""Read-only, predeclared four-pair lung-lesion_3 evaluation.

The predictions must exist before this script reads the 50%-scale landmarks.
Coordinates are converted to the provided 5%-scale JPEGs using the nominal
center-preserving factor 10; that convention has subpixel uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.digital_birl_landmark_score import (
    canvas_unit_to_original_pixel, original_pixel_to_canvas_unit, p1_at_queries,
)
from tools.digital_q1_real_eval import load_effective_vertices


CASES = {
    "cc10": "Cc10-5", "cd31": "CD31-3", "ki67": "Ki67-7",
    "prospc": "proSPC-4",
}
PREFIX = "29-041-Izd2-w35-"


def scaled_landmarks(path: Path) -> dict[str, np.ndarray]:
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = row[" "].strip()
            if not name or name in result:
                raise ValueError(f"empty or duplicate landmark ID in {path}")
            point = np.array([float(row["X"]), float(row["Y"])], dtype=np.float64)
            result[name] = (point + .5) / 10.0 - .5
    return result


def statistics(predicted: np.ndarray, target: np.ndarray, names: list[str]) -> dict:
    delta = predicted - target
    distance = np.sqrt(np.sum(delta * delta, axis=1))
    return {
        "mean_px": float(np.mean(distance)),
        "median_px": float(np.median(distance)),
        "p95_px": float(np.percentile(distance, 95)),
        "max_px": float(np.max(distance)),
        "per_landmark_px": dict(zip(names, distance.tolist(), strict=True)),
    }


def score_case(canvas: Path, annotations: Path, case: str, stain: str, *,
               map_dir: Path | None = None,
               map_suffix: str = "directSG_mind_dense257") -> dict:
    layout = json.loads((canvas / f"{case}_layout.json").read_text(encoding="utf-8"))
    vertices, certificate = load_effective_vertices(
        (map_dir or canvas) / f"{case}_{map_suffix}.npz")
    if not certificate["composite_representation_valid"]:
        raise ValueError(f"invalid map certificate: {case}")
    if vertices.shape != (257, 257, 2):
        raise ValueError(f"not a 257-square map: {case}")
    with np.load(canvas / f"{case}_directSG_affine.npz") as data:
        matrix = np.asarray(data["post_affine_matrix"], dtype=np.float64)
        offset = np.asarray(data["post_affine_offset"], dtype=np.float64)
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if determinant <= 0:
        raise ValueError(f"nonpositive affine: {case}")
    fixed = scaled_landmarks(annotations / f"{PREFIX}He-les3.csv")
    moving = scaled_landmarks(annotations / f"{PREFIX}{stain}-les3.csv")
    names = sorted(fixed.keys() & moving.keys(), key=lambda v: int(v))
    if len(names) != 80 or len(fixed) != 80 or len(moving) != 80:
        raise ValueError(f"expected exactly 80 paired landmarks: {case}")
    fixed_px = np.stack([fixed[name] for name in names])
    moving_px = np.stack([moving[name] for name in names])
    fixed_wh = np.asarray(layout["fixed"]["original_wh"])
    moving_wh = np.asarray(layout["moving"]["original_wh"])
    if np.any(fixed_px < -.5) or np.any(fixed_px > fixed_wh - .5):
        raise ValueError(f"fixed landmark outside 5% image: {case}")
    if np.any(moving_px < -.5) or np.any(moving_px > moving_wh - .5):
        raise ValueError(f"moving landmark outside 5% image: {case}")
    query = original_pixel_to_canvas_unit(fixed_px, layout["fixed"], 512)
    if np.any(query < 0) or np.any(query > 1):
        raise ValueError(f"fixed canvas query outside square: {case}")
    map_unit = p1_at_queries(vertices, query)
    affine_unit = np.stack((
        query[:, 0] * matrix[0, 0] + query[:, 1] * matrix[0, 1] + offset[0],
        query[:, 0] * matrix[1, 0] + query[:, 1] * matrix[1, 1] + offset[1],
    ), axis=1)
    target = moving_px
    return {
        "case": case, "stain": stain, "landmarks": len(names),
        "fixed_5pc_size_wh": fixed_wh.tolist(),
        "moving_5pc_size_wh": moving_wh.tolist(),
        "positive_affine_determinant": float(determinant),
        "map_certificate": certificate,
        "initial_affine": statistics(canvas_unit_to_original_pixel(
            affine_unit, layout["moving"], 512), target, names),
        "dense257_p1": statistics(canvas_unit_to_original_pixel(
            map_unit, layout["moving"], 512), target, names),
        "canvas_identity": statistics(canvas_unit_to_original_pixel(
            query, layout["moving"], 512), target, names),
    }


def evaluate(canvas: Path, annotations: Path, output: Path, *,
             map_dir: Path | None = None,
             map_suffix: str = "directSG_mind_dense257") -> dict:
    if output.exists():
        raise FileExistsError(output)
    cases = {case: score_case(canvas, annotations, case, stain,
                             map_dir=map_dir, map_suffix=map_suffix)
             for case, stain in CASES.items()}
    report = {
        "protocol": (
            "predeclared frozen direct SuperGlue affine plus frozen global-context and dense257 CNN, four He-to-stain pairs"
            if map_suffix == "directSG_mind_dense257" else
            "post-label-access exploratory comparison: weights trained without lung labels but arm assessed after original four-pair labels were opened"),
        "source_specimen": "lung-lesion_3, one specimen and one annotator PS; pairs are not independent patients",
        "training_tuning_labels_used": "LABEL_ORACLE" in map_suffix,
        "not_deployable_label_oracle": "LABEL_ORACLE" in map_suffix,
        "prospective_map_freeze_before_label_access": map_suffix == "directSG_mind_dense257",
        "map_source": str(map_dir or canvas), "map_suffix": map_suffix,
        "landmark_coordinate_conversion": "provided 50%-scale CSV to 5%-scale JPEG: (coordinate+0.5)/10-0.5, nominal 10x convention; subpixel uncertainty",
        "metric": "forward P1 fixed landmark to moving 5%-scale native JPEG Euclidean pixel TRE",
        "cases": cases,
        "aggregate_four_pair_mean_of_means_px": {
            method: float(np.mean([cases[c][method]["mean_px"] for c in CASES]))
            for method in ("canvas_identity", "initial_affine", "dense257_p1")
        },
        "dense257_better_than_affine_cases": sum(
            cases[c]["dense257_p1"]["mean_px"] < cases[c]["initial_affine"]["mean_px"]
            for c in CASES),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path)
    parser.add_argument("--map-suffix", default="directSG_mind_dense257")
    args = parser.parse_args()
    report = evaluate(args.canvas, args.annotations, args.output,
                      map_dir=args.map_dir, map_suffix=args.map_suffix)
    print(json.dumps({
        "mean_of_means_px": report["aggregate_four_pair_mean_of_means_px"],
        "dense_better_cases": report["dense257_better_than_affine_cases"],
        "per_case": {c: {
            "affine": v["initial_affine"]["mean_px"],
            "dense": v["dense257_p1"]["mean_px"],
        } for c, v in report["cases"].items()},
    }))


if __name__ == "__main__":
    main()
