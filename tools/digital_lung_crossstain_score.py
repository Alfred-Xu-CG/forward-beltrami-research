"""Evaluate three preregistered non-H&E-fixed lung stain pairs after maps exist.

This is a same-specimen distribution-shift diagnostic, not an unseen-patient test.
No landmark CSV is read until all three frozen map and affine files exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.digital_birl_landmark_score import (
    canvas_unit_to_original_pixel, original_pixel_to_canvas_unit, p1_at_queries,
)
from tools.digital_lung_lesion3_score import PREFIX, scaled_landmarks, statistics
from tools.digital_q1_real_eval import load_effective_vertices


PAIRS = (
    ("cd31_to_ki67", "cd31", "ki67", "CD31-3", "Ki67-7"),
    ("ki67_to_prospc", "ki67", "prospc", "Ki67-7", "proSPC-4"),
    ("prospc_to_cc10", "prospc", "cc10", "proSPC-4", "Cc10-5"),
)


def score(canvas: Path, annotations: Path, predictions: Path) -> dict:
    for name, fixed, moving, _, _ in PAIRS:
        for path in (
            canvas / f"{fixed}_moving512.png",
            canvas / f"{moving}_moving512.png",
            predictions / f"{name}_affine.npz",
            predictions / f"{name}_dynamic_safe257.npz",
            predictions / f"{name}_frozen_safe257.npz",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
    result = {}
    for name, fixed, moving, fixed_stain, moving_stain in PAIRS:
        fixed_layout = json.loads((canvas / f"{fixed}_layout.json").read_text())["moving"]
        moving_layout = json.loads((canvas / f"{moving}_layout.json").read_text())["moving"]
        first = scaled_landmarks(annotations / f"{PREFIX}{fixed_stain}-les3.csv")
        second = scaled_landmarks(annotations / f"{PREFIX}{moving_stain}-les3.csv")
        names = sorted(first.keys() & second.keys(), key=int)
        if len(names) != 80 or len(first) != 80 or len(second) != 80:
            raise ValueError(f"expected 80 paired landmarks: {name}")
        source_px = np.stack([first[item] for item in names])
        target_px = np.stack([second[item] for item in names])
        query = original_pixel_to_canvas_unit(source_px, fixed_layout, 512)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > 1):
            raise ValueError(f"landmark outside fixed canvas: {name}")
        map_path = predictions / f"{name}_dynamic_safe257.npz"
        vertices, certificate = load_effective_vertices(map_path)
        if not certificate["composite_representation_valid"]:
            raise ValueError(f"invalid map: {name}")
        frozen_path = predictions / f"{name}_frozen_safe257.npz"
        frozen_vertices, frozen_certificate = load_effective_vertices(frozen_path)
        if not frozen_certificate["composite_representation_valid"]:
            raise ValueError(f"invalid frozen map: {name}")
        with np.load(predictions / f"{name}_affine.npz") as data:
            matrix = np.asarray(data["post_affine_matrix"], dtype=np.float64)
            offset = np.asarray(data["post_affine_offset"], dtype=np.float64)
        with np.load(map_path) as data:
            if not (np.array_equal(data["post_affine_matrix"], matrix) and
                    np.array_equal(data["post_affine_offset"], offset)):
                raise ValueError(f"map/affine mismatch: {name}")
        with np.load(frozen_path) as data:
            if not (np.array_equal(data["post_affine_matrix"], matrix) and
                    np.array_equal(data["post_affine_offset"], offset)):
                raise ValueError(f"frozen/affine mismatch: {name}")
        determinant = float(np.linalg.det(matrix))
        if determinant <= 0:
            raise ValueError(f"nonpositive affine: {name}")
        mapped_unit = p1_at_queries(vertices, query)
        frozen_unit = p1_at_queries(frozen_vertices, query)
        affine_unit = query @ matrix.T + offset
        result[name] = {
            "source_stain": fixed_stain,
            "target_stain": moving_stain,
            "landmark_count": len(names),
            "positive_affine_determinant": determinant,
            "map_certificate": certificate,
            "frozen_certificate": frozen_certificate,
            "initial_affine": statistics(canvas_unit_to_original_pixel(
                affine_unit, moving_layout, 512), target_px, names),
            "frozen_safe257": statistics(canvas_unit_to_original_pixel(
                frozen_unit, moving_layout, 512), target_px, names),
            "dynamic_safe257": statistics(canvas_unit_to_original_pixel(
                mapped_unit, moving_layout, 512), target_px, names),
        }
    return {
        "protocol": "three fixed non-H&E lung stain pairs, same known specimen; "
                    "predictions frozen before pairwise landmark scoring",
        "metric": "mean fixed-P1-to-moving native-5pc-JPEG pixel TRE",
        "pairs": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = score(args.canvas, args.annotations, args.predictions)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {
        "affine": row["initial_affine"]["mean_px"],
        "frozen": row["frozen_safe257"]["mean_px"],
        "safe_p1": row["dynamic_safe257"]["mean_px"],
    } for name, row in report["pairs"].items()}))


if __name__ == "__main__":
    main()
