"""Restrict saved 257-grid maps to nested controls and independently certify.

The result is a post-hoc one-specimen output-compression diagnostic. A coarse
P1 interpolation is not the exact original 257-grid P1 map, and no coarse
student is trained or selected here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_birl_landmark_score import (
    canvas_unit_to_original_pixel, original_pixel_to_canvas_unit, p1_at_queries,
)
from tools.digital_lung_all20_score import STAIN_NAME
from tools.digital_lung_lesion3_score import PREFIX, scaled_landmarks


def _certify_arrays(vertices: np.ndarray, reference: np.ndarray,
                    artifact_parent: Path) -> dict:
    descriptor, name = tempfile.mkstemp(suffix=".npz", dir=artifact_parent)
    os.close(descriptor)
    path = Path(name)
    try:
        np.savez_compressed(path, vertices=vertices,
                            boundary_reference=reference)
        return certify_q1_binary_map(path)
    finally:
        path.unlink(missing_ok=True)


def run(canvas: Path, annotations: Path, predictions: Path,
        artifact_parent: Path) -> dict:
    layouts = {}
    landmarks = {}
    for stain, original in STAIN_NAME.items():
        layout_file = (canvas / "cc10_layout.json" if stain == "he" else
                       canvas / f"{stain}_layout.json")
        layout_key = "fixed" if stain == "he" else "moving"
        layouts[stain] = json.loads(layout_file.read_text())[layout_key]
        landmarks[stain] = scaled_landmarks(
            annotations / f"{PREFIX}{original}-les3.csv")
    sides = (17, 33, 65, 129, 257)
    rows = []
    for fixed, moving in itertools.permutations(STAIN_NAME, 2):
        name = f"{fixed}_to_{moving}"
        source = landmarks[fixed]
        target = landmarks[moving]
        ids = sorted(source.keys() & target.keys(), key=int)
        if len(ids) != 80 or len(source) != 80 or len(target) != 80:
            raise ValueError(f"unexpected landmark cohort: {name}")
        source_px = np.stack([source[item] for item in ids])
        target_px = np.stack([target[item] for item in ids])
        query = original_pixel_to_canvas_unit(source_px, layouts[fixed], 512)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > 1):
            raise ValueError(f"source annotation outside canvas: {name}")
        with np.load(predictions / f"{name}_dynamic_safe257.npz") as archive:
            vertices = np.asarray(archive["vertices"])
            reference = np.asarray(archive["boundary_reference"])
            matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float64)
            offset = np.asarray(archive["post_affine_offset"], dtype=np.float64)
        if vertices.shape != (1, 257, 257, 2) or (
            reference.shape != vertices.shape or np.linalg.det(matrix) <= 0
        ):
            raise ValueError(f"invalid original archive: {name}")
        by_side = {}
        for side in sides:
            stride = 256 // (side - 1)
            coarse = vertices[:, ::stride, ::stride].copy()
            coarse_reference = reference[:, ::stride, ::stride].copy()
            if coarse.shape != (1, side, side, 2):
                raise AssertionError(f"nested index error: {name} {side}")
            cert = _certify_arrays(coarse, coarse_reference, artifact_parent)
            mapped_aligned = p1_at_queries(coarse[0].astype(np.float64), query)
            mapped_px = canvas_unit_to_original_pixel(
                mapped_aligned @ matrix.T + offset, layouts[moving], 512)
            tre = np.linalg.norm(mapped_px - target_px, axis=1)
            by_side[str(side)] = {
                "mean_native_pixel_TRE": float(np.mean(tre)),
                "median_native_pixel_TRE": float(np.median(tre)),
                "exact_q1_certificate_valid": bool(cert["valid"]),
                "exact_positive_corners": int(cert["positive_corners"]),
                "exact_nonpositive_corners": int(cert["nonpositive_corners"]),
                "exact_fallback_corners": int(cert["exact_fallback_corners"]),
                "boundary_exact": bool(cert["boundary_exact"]),
            }
        for side in sides[:-1]:
            by_side[str(side)]["delta_mean_TRE_vs_257_native_px"] = (
                by_side[str(side)]["mean_native_pixel_TRE"] -
                by_side["257"]["mean_native_pixel_TRE"])
        rows.append({"direction": name, "landmarks": len(ids),
                     "by_side": by_side})
    return {
        "question": "does this saved dense map's current anatomy gain require 257-grid detail?",
        "cohort": "20 correlated directions of one previously viewed lung specimen",
        "method": "nested binary32 vertex restriction then new coarse fixed-diagonal P1",
        "not_exact_refinement_or_trained_coarse_network": True,
        "anatomy_unit": "native moving 5%-JPEG pixels",
        "direction_count": len(rows),
        "sides": list(sides),
        "summary": {str(side): {
            "equal_direction_mean_TRE_native_px": float(np.mean([
                row["by_side"][str(side)]["mean_native_pixel_TRE"]
                for row in rows])),
            "exact_valid_count": sum(
                row["by_side"][str(side)]["exact_q1_certificate_valid"]
                for row in rows),
            "worse_than_257_direction_count": (None if side == 257 else sum(
                row["by_side"][str(side)]["delta_mean_TRE_vs_257_native_px"] > 0
                for row in rows)),
        } for side in sides},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("canvas", "annotations", "predictions", "output"):
        parser.add_argument("--" + item, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run(args.canvas, args.annotations, args.predictions,
                 args.output.parent)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
