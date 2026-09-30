"""Unlabeled inverse-consistency audit of the 20 saved stain-direction maps.

Evaluates two-map compositions at queries only; the composition is not claimed
to be P1 on the original grid. All 20 maps belong to one known specimen.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from tools.digital_birl_landmark_score import p1_at_queries
from tools.digital_lung_all20_score import STAIN_NAME
from tools.digital_q1_real_eval import load_effective_vertices


def run(predictions: Path, side: int = 65) -> dict:
    if side < 3:
        raise ValueError("query side must be at least three")
    axis = (np.arange(side, dtype=np.float64) + .5) / side
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    queries = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    directions = {}
    for fixed, moving in itertools.permutations(STAIN_NAME, 2):
        name = f"{fixed}_to_{moving}"
        with np.load(predictions / f"{name}_affine.npz") as data:
            matrix = data["post_affine_matrix"].astype(np.float64)
            offset = data["post_affine_offset"].astype(np.float64)
        if np.linalg.det(matrix) <= 0:
            raise ValueError(f"nonpositive affine: {name}")
        stored = {"affine": (matrix, offset)}
        for method, suffix in (("frozen", "frozen"), ("matched", "dynamic")):
            vertices, certificate = load_effective_vertices(
                predictions / f"{name}_{suffix}_safe257.npz")
            if not certificate["composite_representation_valid"]:
                raise ValueError(f"invalid map: {name} {method}")
            stored[method] = vertices
        directions[name] = stored

    def apply(stored: dict, method: str, points: np.ndarray) -> np.ndarray:
        if method == "affine":
            matrix, offset = stored["affine"]
            return points @ matrix.T + offset
        return p1_at_queries(stored[method], points)

    rows = []
    methods = ("affine", "frozen", "matched")
    for fixed, moving in itertools.permutations(STAIN_NAME, 2):
        forward = directions[f"{fixed}_to_{moving}"]
        backward = directions[f"{moving}_to_{fixed}"]
        intermediates = {method: apply(forward, method, queries)
                         for method in methods}
        common = np.ones(len(queries), dtype=bool)
        for values in intermediates.values():
            common &= np.isfinite(values).all(axis=1)
            common &= ((values >= 0.) & (values <= 1.)).all(axis=1)
        if not common.any():
            raise ValueError(f"empty common field of view: {fixed}->{moving}")
        errors = {}
        for method in methods:
            cycle = apply(backward, method, intermediates[method][common])
            errors[method] = np.linalg.norm(cycle - queries[common], axis=1) * 512.
        rows.append({
            "direction": f"{fixed}_to_{moving}",
            "query_total": len(queries), "common_query_count": int(common.sum()),
            **{f"{method}_mean_cycle_canvas_px": float(errors[method].mean())
               for method in methods},
            **{f"{method}_median_cycle_canvas_px": float(np.median(errors[method]))
               for method in methods},
            "matched_better_than_affine_query_count": int(
                (errors["matched"] < errors["affine"]).sum()),
            "matched_better_than_frozen_query_count": int(
                (errors["matched"] < errors["frozen"]).sum()),
        })
    return {
        "question": "Do independently inferred forward and reverse maps agree "
                    "as approximate inverses?",
        "cohort": "20 ordered directions from one already viewed specimen; no landmarks",
        "query_grid": f"{side}x{side} pixel-centered unit-canvas positions",
        "common_fov_rule": "all three forward intermediate locations inside [0,1]^2",
        "distance_unit": "512-canvas pixels, not original/native stain JPEG pixels",
        "direction_count": len(rows),
        "pooled_common_query_count": sum(row["common_query_count"] for row in rows),
        "equal_direction_mean_cycle_canvas_px": {
            method: float(np.mean([
                row[f"{method}_mean_cycle_canvas_px"] for row in rows]))
            for method in methods},
        "matched_lower_mean_than_affine_direction_count": sum(
            row["matched_mean_cycle_canvas_px"] <
            row["affine_mean_cycle_canvas_px"] for row in rows),
        "matched_lower_mean_than_frozen_direction_count": sum(
            row["matched_mean_cycle_canvas_px"] <
            row["frozen_mean_cycle_canvas_px"] for row in rows),
        "rows": rows,
        "manual_landmarks_read": False,
        "composed_map_claimed_P1_on_original_grid": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-side", type=int, default=65)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.predictions, args.query_side)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "direction_count", "pooled_common_query_count",
        "equal_direction_mean_cycle_canvas_px",
        "matched_lower_mean_than_affine_direction_count",
        "matched_lower_mean_than_frozen_direction_count")}))


if __name__ == "__main__":
    main()
