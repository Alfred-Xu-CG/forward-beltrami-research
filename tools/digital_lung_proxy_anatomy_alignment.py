"""Post-hoc rank audit of two archived models on one known lung specimen.

No model is selected, trained or inferred here. Scores had already been opened
for this development specimen; this cannot serve as held-out validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.digital_birl_landmark_score import p1_at_queries
from tools.digital_lung_all20_score import STAIN_NAME
from tools.digital_q1_real_eval import load_effective_vertices


def _rows(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in data["rows"]}
    if len(rows) != len(data["rows"]):
        raise ValueError(f"duplicate direction: {path}")
    return rows


def run(six: Path, twelve: Path) -> dict:
    a_pred, b_pred = _rows(six / "predictions.json"), _rows(
        twelve / "predictions.json")
    a_score, b_score = _rows(six / "scores.json"), _rows(
        twelve / "scores.json")
    expected = {f"{a}_to_{b}" for a in STAIN_NAME for b in STAIN_NAME if a != b}
    if not all(set(rows) == expected for rows in (
        a_pred, b_pred, a_score, b_score
    )):
        raise ValueError("four archives do not contain the same 20 directions")

    rows = []
    for name in sorted(expected):
        for suffix in ("affine", "frozen_safe257"):
            pa = six / f"{name}_{suffix}.npz"
            pb = twelve / f"{name}_{suffix}.npz"
            with np.load(pa) as first, np.load(pb) as second:
                for key in ("post_affine_matrix", "post_affine_offset"):
                    if not np.array_equal(first[key], second[key]):
                        raise ValueError(f"different external affine: {name}")
            if suffix == "frozen_safe257":
                va, ca = load_effective_vertices(pa)
                vb, cb = load_effective_vertices(pb)
                if not (ca["composite_representation_valid"] and
                        cb["composite_representation_valid"] and
                        np.array_equal(va, vb)):
                    raise ValueError(f"different frozen map: {name}")

        for method in ("affine", "frozen"):
            if not np.isclose(a_score[name][method]["mean_px"],
                              b_score[name][method]["mean_px"],
                              rtol=0., atol=1e-10):
                raise ValueError(f"different {method} anatomy baseline: {name}")
        image6 = float(a_pred[name]["dynamic_image"])
        image12 = float(b_pred[name]["dynamic_image"])
        tre6 = float(a_score[name]["dynamic"]["mean_px"])
        tre12 = float(b_score[name]["dynamic"]["mean_px"])
        with np.load(twelve / f"{name}_aligned_matches.npz") as matches:
            source = np.asarray(matches["source_fixed_unit"], dtype=np.float64)
            target = np.asarray(matches["target_aligned_unit"], dtype=np.float64)
            with np.load(twelve / f"{name}_dynamic_safe257.npz") as archive:
                for key in ("post_affine_matrix", "post_affine_offset"):
                    if not np.array_equal(matches[key], archive[key]):
                        raise ValueError(f"match/12-channel affine mismatch: {name}")
                vertices12 = np.asarray(archive["vertices"][0], dtype=np.float64)
            with np.load(six / f"{name}_dynamic_safe257.npz") as archive:
                for key in ("post_affine_matrix", "post_affine_offset"):
                    if not np.array_equal(matches[key], archive[key]):
                        raise ValueError(f"match/six-channel affine mismatch: {name}")
                vertices6 = np.asarray(archive["vertices"][0], dtype=np.float64)
        if (source.ndim != 2 or source.shape[1] != 2 or
            target.shape != source.shape or not len(source) or
            not np.isfinite(source).all() or not np.isfinite(target).all() or
            np.any(source < 0) or np.any(source > 1) or
            np.any(target < 0) or np.any(target > 1)):
            raise ValueError(f"invalid selected match coordinates: {name}")
        def robust(vertices: np.ndarray) -> float:
            residual_px = 512. * (p1_at_queries(vertices, source) - target)
            return float(np.mean(np.sqrt(np.sum(residual_px ** 2, axis=1) + 16.) - 4.))
        match6, match12 = robust(vertices6), robust(vertices12)
        rows.append({
            "direction": name,
            "six_image_proxy": image6,
            "twelve_image_proxy": image12,
            "twelve_minus_six_image_proxy": image12 - image6,
            "six_native_pixel_TRE": tre6,
            "twelve_native_pixel_TRE": tre12,
            "twelve_minus_six_native_pixel_TRE": tre12 - tre6,
            "common_machine_match_count": len(source),
            "six_robust_machine_match_canvas_px": match6,
            "twelve_robust_machine_match_canvas_px": match12,
            "twelve_minus_six_robust_machine_match_canvas_px": match12 - match6,
            "twelve_better_image": image12 < image6,
            "twelve_better_TRE": tre12 < tre6,
            "twelve_better_machine_match": match12 < match6,
        })
    counts = {f"image_{i}_TRE_{j}": sum(
        row["twelve_better_image"] == i and row["twelve_better_TRE"] == j
        for row in rows) for i in (False, True) for j in (False, True)}
    match_counts = {f"match_{i}_TRE_{j}": sum(
        row["twelve_better_machine_match"] == i and row["twelve_better_TRE"] == j
        for row in rows) for i in (False, True) for j in (False, True)}
    return {
        "question": "do the image and machine-match proxies rank the better anatomy-TRE model?",
        "cohort": "20 correlated ordered directions of one previously viewed specimen",
        "comparison": "fixed six-channel vs 12-channel checkpoints; no model selection",
        "external_affines_and_frozen_composite_maps_bitwise_equal": True,
        "anatomy_unit": "native moving 5%-JPEG pixels",
        "image_unit": "dimensionless MIND-like descriptor loss",
        "direction_count": len(rows),
        "sign_table": counts,
        "machine_match_TRE_sign_table": match_counts,
        "mean_twelve_minus_six_image_proxy": float(np.mean([
            row["twelve_minus_six_image_proxy"] for row in rows])),
        "mean_twelve_minus_six_native_pixel_TRE": float(np.mean([
            row["twelve_minus_six_native_pixel_TRE"] for row in rows])),
        "mean_twelve_minus_six_robust_machine_match_canvas_px": float(np.mean([
            row["twelve_minus_six_robust_machine_match_canvas_px"]
            for row in rows])),
        "image_better_count": sum(row["twelve_better_image"] for row in rows),
        "TRE_better_count": sum(row["twelve_better_TRE"] for row in rows),
        "machine_match_better_count": sum(
            row["twelve_better_machine_match"] for row in rows),
        "machine_match_archive_is_12channel_input_not_independent_truth": True,
        "manual_landmarks_newly_read": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--six", type=Path, required=True)
    parser.add_argument("--twelve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.six, args.twelve)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "direction_count", "sign_table", "machine_match_TRE_sign_table",
        "mean_twelve_minus_six_image_proxy",
        "mean_twelve_minus_six_native_pixel_TRE",
        "mean_twelve_minus_six_robust_machine_match_canvas_px")}))


if __name__ == "__main__":
    main()
