"""Post-label-access diagnostic: spatial coverage of machine matches near true landmarks.

This explicitly reads the same anatomical labels used for evaluation. It is
not an image-only inference component and must not drive blind model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit
from tools.digital_lung_lesion3_score import CASES, PREFIX, scaled_landmarks


def analyze(canvas: Path, annotations: Path, frozen_report: Path,
            optimized_report: Path) -> dict:
    old = json.loads(frozen_report.read_text(encoding="utf-8"))
    new = json.loads(optimized_report.read_text(encoding="utf-8"))
    cases = {}
    for case in CASES:
        layout = json.loads((canvas / f"{case}_layout.json").read_text(encoding="utf-8"))
        landmarks = scaled_landmarks(annotations / f"{PREFIX}He-les3.csv")
        names = sorted(landmarks, key=int)
        query_px = np.stack([landmarks[name] for name in names])
        query = original_pixel_to_canvas_unit(query_px, layout["fixed"], 512)
        with np.load(canvas / f"{case}_alignedSG_matches.npz") as data:
            source = np.asarray(data["source_fixed_unit"], dtype=np.float64)
        nearest = 512 * np.linalg.norm(
            query[:, None, :] - source[None, :, :], axis=-1).min(axis=1)
        frozen = np.array([
            old["cases"][case]["dense257_p1"]["per_landmark_px"][name]
            for name in names])
        optimized = np.array([
            new["cases"][case]["dense257_p1"]["per_landmark_px"][name]
            for name in names])
        order = np.argsort(nearest)
        bins = np.array_split(order, 4)
        cases[case] = {
            "match_count": len(source), "landmark_count": len(names),
            "nearest_match_distance_px_median": float(np.median(nearest)),
            "nearest_match_distance_px_p90": float(np.percentile(nearest, 90)),
            "distance_error_correlation": float(np.corrcoef(nearest, optimized)[0, 1]),
            "quartiles_by_match_distance": [{
                "count": len(indices),
                "distance_range_px": [float(nearest[indices].min()),
                                      float(nearest[indices].max())],
                "frozen_TRE_mean_px": float(frozen[indices].mean()),
                "optimized_TRE_mean_px": float(optimized[indices].mean()),
                "mean_gain_px": float((frozen[indices] - optimized[indices]).mean()),
            } for indices in bins],
        }
    return {
        "mode": "post-label-access descriptive spatial-coverage diagnostic",
        "source": "four lung-lesion_3 pairs; same specimen, not independent patients",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "annotations", "frozen_report",
                 "optimized_report", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.canvas, args.annotations,
                     args.frozen_report, args.optimized_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({case: {
        "nearest_match_distance_px_median": row["nearest_match_distance_px_median"],
        "distance_error_correlation": row["distance_error_correlation"],
    } for case, row in result["cases"].items()}))


if __name__ == "__main__":
    main()
