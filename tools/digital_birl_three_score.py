"""Read-only score of a declared new safe-P1 arm on three reused development pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_birl_landmark_score import score_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--map-suffix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.root
    canvas = root / "canvas"
    labels = root / "labels_eval_only"
    configurations = {
        "lesions": (
            labels / "lesions_" / "scale-5pc" / "Izd2-29-041-w35_HE.csv",
            labels / "lesions_" / "scale-5pc" / "Izd2-29-041-w35_proSPC.csv",
        ),
        "rat_kidney": (
            labels / "rat-kidney_" / "scale-5pc" / "Rat-Kidney_HE.csv",
            labels / "rat-kidney_" / "scale-5pc" / "Rat-Kidney_PanCytokeratin.csv",
        ),
        "histo": (
            root.parent / "HistoReg_CD68_CD4" / "Landmarks_CD4.csv",
            root.parent / "HistoReg_CD68_CD4" / "Landmarks_CD68.csv",
        ),
    }
    reports = {}
    for case, (fixed_labels, moving_labels) in configurations.items():
        saved = args.map_dir / f"{case}_{args.map_suffix}.npz"
        case_out = args.output.with_name(args.output.stem + f"_{case}.json")
        if case_out.exists():
            raise FileExistsError(case_out)
        reports[case] = score_pair(
            canvas / f"{case}_layout.json", fixed_labels, moving_labels,
            {"new_map": saved}, canvas / f"{case}_direct_superglue_affine.npz",
            Path("unused-full-field"), Path("unused-full-params"),
            case_out, include_full_dhr=False, interpolation="p1")
    aggregate = {
        "mode": "reused-development three-pair exploratory score, not blind",
        "maps": str(args.map_dir), "map_suffix": args.map_suffix,
        "per_case": {case: {
            arm: row["mean_tre_native_moving_px"]
            for arm, row in report["results"].items()
        } for case, report in reports.items()},
    }
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate["per_case"]))


if __name__ == "__main__":
    main()
