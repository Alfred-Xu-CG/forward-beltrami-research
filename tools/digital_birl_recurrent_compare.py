"""Read-only BIRL/ANHIR development landmark comparison after frozen predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_birl_landmark_score import score_pair


LABELS = {
    "rat_kidney": ("rat-kidney_", "Rat-Kidney_HE.csv",
                   "Rat-Kidney_PanCytokeratin.csv"),
    "lesions": ("lesions_", "Izd2-29-041-w35_HE.csv",
                "Izd2-29-041-w35_proSPC.csv"),
}


def compare(root: Path, case: str, output: Path, *,
            extra_predictions: dict[str, Path] | None = None,
            include_full_dhr: bool = True) -> dict:
    if case not in LABELS:
        raise ValueError("known development case required")
    canvas = root / "canvas"
    folder, fixed_label, moving_label = LABELS[case]
    labels = root / "labels_eval_only" / folder / "scale-5pc"
    maps = {
        "old22_actual": canvas / f"{case}_aug2400_prediction/actual_safe_q1.npz",
        "old22_blank": canvas / f"{case}_aug2400_prediction/blank_safe_q1.npz",
        "base102_actual": canvas / f"{case}_scaleup102_prediction/actual_safe_q1.npz",
        "base102_blank": canvas / f"{case}_scaleup102_prediction/blank_safe_q1.npz",
        "recurrent102_actual": canvas / f"{case}_recurrent102_prediction/actual_safe_q1.npz",
        "recurrent102_blank": canvas / f"{case}_recurrent102_prediction/blank_safe_q1.npz",
    }
    if extra_predictions:
        if set(extra_predictions).intersection(maps):
            raise ValueError("extra prediction names must be distinct")
        maps.update(extra_predictions)
    return score_pair(
        canvas / f"{case}_layout.json", labels / fixed_label,
        labels / moving_label, maps,
        canvas / f"{case}_initial_affine.npz",
        canvas / f"{case}_full_field.mha",
        canvas / f"{case}_full_params.json", output,
        include_full_dhr=include_full_dhr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(LABELS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-prediction", nargs=2, action="append",
                        metavar=("NAME", "SAVED_MAP"), default=[])
    parser.add_argument("--skip-full-dhr", action="store_true",
                        help="reuse prior DHR score without loading its field")
    args = parser.parse_args()
    extras = {name: Path(path) for name, path in args.extra_prediction}
    if len(extras) != len(args.extra_prediction):
        raise ValueError("duplicate extra prediction name")
    report = compare(args.root, args.case, args.output,
                     extra_predictions=extras,
                     include_full_dhr=not args.skip_full_dhr)
    print(json.dumps({key: value["mean_tre_native_moving_px"]
                      for key, value in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
