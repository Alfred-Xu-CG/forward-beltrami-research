"""Predeclared 102-vs-22 ACROBAT pseudo-teacher held-out summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(new_report: Path, old_report: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    new = json.loads(new_report.read_text(encoding="utf-8"))
    old = json.loads(old_report.read_text(encoding="utf-8"))
    if (new["mode"] != "score_presealed_predictions_against_subsequent_full_DHR"
            or old["mode"] != "score_presealed_predictions_against_subsequent_full_DHR"):
        raise ValueError("both inputs must be sealed-prediction scores")
    if new["test_case_ids"] != old["test_case_ids"] or len(new["cases"]) != 8:
        raise ValueError("eight case IDs must match in their frozen order")
    if len(new["train_case_ids"]) != 102 or len(old["train_case_ids"]) != 22:
        raise ValueError("unexpected 102/22 train split")
    old_cases = {row["case"]: row for row in old["cases"]}
    keys = {
        "actual": "actual_to_DHR_full_vertex_rmse",
        "blank": "blank_to_DHR_full_vertex_rmse",
        "affine": "initial_affine_to_DHR_full_vertex_rmse",
    }
    cases = []
    for item in new["cases"]:
        case = item["case"]
        if case not in old_cases or (item[keys["affine"]]
                                      != old_cases[case][keys["affine"]]):
            raise ValueError("old-model case or common affine mismatch")
        cases.append({
            "case": case,
            "new_actual": item[keys["actual"]],
            "new_blank": item[keys["blank"]],
            "old_actual": old_cases[case][keys["actual"]],
            "old_blank": old_cases[case][keys["blank"]],
            "affine": item[keys["affine"]],
        })
    values = {key: np.asarray([row[key] for row in cases], dtype=np.float64)
              for key in ("new_actual", "new_blank", "old_actual", "old_blank", "affine")}
    actual = values["new_actual"]
    # Case bootstrap resamples the paired eight-case rows, not individual vertices.
    rng = np.random.default_rng(20261003)
    picks = rng.integers(0, 8, size=(10_000, 8))
    comparisons = {}
    for name in ("new_blank", "affine", "old_actual"):
        differences = actual - values[name]
        sampled_means = differences[picks].mean(axis=1)
        comparisons[f"new_actual_minus_{name}"] = {
            "mean_difference": float(differences.mean()),
            "paired_case_bootstrap_95_percentile_interval": [
                float(value) for value in np.quantile(sampled_means, [.025, .975])
            ],
            "casewise_better_count": int(np.sum(differences < 0)),
        }
    joint = (actual < values["new_blank"]) & (actual < values["affine"])
    report = {
        "metric": "all-vertex full-coordinate RMSE versus subsequent DHR teacher",
        "test_case_ids": new["test_case_ids"],
        "cases": cases,
        "means": {key: float(value.mean()) for key, value in values.items()},
        "new_actual_better_than_both_own_controls_cases": int(joint.sum()),
        "new_actual_better_than_both_own_controls_ids": [
            row["case"] for row, passed in zip(cases, joint, strict=True) if passed
        ],
        "predeclared_suggestive_screen": bool(
            actual.mean() < values["new_blank"].mean()
            and actual.mean() < values["affine"].mean()
            and joint.sum() >= 6
        ),
        "bootstrap_seed": 20261003,
        "bootstrap_resamples": 10_000,
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-report", type=Path, required=True)
    parser.add_argument("--old-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.new_report, args.old_report, args.output), indent=2))


if __name__ == "__main__":
    main()
