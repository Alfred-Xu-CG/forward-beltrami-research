"""Fit unlabeled safe multilevel maps for the declared ACROBAT 81-case split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_multilevel_safe_pair_optimize import optimize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows = []
    baseline_maps = sorted(args.baseline_dir.glob("*_old_safe257.npz"),
                           key=lambda p: int(p.stem.split("_")[0]))
    if len(baseline_maps) != 81:
        raise ValueError(f"expected 81 distinct baselines, got {len(baseline_maps)}")
    for baseline in baseline_maps:
        case = baseline.stem.split("_")[0]
        source_report = json.loads(baseline.with_suffix(".json").read_text())
        fixed = args.root / source_report["fixed"]
        moving = args.root / source_report["moving"]
        matches = args.root / "alignedSG_matches" / f"{case}_alignedSG_matches.npz"
        for path in (fixed, moving, matches):
            if not path.is_file():
                raise FileNotFoundError(path)
        output = args.output_dir / f"{case}_multilevel{args.steps}_safe257.npz"
        if output.exists() and output.with_suffix(".json").exists():
            report = json.loads(output.with_suffix(".json").read_text())
        else:
            report = optimize(
                baseline, output, teacher_path=None, fixed_path=fixed,
                moving_path=moving, matches_path=matches,
                steps=args.steps, device_name=args.device)
        row = {"case": case, "best_step": report["best_step"],
               "initial_or_better": report["best_step"] > 0,
               "seconds": report["training_seconds"],
               "valid": report["saved_certificate"]["valid"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"cases": len(rows),
                      "optimized": sum(r["initial_or_better"] for r in rows),
                      "seconds": sum(r["seconds"] for r in rows)}), flush=True)


if __name__ == "__main__":
    main()
