"""Run the same landmark-blind multilevel safe fit on seven declared pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_multilevel_safe_pair_optimize import optimize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-tag", default="imageonlyfrozen")
    parser.add_argument("--tag")
    args = parser.parse_args()
    root, out = args.root, args.output_dir
    lung = root / "lung_lesion3_eval" / "canvas"
    birl = root / "birl_anhir_dev"
    cases = []
    for name in ("cc10", "cd31", "ki67", "prospc"):
        cases.append((name, lung / f"{name}_fixed512.png",
                      lung / f"{name}_moving512.png",
                      lung / f"{name}_alignedSG_matches.npz"))
    for name in ("lesions", "rat_kidney", "histo"):
        cases.append((name, birl / "canvas" / f"{name}_fixed512.png",
                      birl / "canvas" / f"{name}_moving512.png",
                      birl / "directSG_aligned_matches" / f"{name}_alignedSG_matches.npz"))
    rows = []
    for name, fixed, moving, matches in cases:
        tag = args.tag or f"multilevel{args.steps}"
        saved = out / f"{name}_{tag}_safe257.npz"
        if saved.exists() and saved.with_suffix(".json").exists():
            report = json.loads(saved.with_suffix(".json").read_text())
        else:
            report = optimize(
                out / f"{name}_{args.baseline_tag}_safe257.npz", saved,
                teacher_path=None, fixed_path=fixed, moving_path=moving,
                matches_path=matches, steps=args.steps, device_name=args.device)
        rows.append({"case": name, "seconds": report["training_seconds"],
                     "objective": report["best_total"],
                     "best_step": report["best_step"],
                     "valid": report["saved_certificate"]["valid"]})
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"steps": args.steps, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
