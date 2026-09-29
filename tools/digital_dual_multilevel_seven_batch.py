"""Predeclared seven-pair ridge sweep, never using anatomy in construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_dual_multilevel_match_safe257 import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridges", default=".001,.01,.1,1,10")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, out = args.root, args.output_dir
    lung = root / "lung_lesion3_eval" / "canvas"
    birl = root / "birl_anhir_dev" / "directSG_aligned_matches"
    pairs = [(name, lung / f"{name}_alignedSG_matches.npz")
             for name in ("cc10", "cd31", "ki67", "prospc")]
    pairs += [(name, birl / f"{name}_alignedSG_matches.npz")
              for name in ("lesions", "rat_kidney", "histo")]
    rows = []
    for ridge_text in args.ridges.split(","):
        ridge = float(ridge_text)
        tag = ridge_text.replace(".", "p")
        for name, matches in pairs:
            output = out / f"{name}_dualridge{tag}_safe257.npz"
            report = run(out / f"{name}_imageonlyfrozen_safe257.npz",
                         matches, output, ridge=ridge, device_name=args.device)
            row = {"case": name, "ridge": ridge,
                   "forward_seconds": report["forward_seconds"],
                   "backward_seconds": report["backward_seconds"],
                   "finite_vjp": report["finite_match_coordinate_vjp"],
                   "valid": report["saved_certificate"]["valid"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps({"count": len(rows), "all_valid": all(
        row["valid"] and row["finite_vjp"] for row in rows)}), flush=True)


if __name__ == "__main__":
    main()
