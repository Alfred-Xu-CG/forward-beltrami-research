"""Read-only seven-pair inference for a trained frozen-baseline residual head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_frozen_residual_predict import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--one-head", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--omit-matches", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root
    lung = root / "lung_lesion3_eval" / "canvas"
    birl = root / "birl_anhir_dev"
    pairs = []
    for name in ("cc10", "cd31", "ki67", "prospc"):
        pairs.append((name, lung / f"{name}_fixed512.png",
                      lung / f"{name}_moving512.png",
                      lung / f"{name}_directSG_affine.npz",
                      lung / f"{name}_alignedSG_matches.npz"))
    for name in ("lesions", "rat_kidney", "histo"):
        pairs.append((name, birl / "canvas" / f"{name}_fixed512.png",
                      birl / "canvas" / f"{name}_moving512.png",
                      root / "scaleup102" / "match_spatial_birl"
                      / f"{name}_direct_superglue_affine.npz",
                      birl / "directSG_aligned_matches"
                      / f"{name}_alignedSG_matches.npz"))
    rows = []
    for name, fixed, moving, affine, matches in pairs:
        output = args.output_dir / f"{name}_{args.tag}_safe257.npz"
        selected_matches = None if args.omit_matches else matches
        for path in (fixed, moving, affine) + (
            () if selected_matches is None else (selected_matches,)):
            if not path.is_file():
                raise FileNotFoundError(path)
        report = predict(args.base, args.one_head, args.residual,
                         fixed, moving, affine, selected_matches, output,
                         device_name=args.device, repeats=2)
        rows.append({"name": name, "forward_s": report["full_forward_seconds_median"],
                     "vjp_s": report[
                         "full_forward_and_parameter_match_vjp_seconds_median"],
                     "finite_vjp": report["finite_full_vjp"]})
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"tag": args.tag, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
