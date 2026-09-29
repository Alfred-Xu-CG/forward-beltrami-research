"""Compare frozen global networks on identical image-only affines, no labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_amortized_network import image_features, structural_loss
from tools.digital_mind_dense257_head import load_global_base
from tools.digital_q1_dhr_distill import identity_vertices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    old = load_global_base(args.old, device)
    new = load_global_base(args.new, device)
    old_saved = torch.load(args.old, map_location="cpu", weights_only=False)
    new_saved = torch.load(args.new, map_location="cpu", weights_only=False)
    old_val = set(old_saved["validation_ids"])
    ids = new_saved["validation_ids"]
    if not set(ids) <= old_val:
        raise ValueError("new validation IDs are not in original validation split")
    identity = identity_vertices(257, device=device)
    rows = []
    with torch.no_grad():
        for case in ids:
            item = load_case_inputs(args.root, case, device,
                                    affine_source="image_only")
            fixed = F.interpolate(item["fixed"], size=(128, 128), mode="area")
            moving = F.interpolate(item["prewarped"], size=(128, 128), mode="area")
            features, fdesc, mdesc, mask = image_features(fixed, moving)
            rows.append({
                "case": case,
                "affine": float(structural_loss(fdesc, mdesc, mask, identity)),
                "old_DHR_trained_on_same_image_only_affine": float(structural_loss(
                    fdesc, mdesc, mask, old(features, final_side=257))),
                "new_image_only_trained_on_same_affine": float(structural_loss(
                    fdesc, mdesc, mask, new(features, final_side=257))),
            })
    report = {
        "method": "fixed same-input heldout image-only-affine objective comparison",
        "old_checkpoint": str(args.old), "new_checkpoint": str(args.new),
        "count": len(rows), "rows": rows,
        "means": {key: sum(row[key] for row in rows) / len(rows)
                  for key in ("affine", "old_DHR_trained_on_same_image_only_affine",
                              "new_image_only_trained_on_same_affine")},
        "new_better_cases": sum(
            row["new_image_only_trained_on_same_affine"]
            < row["old_DHR_trained_on_same_image_only_affine"] for row in rows),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": report["count"], "means": report["means"],
                      "new_better_cases": report["new_better_cases"]}))


if __name__ == "__main__":
    main()
