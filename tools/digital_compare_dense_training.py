"""Compare old/new frozen complete 257-P1 networks on the same image-only affines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_amortized_network import image_features, structural_loss
from tools.digital_mind_dense257_head import DenseSafeHead, dense_features, load_global_base
from tools.digital_mind_objective_probe import self_similarity


def load_head(path: Path, device: torch.device):
    saved = torch.load(path, map_location=device, weights_only=False)
    head = DenseSafeHead().to(device)
    head.load_state_dict(saved["head_state_dict"])
    head.eval()
    return head, saved.get("feature_geometry", "legacy_align_corners")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    for name in ("old_base", "old_head", "new_base", "new_head", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    old_base, new_base = (load_global_base(path, device) for path in
                          (args.old_base, args.new_base))
    old_head, old_geometry = load_head(args.old_head, device)
    new_head, new_geometry = load_head(args.new_head, device)
    old_saved = torch.load(args.old_base, map_location="cpu", weights_only=False)
    new_saved = torch.load(args.new_base, map_location="cpu", weights_only=False)
    ids = new_saved["validation_ids"]
    if not set(ids) <= set(old_saved["validation_ids"]):
        raise ValueError("new validation IDs outside old split")
    rows = []
    with torch.no_grad():
        for case in ids:
            item = load_case_inputs(args.root, case, device, affine_source="image_only")
            fixed = F.interpolate(item["fixed"], size=(128, 128), mode="area")
            moving = F.interpolate(item["prewarped"], size=(128, 128), mode="area")
            feature, _, _, _ = image_features(fixed, moving)
            fixed256 = F.interpolate(item["fixed"], size=(256, 256), mode="area")
            moving256 = F.interpolate(item["prewarped"], size=(256, 256), mode="area")
            fdesc, fscale = self_similarity(fixed256)
            mdesc, _ = self_similarity(moving256)
            mask = ((fixed256 > .04) & (fscale > 1e-4)).to(fixed256.dtype)
            losses = {}
            for name, base, head, geometry in (
                ("old", old_base, old_head, old_geometry),
                ("new", new_base, new_head, new_geometry),
            ):
                coarse = base(feature, final_side=257)
                dense_feature, *_ = dense_features(
                    feature, item["fixed"], item["prewarped"], coarse,
                    feature_geometry=geometry)
                dense = head(coarse, dense_feature)
                losses[name + "_base"] = float(structural_loss(
                    fdesc, mdesc, mask, coarse))
                losses[name + "_dense"] = float(structural_loss(
                    fdesc, mdesc, mask, dense))
            rows.append({"case": case, **losses})
    keys = ("old_base", "old_dense", "new_base", "new_dense")
    report = {"case_count": len(rows), "rows": rows,
              "means": {key: sum(row[key] for row in rows) / len(rows) for key in keys},
              "new_dense_better_cases": sum(r["new_dense"] < r["old_dense"] for r in rows),
              "metric": "same image-only affine, 256-square P1 descriptor loss",
              "old_feature_geometry": old_geometry,
              "new_feature_geometry": new_geometry}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": report["case_count"], "means": report["means"],
                      "new_better_cases": report["new_dense_better_cases"]}))


if __name__ == "__main__":
    main()
