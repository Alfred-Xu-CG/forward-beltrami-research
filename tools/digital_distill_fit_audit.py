"""Audit train-versus-case-disjoint validation fit of a frozen safe-map student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_amortized_network import image_features, split_ids
from tools.digital_mind_dense257_head import dense_features, load_global_base
from tools.digital_mind_match_conditioned257 import gaussian_match_features
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "selection", "matches", "teachers",
                 "base_checkpoint", "student_checkpoint", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frozen-maps", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ids = json.loads(args.selection.read_text(encoding="utf-8"))["combined_train_ids"]
    train_ids, val_ids = split_ids(ids)
    device = torch.device(args.device)
    base = load_global_base(args.base_checkpoint, device).eval()
    saved = torch.load(args.student_checkpoint, map_location=device,
                       weights_only=False)
    model = RecurrentDenseSafeHead(
        passes=4, match_channels=int(saved.get("match_channels", 6)),
        context=saved.get("context", "local")).to(device).eval()
    model.load_state_dict(saved["model_state_dict"])
    rows = []
    with torch.no_grad():
        for case in ids:
            mpath = args.matches / f"{case}_alignedSG_matches.npz"
            tpath = args.teachers / f"{case}_matchopt4_safe257.npz"
            if not mpath.exists() or not tpath.exists():
                continue
            with np.load(mpath) as data:
                source = torch.from_numpy(data["source_fixed_unit"].copy()).to(device)
                target = torch.from_numpy(data["target_aligned_unit"].copy()).to(device)
            if len(source) < 16:
                continue
            with np.load(tpath) as data:
                teacher = torch.from_numpy(data["vertices"].copy()).to(device)
            item = load_case_inputs(args.root, case, device,
                                    affine_source="image_only")
            feature, _, _, _ = image_features(
                F.interpolate(item["fixed"], size=(128, 128), mode="area"),
                F.interpolate(item["prewarped"], size=(128, 128), mode="area"))
            coarse = base(feature, final_side=257)
            if args.frozen_maps is not None:
                with np.load(args.frozen_maps / f"{case}_old_safe257.npz") as data:
                    coarse = torch.from_numpy(data["vertices"].copy()).to(device)
                    if not np.array_equal(data["post_affine_matrix"],
                                          item["matrix"][0].cpu().numpy()):
                        raise ValueError(f"frozen/affine matrix mismatch for {case}")
                    if not np.array_equal(data["post_affine_offset"],
                                          item["offset"][0].cpu().numpy()):
                        raise ValueError(f"frozen/affine offset mismatch for {case}")
            fine, _, _, _ = dense_features(
                feature, item["fixed"], item["prewarped"], coarse,
                feature_geometry=saved.get(
                    "residual_feature_geometry", "legacy_align_corners"))
            raster = (gaussian_match_features(source, target)
                      if model.match_channels else None)
            mapped = model(coarse, fine, match_feature=raster)
            def rmse(a):
                return float((a - teacher).square().sum(-1).mean().sqrt())
            rows.append({"case": case,
                         "split": "train" if case in train_ids else "validation",
                         "source_count": len(source),
                         "coarse_to_teacher_rmse_unit": rmse(coarse),
                         "student_to_teacher_rmse_unit": rmse(mapped)})
    summary = {}
    for split in ("train", "validation"):
        subset = [r for r in rows if r["split"] == split]
        summary[split] = {"count": len(subset),
                          "coarse_to_teacher_rmse_unit": float(np.mean([
                              r["coarse_to_teacher_rmse_unit"] for r in subset])),
                          "student_to_teacher_rmse_unit": float(np.mean([
                              r["student_to_teacher_rmse_unit"] for r in subset]))}
    result = {"method": "read-only frozen student teacher-map fit audit",
              "initial_map": ("frozen_onehead_257" if args.frozen_maps is not None
                              else "global65_refined_257"),
              "anatomical_labels_used": False,
              "student": str(args.student_checkpoint),
              "teacher": str(args.teachers),
              "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
