"""Read-only diagnostic of ridge-goal versus saved safe maps on two opened cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.digital_birl_landmark_score import score_pair
from tools.digital_dual_multilevel_match_safe257 import dual_fit_map, multilevel_design
from tools.digital_lung_lesion3_score import score_case
from tools.digital_mind_sparse_match_finetune import p1_at_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root, out, image_root = args.root, args.output, args.image_root
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for name in ("cc10", "histo"):
        match_file = (root / "lung_lesion3_eval/canvas/cc10_alignedSG_matches.npz"
                      if name == "cc10" else
                      root / "birl_anhir_dev/directSG_aligned_matches/histo_alignedSG_matches.npz")
        with np.load(image_root / f"{name}_imageonlyfrozen_safe257.npz") as data:
            baseline = torch.from_numpy(data["vertices"].copy()).to(args.device)
        with np.load(match_file) as data:
            source = torch.from_numpy(data["source_fixed_unit"].copy()).to(args.device)
            target = torch.from_numpy(data["target_aligned_unit"].copy()).to(args.device)
        _, controls = dual_fit_map(baseline, source, target, ridge=.1, passes=4)
        design = multilevel_design(source)
        base_at = p1_at_points(baseline, source)
        goal_at = base_at + design @ controls
        for passes in (4, 8, 16):
            filename = (f"{name}_dualridgep1_safe257.npz" if passes == 4 else
                        f"{name}_dualridge0p1_pass{passes}_safe257.npz")
            with np.load(image_root / filename) as data:
                saved = torch.from_numpy(data["vertices"].copy()).to(args.device)
            safe_at = p1_at_points(saved, source)
            if name == "cc10":
                anatomy = score_case(root / "lung_lesion3_eval/canvas",
                                     root / "lung_lesion3_eval/annotations50",
                                     "cc10", "Cc10-5", map_dir=image_root,
                                     map_suffix=filename.removeprefix("cc10_").removesuffix(".npz"))
                tre = anatomy["dense257_p1"]["mean_px"]
            else:
                labels = root / "HistoReg_CD68_CD4"
                detail_path = out.with_name(out.stem + f"_histo_p{passes}.json")
                anatomy = score_pair(
                    root / "birl_anhir_dev/canvas/histo_layout.json",
                    labels / "Landmarks_CD4.csv", labels / "Landmarks_CD68.csv",
                    {"safe": image_root / filename},
                    root / "birl_anhir_dev/canvas/histo_direct_superglue_affine.npz",
                    Path("unused-full-field"), Path("unused-full-params"),
                    detail_path, include_full_dhr=False, interpolation="p1")
                tre = anatomy["results"]["safe"]["mean_tre_native_moving_px"]
            row = {
                "case": name, "passes": passes, "match_count": len(source),
                "baseline_match_mean_canvas_px": float(
                    (base_at - target).norm(dim=-1).mean() * 512),
                "ridge_goal_match_mean_canvas_px": float(
                    (goal_at - target).norm(dim=-1).mean() * 512),
                "safe_match_mean_canvas_px": float(
                    (safe_at - target).norm(dim=-1).mean() * 512),
                "safe_to_goal_mean_canvas_px": float(
                    (safe_at - goal_at).norm(dim=-1).mean() * 512),
                "anatomy_mean_native_moving_px": tre,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    out.write_text(json.dumps({"rows": rows,
                               "note": "Anatomy pairs were previously viewed; diagnostics only."},
                              indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
