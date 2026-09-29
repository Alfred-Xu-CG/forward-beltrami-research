"""Read-only image-proxy comparison on case-disjoint ACROBAT confirmation pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_amortized_network import structural_loss
from tools.digital_mind_objective_probe import self_similarity
from tools.digital_mind_safe_optimize import strain_penalty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    ids = selection["new_confirmation_ids"]
    if set(ids) & set(selection["combined_train_ids"]):
        raise ValueError("case IDs overlap")
    device = torch.device(args.device)
    rows = []
    with torch.no_grad():
        for case in ids:
            item = load_case_inputs(args.root, case, device,
                                    affine_source="image_only")
            fixed = F.interpolate(item["fixed"], size=(256, 256), mode="area")
            moving = F.interpolate(item["prewarped"], size=(256, 256),
                                   mode="area")
            fdesc, scale = self_similarity(fixed)
            mdesc, _ = self_similarity(moving)
            mask = ((fixed > .04) & (scale > 1e-4)).to(fixed.dtype)
            scores = {}
            for name, path in (
                ("frozen", args.baseline_dir / f"{case}_onepass_safe257.npz"),
                ("kernel", args.kernel_dir / f"{case}_kernelmatch4_safe257.npz"),
            ):
                with np.load(path) as data:
                    vertices = torch.from_numpy(data["vertices"].copy()).to(device)
                scores[name] = {
                    "image": float(structural_loss(fdesc, mdesc, mask, vertices)),
                    "strain": float(strain_penalty(vertices)),
                }
            rows.append({"case": case, **scores})
    result = {
        "method": "disjoint-case image-proxy comparison, no anatomical labels",
        "not_anatomical_validation": True,
        "case_count": len(rows), "rows": rows,
        "mean_frozen_image": float(np.mean([r["frozen"]["image"] for r in rows])),
        "mean_kernel_image": float(np.mean([r["kernel"]["image"] for r in rows])),
        "kernel_better_cases": sum(
            r["kernel"]["image"] < r["frozen"]["image"] for r in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "case_count", "mean_frozen_image", "mean_kernel_image",
        "kernel_better_cases",
    )}))


if __name__ == "__main__":
    main()
