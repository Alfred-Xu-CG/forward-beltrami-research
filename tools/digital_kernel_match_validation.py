"""Select kernel hyperparameters using disjoint ACROBAT case IDs and held-back matches.

Held-back points are not independent anatomical ground truth: the original
matcher and RANSAC saw all correspondences before this split. This is a
within-matcher generalization diagnostic, not a blind registration benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.digital_kernel_match_safe257 import kernel_target, steer
from tools.digital_mind_amortized_network import split_ids
from tools.digital_mind_match_conditioned257 import split_matches
from tools.digital_mind_sparse_match_finetune import p1_at_points


def point_error(mapped: torch.Tensor, source: torch.Tensor,
                target: torch.Tensor) -> float:
    return float((512 * (p1_at_points(mapped, source) - target))
                 .norm(dim=-1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "baselines", "matches", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ids = json.loads(args.selection.read_text(encoding="utf-8"))["combined_train_ids"]
    train, validation = split_ids(ids)
    assert not set(train) & set(validation)
    candidates = [(s, r) for s in (.05, .1, .2) for r in (.01, .1, 1.)]
    device = torch.device(args.device)
    rows = []
    with torch.no_grad():
        for case in validation:
            bpath = args.baselines / f"{case}_old_safe257.npz"
            mpath = args.matches / f"{case}_alignedSG_matches.npz"
            if not bpath.exists() or not mpath.exists():
                continue
            with np.load(bpath) as data:
                baseline = torch.from_numpy(data["vertices"].copy()).to(device)
            with np.load(mpath) as data:
                source = torch.from_numpy(data["source_fixed_unit"].copy()).to(device)
                target = torch.from_numpy(data["target_aligned_unit"].copy()).to(device)
            source_in, target_in, source_held, target_held = split_matches(
                source, target, case)
            if len(source_in) < 16 or len(source_held) < 1:
                continue
            old = point_error(baseline, source_held, target_held)
            for sigma, ridge in candidates:
                goal = kernel_target(baseline, source_in, target_in,
                                     sigma=sigma, ridge=ridge)
                mapped = steer(baseline, goal, passes=4, minimum_jacobian=.05)
                rows.append({
                    "case": case, "total_matches": len(source),
                    "input_count": len(source_in), "held_count": len(source_held),
                    "sigma": sigma, "ridge": ridge,
                    "baseline_heldpoint_mean_px": old,
                    "kernel_input_mean_px": point_error(
                        mapped, source_in, target_in),
                    "kernel_heldpoint_mean_px": point_error(
                        mapped, source_held, target_held),
                })
    aggregate = []
    for sigma, ridge in candidates:
        group = [r for r in rows if r["sigma"] == sigma and r["ridge"] == ridge]
        aggregate.append({"sigma": sigma, "ridge": ridge,
                          "case_count": len(group),
                          "baseline_heldpoint_mean_px": float(np.mean([
                              r["baseline_heldpoint_mean_px"] for r in group])),
                          "kernel_heldpoint_mean_px": float(np.mean([
                              r["kernel_heldpoint_mean_px"] for r in group])),
                          "kernel_input_mean_px": float(np.mean([
                              r["kernel_input_mean_px"] for r in group]))})
    best = min(aggregate, key=lambda r: r["kernel_heldpoint_mean_px"])
    report = {
        "method": "case-disjoint ACROBAT validation; 80pct SG matches as input",
        "not_independent_anatomical_landmarks": True,
        "train_case_count": len(train), "validation_case_count": len(validation),
        "evaluated_case_count": len(set(row["case"] for row in rows)),
        "selection_metric": "unweighted mean of per-case heldpoint mean pixel error",
        "best": best, "aggregate": aggregate, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "aggregate": aggregate}))


if __name__ == "__main__":
    main()
