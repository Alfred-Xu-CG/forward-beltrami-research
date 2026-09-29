"""Case-disjoint held-back-match depth ablation for the fixed kernel target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.digital_kernel_match_safe257 import kernel_target, steer
from tools.digital_kernel_match_validation import point_error
from tools.digital_mind_amortized_network import split_ids
from tools.digital_mind_match_conditioned257 import split_matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "baselines", "matches", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ids = json.loads(args.selection.read_text(encoding="utf-8"))["combined_train_ids"]
    _, validation = split_ids(ids)
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
            if len(source_in) < 16 or not len(source_held):
                continue
            goal = kernel_target(baseline, source_in, target_in)
            baseline_held = point_error(baseline, source_held, target_held)
            for passes in (1, 2, 4, 8):
                mapped = steer(baseline, goal, passes=passes)
                rows.append({"case": case, "passes": passes,
                             "input_count": len(source_in),
                             "held_count": len(source_held),
                             "baseline_heldpoint_mean_px": baseline_held,
                             "kernel_input_mean_px": point_error(
                                 mapped, source_in, target_in),
                             "kernel_heldpoint_mean_px": point_error(
                                 mapped, source_held, target_held)})
    aggregate = [{
        "passes": passes, "case_count": len(group),
        "baseline_heldpoint_mean_px": float(np.mean([
            r["baseline_heldpoint_mean_px"] for r in group])),
        "kernel_heldpoint_mean_px": float(np.mean([
            r["kernel_heldpoint_mean_px"] for r in group])),
        "kernel_input_mean_px": float(np.mean([
            r["kernel_input_mean_px"] for r in group])),
    } for passes in (1, 2, 4, 8)
        for group in [[r for r in rows if r["passes"] == passes]]]
    report = {"method": "ACROBAT case-disjoint held-back-matcher depth ablation",
              "not_independent_anatomical_landmarks": True,
              "sigma": .1, "ridge": .1, "validation_case_count": len(validation),
              "evaluated_case_count": len(set(r["case"] for r in rows)),
              "aggregate": aggregate, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate))


if __name__ == "__main__":
    main()
