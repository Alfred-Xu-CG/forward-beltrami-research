"""Case-held-out match interpolation comparison with no anatomical labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_forward_kernel import forward_kernel_map
from tools.digital_q1_match_neural_decoder import (
    MatchSpatialDecoder, match_rmse, split_cases, split_matches, tensors,
)


def evaluate(report_path: Path, checkpoint: Path, output: Path, *, device_name: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(device_name)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _, val_rows = split_cases(report["rows"])
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if [row["case"] for row in val_rows] != saved["validation_case_ids"]:
        raise ValueError("checkpoint validation split differs")
    trained = MatchSpatialDecoder(update_sides=tuple(saved["update_sides"]),
                                  final_side=int(saved["final_side"])).to(device)
    initial = MatchSpatialDecoder(update_sides=tuple(saved["update_sides"]),
                                  final_side=int(saved["final_side"])).to(device)
    trained.net.load_state_dict(saved["state_dict"])
    trained.eval()
    initial.eval()
    rows = []
    with torch.no_grad():
        for row in val_rows:
            source, target = tensors(row, device)
            inp_s, inp_t, held_s, held_t = split_matches(
                source, target, seed=row["case"] + 470000, input_fraction=.7,
            )
            identity = identity_vertices(trained.final_side, device=device)
            fixed = forward_kernel_map(inp_s, inp_t, final_side=trained.final_side,
                                       sigma=.12)
            predictions = {
                "identity": identity, "fixed_gaussian": fixed,
                "untrained_blend": initial(inp_s, inp_t),
                "trained_blend": trained(inp_s, inp_t),
            }
            rows.append({"case": row["case"], "inliers": row["ransac_inliers"],
                         "occupied_4x4_bins": row["occupied_quadrants_4x4"],
                         "input_matches": len(inp_s), "held_matches": len(held_s),
                         **{name + "_held_rmse": float(match_rmse(map_, held_s, held_t))
                            for name, map_ in predictions.items()}})
    means = {name: float(np.mean([row[name + "_held_rmse"] for row in rows]))
             for name in ("identity", "fixed_gaussian", "untrained_blend", "trained_blend")}
    wins = {name: sum(row[name + "_held_rmse"] < row["identity_held_rmse"] for row in rows)
            for name in ("fixed_gaussian", "untrained_blend", "trained_blend")}
    result = {"claim_scope": "19 case-held-out, post-all-match-RANSAC machine correspondences",
              "P1_query_interpolation": True,
              "DHR_derived_initial_affine_loaded_upstream": True,
              "landmarks_or_full_DHR_displacement_loaded": False,
              "train_case_count": len(saved["train_case_ids"]),
              "validation_case_count": len(rows), "mean_heldout_match_rmse_unit": means,
              "cases_better_than_identity": wins, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = evaluate(args.matches, args.checkpoint, args.output, device_name=args.device)
    print(json.dumps({"validation_case_count": result["validation_case_count"],
                      "mean_heldout_match_rmse_unit": result["mean_heldout_match_rmse_unit"],
                      "cases_better_than_identity": result["cases_better_than_identity"]}))


if __name__ == "__main__":
    main()
