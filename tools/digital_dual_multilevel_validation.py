"""Case-disjoint unlabeled ridge audit of the dual multilevel safe map."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_dual_multilevel_match_safe257 import dual_fit_map
from tools.digital_mind_amortized_network import split_ids
from tools.digital_mind_match_conditioned257 import split_matches
from tools.digital_mind_sparse_match_finetune import p1_at_points, robust_match_loss
from tools.digital_q1_dhr_distill import identity_vertices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--matches-dir", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridges", default=".001,.01,.1,1,10")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ids = json.loads(args.selection.read_text())["combined_train_ids"]
    _, val_ids = split_ids(ids)
    ridges = [float(x) for x in args.ridges.split(",")]
    device = torch.device(args.device)
    reference = identity_vertices(257, device=device)
    rows = []
    for case in val_ids:
        if not all(path.is_file() for path in (
            args.baseline_dir / f"{case}_old_safe257.npz",
            args.teacher_dir / f"{case}_multilevel25_safe257.npz",
            args.matches_dir / f"{case}_alignedSG_matches.npz",
        )):
            continue
        with np.load(args.baseline_dir / f"{case}_old_safe257.npz") as data:
            baseline = torch.from_numpy(data["vertices"].copy()).to(device)
            matrix = data["post_affine_matrix"].copy()
            offset = data["post_affine_offset"].copy()
        with np.load(args.teacher_dir / f"{case}_multilevel25_safe257.npz") as data:
            teacher = torch.from_numpy(data["vertices"].copy()).to(device)
            if not (np.array_equal(matrix, data["post_affine_matrix"]) and
                    np.array_equal(offset, data["post_affine_offset"])):
                raise ValueError(f"teacher frame differs for {case}")
        with np.load(args.matches_dir / f"{case}_alignedSG_matches.npz") as data:
            source = torch.from_numpy(data["source_fixed_unit"].copy()).to(device)
            target = torch.from_numpy(data["target_aligned_unit"].copy()).to(device)
        train_s, train_t, held_s, held_t = split_matches(source, target, case)
        with torch.no_grad():
            base_map_rmse = float((baseline - teacher).square().sum(-1).mean().sqrt())
            base_held = float(robust_match_loss(
                p1_at_points(baseline, held_s), held_t))
            for ridge in ridges:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                full, _ = dual_fit_map(baseline, source, target, ridge=ridge)
                held, _ = dual_fit_map(baseline, train_s, train_t, ridge=ridge)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                if not validate_q1_map(full, reference)["valid"] or not (
                    validate_q1_map(held, reference)["valid"]
                ):
                    raise RuntimeError(f"invalid dual map for {case} ridge {ridge}")
                rows.append({
                    "case": case, "ridge": ridge, "match_count": len(source),
                    "train_matches": len(train_s), "held_matches": len(held_s),
                    "baseline_teacher_rmse_unit": base_map_rmse,
                    "dual_teacher_rmse_unit": float(
                        (full - teacher).square().sum(-1).mean().sqrt()),
                    "baseline_held_match_robust_px": base_held,
                    "dual_held_match_robust_px": float(robust_match_loss(
                        p1_at_points(held, held_s), held_t)),
                    "two_forward_seconds": seconds,
                })
    aggregates = []
    if len({row["case"] for row in rows}) != 18:
        raise ValueError("expected exactly 18 available validation cases")
    for ridge in ridges:
        group = [row for row in rows if row["ridge"] == ridge]
        aggregates.append({"ridge": ridge, "cases": len(group),
            "baseline_teacher_rmse_unit": float(np.mean([
                row["baseline_teacher_rmse_unit"] for row in group])),
            "dual_teacher_rmse_unit": float(np.mean([
                row["dual_teacher_rmse_unit"] for row in group])),
            "baseline_held_match_robust_px": float(np.mean([
                row["baseline_held_match_robust_px"] for row in group])),
            "dual_held_match_robust_px": float(np.mean([
                row["dual_held_match_robust_px"] for row in group])),
            "mean_two_forward_seconds": float(np.mean([
                row["two_forward_seconds"] for row in group]))})
    result = {"mode": "18 case-disjoint ACROBAT image/match-only validation",
              "ridges": ridges, "aggregates": aggregates, "rows": rows,
              "anatomical_labels_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregates), flush=True)


if __name__ == "__main__":
    main()
