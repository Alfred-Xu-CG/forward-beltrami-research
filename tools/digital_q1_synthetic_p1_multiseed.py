"""Repeat the fixed synthetic slide-texture P1 image-loss probe across seeds.

The three held-out texture groups and coefficient lists are unchanged from
digital_q1_synthetic_slide_holdout.py. Only network initialization and the
training minibatch order vary. These are development folds, not new patients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.digital_q1_synthetic_slide_holdout import run


FOLDS = {
    "histo": ([2, 3, 4, 5], [0, 1]),
    "lesion": ([0, 1, 4, 5], [2, 3]),
    "kidney": ([0, 1, 2, 3], [4, 5]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS), action="append",
                        default=None)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--additional-train-texture-index", type=int,
                        action="append", default=[])
    args = parser.parse_args()
    if len(set(args.seed)) != len(args.seed):
        raise ValueError("repeated seed")
    with np.load(args.textures) as archive:
        textures = torch.from_numpy(archive["images"].copy())
        sources = archive["sources"].tolist() if "sources" in archive else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fold in args.fold or list(FOLDS):
        base_train_indices, test_indices = FOLDS[fold]
        train_indices = base_train_indices + args.additional_train_texture_index
        if len(set(train_indices)) != len(train_indices) or (
            set(train_indices) & set(test_indices)
        ):
            raise ValueError("additional training textures overlap a fold")
        for seed in args.seed:
            prefix = args.output_dir / f"p1_image_{fold}_seed{seed}"
            report = run(
                textures=textures, train_texture_indices=train_indices,
                test_texture_indices=test_indices, side=257, steps=args.steps,
                batch=4, train_count=128, test_count=64,
                device=args.device, loss_mode="p1_image",
                model_seed=seed, minibatch_seed=seed + 3,
                output_example=prefix.with_suffix(".npz"),
                output_weights=Path(f"{prefix}_weights.npz"),
            )
            report["texture_sources"] = sources
            Path(f"{prefix}_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            summary = {
                "fold": fold,
                "train_texture_indices": train_indices,
                "test_texture_indices": test_indices,
                "model_seed": seed,
                "minibatch_seed": seed + 3,
                "test_map_rmse_mean": report["test_map_rmse_mean"],
                "test_identity_rmse_mean": report["test_identity_rmse_mean"],
                "test_image_mse_mean": report["test_image_mse_mean"],
                "train_step_seconds_median": report["train_step_seconds_median"],
                "training_peak_cuda_allocated_bytes": report[
                    "training_peak_cuda_allocated_bytes"
                ],
                "saved_example_exact_q1_valid": report[
                    "saved_example_certificate"
                ]["valid"],
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (args.output_dir / "p1_image_multiseed_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
