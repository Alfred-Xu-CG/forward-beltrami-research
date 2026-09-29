"""Post-hoc train-set fit diagnostic for a frozen ACROBAT Q1 network.

This reads training pseudo-labels only. It does not retrain a model or alter a
held-out prediction, and its report must not be described as test accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from tools.digital_acrobat_blank_ablation import blank_output
from tools.digital_acrobat_teacher_probe import load_case
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_network_teacher import load_checkpoint


def split_vertex_rmse(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Vector RMSE per vertex, split by strictly interior versus boundary nodes."""
    if predicted.shape != target.shape or predicted.ndim != 4 or predicted.shape[-1] != 2:
        raise ValueError("matching BHWC coordinate tensors are required")
    if min(predicted.shape[1:3]) < 3:
        raise ValueError("at least 3x3 vertices required for an interior")
    squared = (predicted - target).square().sum(dim=-1)
    boundary = torch.ones_like(squared, dtype=torch.bool)
    boundary[:, 1:-1, 1:-1] = False
    return {
        "all": math.sqrt(float(squared.mean())),
        "interior": math.sqrt(float(squared[:, 1:-1, 1:-1].mean())),
        "boundary": math.sqrt(float(squared[boundary].mean())),
    }


def evaluate_training_fit(*, root: Path, output: Path, device: str) -> dict:
    report_path = output / "train_fit_report.json"
    if report_path.exists():
        raise FileExistsError(f"training-fit report already exists: {report_path}")
    with (output / "train_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    target_device = torch.device(device)
    model = load_checkpoint(output / "frozen_weights.npz", device=device).eval()
    identity = identity_vertices(257, device=target_device)
    cases: list[dict] = []
    with torch.no_grad():
        for case in manifest["train_case_ids"]:
            example = load_case(root, case, target_device)
            actual, _, _ = model(example["fixed"], example["prewarped"])
            blank = blank_output(model, example["fixed"], example["prewarped"])
            matrix, offset = example["matrix"], example["offset"]
            target = example["raw_teacher"]
            predictions = {
                "actual": Q1ImageRegistrationNetwork.apply_affine(actual, matrix, offset),
                "blank": Q1ImageRegistrationNetwork.apply_affine(blank, matrix, offset),
                "initial_affine": Q1ImageRegistrationNetwork.apply_affine(identity, matrix, offset),
            }
            cases.append({"case": case, **{
                arm: split_vertex_rmse(prediction, target)
                for arm, prediction in predictions.items()
            }})
    report = {
        "mode": "posthoc_frozen_training_fit_not_heldout_accuracy",
        "train_case_ids": manifest["train_case_ids"],
        "cases": cases,
        "mean_per_case": {
            arm: {
                region: sum(row[arm][region] for row in cases) / len(cases)
                for region in ("all", "interior", "boundary")
            }
            for arm in ("actual", "blank", "initial_affine")
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate_training_fit(root=args.root, output=args.output,
                                           device=args.device), indent=2))


if __name__ == "__main__":
    main()
