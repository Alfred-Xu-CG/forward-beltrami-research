"""Frozen-checkpoint blank-image control for the ACROBAT pseudo-teacher probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from tools.digital_acrobat_teacher_probe import _map_rmse, load_case
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_network_teacher import load_checkpoint


def actual_and_blank_outputs(model: torch.nn.Module, fixed: torch.Tensor,
                             moving: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run unchanged model on real pair and constant-zero grayscale pair."""
    actual, _, _ = model(fixed, moving)
    blank, _, _ = model(torch.zeros_like(fixed), torch.zeros_like(moving))
    return actual, blank


def run(root: Path, output: Path, *, device: str = "cpu") -> dict:
    folds = (
        ("8train2test", (495, 733)),
        ("100_638", (100, 638)),
        ("156_585", (156, 585)),
        ("315_586", (315, 586)),
        ("330_399", (330, 399)),
    )
    target_device = torch.device(device)
    output.mkdir(parents=True, exist_ok=True)
    reference = identity_vertices(257, device=target_device).cpu().numpy().astype(np.float32)
    results = []
    for fold, test_ids in folds:
        checkpoint = root / f"acrobat_teacher_probe_{fold}" / "acrobat_teacher_probe_weights.npz"
        model = load_checkpoint(checkpoint, device=device).eval()
        with torch.no_grad():
            for case in test_ids:
                example = load_case(root, case, target_device)
                actual, blank = actual_and_blank_outputs(
                    model, example["fixed"], example["prewarped"])
                matrix, offset = example["matrix"], example["offset"]
                actual_full = Q1ImageRegistrationNetwork.apply_affine(actual, matrix, offset)
                blank_full = Q1ImageRegistrationNetwork.apply_affine(blank, matrix, offset)
                archive_path = output / f"{case}_blank_safe_q1.npz"
                np.savez_compressed(
                    archive_path, vertices=blank.cpu().numpy().astype(np.float32),
                    boundary_reference=reference,
                    post_affine_matrix=matrix[0].cpu().numpy().astype(np.float32),
                    post_affine_offset=offset[0].cpu().numpy().astype(np.float32),
                )
                certificate = certify_q1_binary_map(archive_path)
                if not certificate["valid"]:
                    raise ArithmeticError(f"invalid blank map for case {case}")
                results.append({
                    "case": case, "fold": fold,
                    "actual_to_DHR_full_vertex_rmse": _map_rmse(actual_full, example["raw_teacher"]),
                    "blank_to_DHR_full_vertex_rmse": _map_rmse(blank_full, example["raw_teacher"]),
                    "actual_minus_blank_map_vertex_rmse": _map_rmse(actual_full, blank_full),
                    "blank_saved_certificate": certificate,
                })
    result = {
        "mode": "frozen_checkpoint_constant_white_image_ablation",
        "full_DHR_used_for_inference": False,
        "test_case_ids": [item["case"] for item in results],
        "cases": results,
    }
    (output / "blank_ablation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
