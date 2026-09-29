"""Frozen train/evaluate split for selected fresh ACROBAT pseudo-teacher cases.

The train command never receives held-out IDs or fields. The evaluate command
requires a saved training manifest and refuses overlap. Full DHR is only a
pseudo-label/evaluation comparator; initial-only DHR affine remains an external
inference input and its cost is not amortized into CNN latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from tools.digital_acrobat_blank_ablation import blank_output
from tools.digital_acrobat_teacher_probe import (
    _map_rmse, load_case, load_case_inputs, load_case_teacher,
)
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_network_teacher import _save_checkpoint, load_checkpoint


def validate_train_ids(ids: list[int]) -> list[int]:
    if not ids or len(set(ids)) != len(ids) or any(case <= 0 for case in ids):
        raise ValueError("nonempty distinct positive training case IDs required")
    return list(ids)


def validate_eval_ids(ids: list[int], train_ids: list[int]) -> list[int]:
    if not ids or len(set(ids)) != len(ids) or any(case <= 0 for case in ids) or (
        set(ids) & set(train_ids)
    ):
        raise ValueError("evaluation IDs must be positive, distinct and disjoint from training")
    return list(ids)


def train_only(*, root: Path, output: Path, train_ids: list[int], steps: int,
               batch: int, device: str, seed: int = 20260929,
               learning_rate: float = .002, flow_hint: bool = False) -> dict:
    train_ids = validate_train_ids(train_ids)
    if steps < 1 or batch < 1 or learning_rate <= 0:
        raise ValueError("positive steps, batch and learning rate required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("fresh training output must not overwrite a prior run")
    target_device = torch.device(device)
    torch.manual_seed(seed)
    # Only training IDs enter this loader. No held-out field path is constructed.
    examples = [load_case(root, case, target_device) for case in train_ids]
    model = Q1ImageRegistrationNetwork(seed_side=17, final_side=257,
                                       width=16, feature_side=257,
                                       flow_hint=flow_hint).to(target_device)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)
    rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    durations: list[float] = []
    losses: list[float] = []
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    model.train()
    for step in range(steps):
        selected = torch.randint(len(examples), (batch,), generator=rng).tolist()
        minibatch = [examples[index] for index in selected]
        fixed = torch.cat([item["fixed"] for item in minibatch])
        prewarped = torch.cat([item["prewarped"] for item in minibatch])
        target = torch.cat([item["target"] for item in minibatch])
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        residual, _, _ = model(fixed, prewarped)
        loss = (residual[:, 1:-1, 1:-1] - target[:, 1:-1, 1:-1]).square().sum(-1).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite training loss at step {step}")
        loss.backward()
        gradients = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        if not gradients or any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError(f"nonfinite gradient at step {step}")
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        durations.append(time.perf_counter() - start)
        losses.append(float(loss.detach()))
    report = {
        "mode": "fresh_ACROBAT_train_only_existing_safe_Q1_CNN",
        "train_case_ids": train_ids,
        "test_case_ids_available_to_training": [],
        "steps": steps, "batch": batch, "seed": seed,
        "learning_rate": learning_rate,
        "flow_hint": flow_hint,
        "first_minibatch_loss": losses[0], "last_minibatch_loss": losses[-1],
        "median_training_step_seconds": statistics.median(durations),
        "peak_torch_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda" else None
        ),
        "external_initial_DHR_affine_required": True,
        "full_DHR_fields_loaded_only_for_training_ids": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(output / "frozen_weights.npz", model)
    (output / "train_manifest.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
    return report


def evaluate_frozen(*, root: Path, output: Path, test_ids: list[int],
                    device: str) -> dict:
    with (output / "train_manifest.json").open(encoding="utf-8") as stream:
        training = json.load(stream)
    ids = validate_eval_ids(test_ids, training["train_case_ids"])
    if (output / "heldout_report.json").exists() or any(
        (output / f"{case}_{arm}_safe_q1.npz").exists()
        for case in ids for arm in ("actual", "blank")
    ):
        raise FileExistsError("existing held-out evaluation must not be overwritten")
    target_device = torch.device(device)
    model = load_checkpoint(output / "frozen_weights.npz", device=device).eval()
    identity = identity_vertices(257, device=target_device)
    reference = identity.cpu().numpy().astype(np.float32)
    cases = []
    sealed_predictions = []
    with torch.no_grad():
        for case in ids:
            # Seal every prediction before opening any held-out full-DHR label.
            example = load_case_inputs(root, case, target_device)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            start = time.perf_counter()
            actual, _, _ = model(example["fixed"], example["prewarped"])
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            inference_seconds = time.perf_counter() - start
            blank = blank_output(model, example["fixed"], example["prewarped"])
            matrix, offset = example["matrix"], example["offset"]
            actual_full = Q1ImageRegistrationNetwork.apply_affine(actual, matrix, offset)
            blank_full = Q1ImageRegistrationNetwork.apply_affine(blank, matrix, offset)
            affine_full = Q1ImageRegistrationNetwork.apply_affine(identity, matrix, offset)
            certificates = {}
            for label, residual in (("actual", actual), ("blank", blank)):
                archive_path = output / f"{case}_{label}_safe_q1.npz"
                np.savez_compressed(
                    archive_path,
                    vertices=residual.cpu().numpy().astype(np.float32),
                    boundary_reference=reference,
                    post_affine_matrix=matrix[0].cpu().numpy().astype(np.float32),
                    post_affine_offset=offset[0].cpu().numpy().astype(np.float32),
                )
                certificate = certify_q1_binary_map(archive_path)
                if not certificate["valid"]:
                    raise ArithmeticError(f"{case} {label} saved Q1 map invalid")
                certificates[label] = certificate
            sealed_predictions.append((case, matrix, offset, actual_full,
                                       blank_full, affine_full, inference_seconds,
                                       certificates))
        for (case, matrix, offset, actual_full, blank_full, affine_full,
             inference_seconds, certificates) in sealed_predictions:
            full = load_case_teacher(root, case, target_device, matrix,
                                     offset)["raw_teacher"]
            cases.append({
                "case": case,
                "actual_to_DHR_full_vertex_rmse": _map_rmse(actual_full, full),
                "blank_to_DHR_full_vertex_rmse": _map_rmse(blank_full, full),
                "initial_affine_to_DHR_full_vertex_rmse": _map_rmse(affine_full, full),
                "actual_minus_blank_vertex_rmse": _map_rmse(actual_full, blank_full),
                "network_forward_seconds_excluding_affine_prewarp_io": inference_seconds,
                "certificates": certificates,
            })
    report = {
        "mode": "fresh_ACROBAT_frozen_heldout_pseudoteacher_evaluation",
        "train_case_ids": training["train_case_ids"], "test_case_ids": ids,
        "external_initial_DHR_affine_required": True,
        "full_DHR_teacher_used_during_model_inference": False,
        "all_prediction_archives_sealed_before_first_full_DHR_teacher_read": True,
        "cases": cases,
    }
    (output / "heldout_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--root", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--train-ids", nargs="+", type=int, required=True)
    training.add_argument("--steps", type=int, default=800)
    training.add_argument("--batch", type=int, default=4)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--seed", type=int, default=20260929)
    training.add_argument("--flow-hint", action="store_true",
                          help="override the predeclared no-flow-hint architecture")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--root", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--test-ids", nargs="+", type=int, required=True)
    evaluation.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "train":
        result = train_only(root=args.root, output=args.output,
                            train_ids=args.train_ids, steps=args.steps,
                            batch=args.batch, device=args.device, seed=args.seed,
                            flow_hint=args.flow_hint)
    else:
        result = evaluate_frozen(root=args.root, output=args.output,
                                 test_ids=args.test_ids, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
