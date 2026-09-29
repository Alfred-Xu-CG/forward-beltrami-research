"""Exploratory ACROBAT training of image-conditioned repeated Q1 corrections.

This follow-up begins from a frozen 102-case one-pass model. Confirmation
labels were already opened before this architecture test, so any repeat use
of those cases is explicitly exploratory, not a second blind confirmation.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.q1_recurrent_image_network import (
    RecurrentQ1ImageRegistrationNetwork,
)
from tools.digital_acrobat_blank_ablation import blank_output
from tools.digital_acrobat_fresh_probe import dihedral_augment, validate_eval_ids
from tools.digital_acrobat_teacher_probe import load_case, load_case_inputs
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_network_teacher import load_checkpoint


ROUNDS = {17: 4, 33: 2, 65: 1}


def parse_rounds(value: str) -> dict[int, int]:
    try:
        pairs = [part.split(":") for part in value.split(",")]
        rounds = {int(side): int(count) for side, count in pairs}
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("expected side:rounds comma-separated") from error
    if len(pairs) != len(rounds) or not rounds or any(len(pair) != 2 for pair in pairs):
        raise argparse.ArgumentTypeError("unique side:rounds pairs required")
    if any(side not in (17, 33, 65, 129, 257) or count < 1 for side, count in rounds.items()):
        raise argparse.ArgumentTypeError("sides must be 17,33,65,129,257 with positive rounds")
    return rounds


def _save_heads(path: Path, model: RecurrentQ1ImageRegistrationNetwork) -> None:
    np.savez_compressed(path, **{
        "state__" + key: value.detach().cpu().numpy()
        for key, value in model.heads.state_dict().items()
    })


def _load_model(output: Path, device: str) -> RecurrentQ1ImageRegistrationNetwork:
    base = load_checkpoint(output / "base_frozen_weights.npz", device=device)
    training = json.loads((output / "train_manifest.json").read_text())
    schedule = {int(side): int(count)
                for side, count in training["rounds_by_side"].items()}
    modes = training.get("proposal_modes_by_side")
    modes = ({int(side): mode for side, mode in modes.items()}
             if modes is not None else None)
    families = training.get("update_families_by_side")
    families = ({int(side): family for side, family in families.items()}
                if families is not None else None)
    model = RecurrentQ1ImageRegistrationNetwork(
        base, rounds_by_side=schedule,
        refresh_moving_evidence=training.get("refresh_moving_evidence", True),
        proposal_mode=training.get("proposal_mode", "current_edge"),
        proposal_modes_by_side=modes,
        update_families_by_side=families,
        f2_raw_span=training.get("f2_raw_span", .5),
        f2_patch_cells=training.get("f2_patch_cells", 8),
        f2_accepted_gain=training.get("f2_accepted_gain", 1.0),
        checkpoint_rounds=training.get("checkpoint_rounds", False),
    ).to(device)
    with np.load(output / "recurrent_heads.npz") as archive:
        state = {
            key[len("state__"):]: torch.from_numpy(archive[key].copy())
            for key in archive.files if key.startswith("state__")
        }
    model.heads.load_state_dict(state, strict=True)
    return model.eval()


def train(root: Path, output: Path, base_checkpoint: Path,
          selection: Path, *, steps: int, batch: int, device: str,
          seed: int = 20261004,
          learning_rate: float = .002,
          rounds_by_side: dict[int, int] | None = None,
          refresh_moving_evidence: bool = True,
          proposal_mode: str = "current_edge",
          update_family: str = "f1",
          f2_raw_span: float = .5,
          f2_patch_cells: int = 8,
          f2_accepted_gain: float = 1.0,
          checkpoint_rounds: bool = False,
          initialize_from: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("fresh recurrent output required")
    if steps < 1 or batch < 1:
        raise ValueError("positive steps and batch required")
    if not 0 < learning_rate < 1:
        raise ValueError("learning rate must be in (0,1)")
    train_ids = json.loads(selection.read_text(encoding="utf-8"))["combined_train_ids"]
    target_device = torch.device(device)
    torch.manual_seed(seed)
    examples = [load_case(root, case, target_device) for case in train_ids]
    schedule = rounds_by_side or ROUNDS
    prior = _load_model(initialize_from, device) if initialize_from is not None else None
    if prior is not None:
        prior_training = json.loads((initialize_from / "train_manifest.json").read_text())
        if set(prior_training["train_case_ids"]) != set(train_ids):
            raise ValueError("initialized heads require the identical training-case set")
    if prior is not None and not set(prior.heads).issubset({str(side) for side in schedule}):
        raise ValueError("initialized heads must be present in new schedule")
    if prior is not None and any(
        schedule[side] != count for side, count in prior.rounds_by_side.items()
    ):
        raise ValueError("initialized recurrent round counts must stay unchanged")
    if prior is not None and prior.refresh_moving_evidence != refresh_moving_evidence:
        raise ValueError("initialized recurrence must keep prior image-evidence policy")
    if prior is not None and any(
        family in ("f2", "f2_f1") for family in prior.update_families_by_side.values()
    ) and (prior.f2_raw_span != f2_raw_span
           or prior.f2_patch_cells != f2_patch_cells
           or prior.f2_accepted_gain != f2_accepted_gain):
        raise ValueError("initialized F2 heads require their prior span and gain")
    if prior is not None and not filecmp.cmp(
        base_checkpoint, initialize_from / "base_frozen_weights.npz", shallow=False,
    ):
        raise ValueError("initialized recurrence must keep identical frozen base")
    modes = {side: proposal_mode for side in schedule}
    families = {side: update_family for side in schedule}
    if prior is not None:
        modes.update(prior.proposal_modes_by_side)
        families.update(prior.update_families_by_side)
    model = RecurrentQ1ImageRegistrationNetwork(
        load_checkpoint(base_checkpoint, device=device), rounds_by_side=schedule,
        refresh_moving_evidence=refresh_moving_evidence,
        proposal_mode=proposal_mode,
        proposal_modes_by_side=modes,
        update_families_by_side=families,
        f2_raw_span=f2_raw_span,
        f2_patch_cells=f2_patch_cells,
        f2_accepted_gain=f2_accepted_gain,
        checkpoint_rounds=checkpoint_rounds,
    ).to(target_device).train()
    initialized_sides = []
    if prior is not None:
        for side, head in prior.heads.items():
            model.heads[side].load_state_dict(head.state_dict(), strict=True)
            for parameter in model.heads[side].parameters():
                parameter.requires_grad_(False)
            initialized_sides.append(int(side))
        del prior
    trainable_sides = sorted(set(schedule) - set(initialized_sides))
    if not trainable_sides:
        raise ValueError("initialization leaves no trainable new head")
    trainable = [parameter for side in trainable_sides
                 for parameter in model.heads[str(side)].parameters()]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    sample_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    augment_rng = torch.Generator(device="cpu").manual_seed(seed + 2)
    durations = []
    loss_trace = []
    first_loss = last_loss = None
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    for step in range(steps):
        chosen = torch.randint(len(examples), (batch,), generator=sample_rng).tolist()
        prepared = [dihedral_augment(
            examples[index]["fixed"], examples[index]["prewarped"],
            examples[index]["target"],
            turns=int(torch.randint(4, (1,), generator=augment_rng)),
            flip=bool(torch.randint(2, (1,), generator=augment_rng)),
        ) for index in chosen]
        fixed, moving, target = (
            torch.cat([item[position] for item in prepared])
            for position in range(3)
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        residual, _, _ = model(fixed, moving)
        loss = (residual[:, 1:-1, 1:-1]
                - target[:, 1:-1, 1:-1]).square().sum(-1).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite recurrent loss at step {step}")
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
               for parameter in trainable):
            raise FloatingPointError(f"nonfinite/missing recurrent gradient at step {step}")
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        durations.append(time.perf_counter() - started)
        last_loss = float(loss.detach())
        first_loss = last_loss if first_loss is None else first_loss
        if (step + 1) % 100 == 0 or step == 0:
            loss_trace.append({"step": step + 1, "minibatch_loss": last_loss})
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_checkpoint, output / "base_frozen_weights.npz")
    _save_heads(output / "recurrent_heads.npz", model)
    report = {
        "mode": "exploratory_recurrent_frozen_base_pseudoteacher_training",
        "train_case_ids": train_ids,
        "test_case_ids_available_to_training": [],
        "steps": steps, "batch": batch, "seed": seed,
        "learning_rate": learning_rate, "dihedral_augmentation": True,
        "loss_trace": loss_trace,
        "rounds_by_side": schedule,
        "refresh_moving_evidence": refresh_moving_evidence,
        "proposal_mode": proposal_mode,
        "proposal_modes_by_side": modes,
        "update_families_by_side": families,
        "f2_raw_span": f2_raw_span,
        "f2_patch_cells": f2_patch_cells,
        "f2_accepted_gain": f2_accepted_gain,
        "checkpoint_rounds": checkpoint_rounds,
        "initialized_head_sides": initialized_sides,
        "trainable_head_sides": trainable_sides,
        "initialize_from": str(initialize_from) if initialize_from is not None else None,
        "frozen_base": True,
        "confirmation_teacher_previously_opened_before_architecture_test": True,
        "first_minibatch_loss": first_loss,
        "last_minibatch_loss": last_loss,
        "median_training_step_seconds": statistics.median(durations),
        "peak_torch_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda" else None
        ),
    }
    (output / "train_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def predict(root: Path, output: Path, ids: list[int], *, device: str) -> dict:
    training = json.loads((output / "train_manifest.json").read_text())
    ids = validate_eval_ids(ids, training["train_case_ids"])
    if (output / "prediction_manifest.json").exists() or any(
        (output / f"{case}_{arm}_safe_q1.npz").exists()
        for case in ids for arm in ("actual", "blank")
    ):
        raise FileExistsError("existing recurrent predictions must not be overwritten")
    model = _load_model(output, device)
    target_device = torch.device(device)
    reference = identity_vertices(257, device=target_device).cpu().numpy().astype(np.float32)
    cases = []
    with torch.no_grad():
        for case in ids:
            example = load_case_inputs(root, case, target_device)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            started = time.perf_counter()
            actual = model(example["fixed"], example["prewarped"])[0]
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            seconds = time.perf_counter() - started
            blank = blank_output(model, example["fixed"], example["prewarped"])
            certificates = {}
            for arm, mapped in (("actual", actual), ("blank", blank)):
                archive = output / f"{case}_{arm}_safe_q1.npz"
                np.savez_compressed(
                    archive, vertices=mapped.cpu().numpy().astype(np.float32),
                    boundary_reference=reference,
                    post_affine_matrix=example["matrix"][0].cpu().numpy().astype(np.float32),
                    post_affine_offset=example["offset"][0].cpu().numpy().astype(np.float32),
                )
                certificates[arm] = certify_q1_binary_map(archive)
                if not certificates[arm]["valid"]:
                    raise ArithmeticError(f"{case} {arm} recurrent map invalid")
            cases.append({"case": case,
                          "network_forward_seconds_excluding_affine_prewarp_io": seconds,
                          "certificates": certificates})
    report = {
        "mode": "exploratory_recurrent_prediction_no_teacher_loaded",
        "train_case_ids": training["train_case_ids"],
        "test_case_ids": ids,
        "full_DHR_teacher_used": False,
        "confirmation_teacher_previously_opened_before_architecture_test": True,
        "cases": cases,
    }
    (output / "prediction_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--root", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--base-checkpoint", type=Path, required=True)
    training.add_argument("--selection", type=Path, required=True)
    training.add_argument("--steps", type=int, default=2400)
    training.add_argument("--batch", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=.002)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--rounds", type=parse_rounds, default=ROUNDS,
                          help="e.g. 17:1,33:1,65:1; default 17:4,33:2,65:1")
    training.add_argument("--freeze-moving-evidence", action="store_true",
                          help="reuse each level-entry image sample through its rounds")
    training.add_argument("--proposal-mode", choices=("current_edge", "fixed_h",
                                                    "fixed_h_soft"),
                          default="current_edge")
    training.add_argument("--update-family", choices=("f1", "f2", "f2_f1"), default="f1",
                          help="family for newly added sides; initialized sides retain their own")
    training.add_argument("--f2-raw-span", type=float, default=.5,
                          help="F2 current-edge span; default .5, does not change F1 span")
    training.add_argument("--f2-patch-cells", type=int, default=8)
    training.add_argument("--f2-accepted-gain", type=float, default=1.,
                          help="multiply each already safe F2 patch step by a number in (0,1]")
    training.add_argument("--initialize-from", type=Path,
                          help="copy and freeze existing recurrent heads; train added sides")
    training.add_argument("--checkpoint-rounds", action="store_true",
                          help="recompute each recurrent round during backward to save memory")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--root", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--test-ids", nargs="+", type=int, required=True)
    prediction.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "train":
        result = train(args.root, args.output, args.base_checkpoint,
                       args.selection, steps=args.steps, batch=args.batch,
                       device=args.device, rounds_by_side=args.rounds,
                       learning_rate=args.learning_rate,
                       refresh_moving_evidence=not args.freeze_moving_evidence,
                       proposal_mode=args.proposal_mode,
                       update_family=args.update_family,
                       f2_raw_span=args.f2_raw_span,
                       f2_patch_cells=args.f2_patch_cells,
                       f2_accepted_gain=args.f2_accepted_gain,
                       checkpoint_rounds=args.checkpoint_rounds,
                       initialize_from=args.initialize_from)
    else:
        result = predict(args.root, args.output, args.test_ids, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
