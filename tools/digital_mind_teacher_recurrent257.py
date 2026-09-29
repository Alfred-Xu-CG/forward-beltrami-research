"""Train safe 257² recurrent P1 head from DHR pseudo-fields on ACROBAT.

Teacher fields are used only as training/validation targets, never as model
inputs. This establishes whether missing anatomical supervision, as opposed
to safe decoder capacity, explains weak image-objective transfer. It is not
an independent registration-ground-truth dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.digital_acrobat_teacher_probe import load_case_inputs, load_case_teacher
from tools.digital_mind_amortized_network import image_features, split_ids, structural_loss
from tools.digital_mind_dense257_head import DenseSafeHead, dense_features, load_global_base
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_mind_safe_optimize import strain_penalty


def train(root: Path, selection: Path, base_checkpoint: Path,
          one_pass_checkpoint: Path, output: Path, *, steps: int, batch_size: int,
          device_name: str, map_weight: float = 1000.,
          image_weight: float = .1, strain_weight: float = .1) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    ids = json.loads(selection.read_text(encoding="utf-8"))["combined_train_ids"]
    train_ids, val_ids = split_ids(ids)
    device = torch.device(device_name)
    torch.manual_seed(20261004)
    rng = np.random.default_rng(20261004)
    base = load_global_base(base_checkpoint, device)
    saved_one = torch.load(one_pass_checkpoint, map_location=device, weights_only=False)
    one = DenseSafeHead().to(device)
    one.load_state_dict(saved_one["head_state_dict"])
    one.eval()
    model = RecurrentDenseSafeHead(passes=4).to(device)
    model.initialize_from_one_pass(one)
    cache = {}
    prep_started = time.perf_counter()
    with torch.no_grad():
        for case in ids:
            item = load_case_inputs(root, case, device, affine_source="DHR")
            teacher = load_case_teacher(root, case, device, item["matrix"],
                                        item["offset"])["target"]
            feature, _, _, _ = image_features(
                F.interpolate(item["fixed"], size=(128, 128), mode="area"),
                F.interpolate(item["prewarped"], size=(128, 128), mode="area"))
            coarse = base(feature, final_side=257)
            fine, fdesc, mdesc, mask = dense_features(
                feature, item["fixed"], item["prewarped"], coarse,
                feature_geometry="legacy_align_corners")
            cache[case] = tuple(t.detach() for t in (
                coarse, fine, fdesc, mdesc, mask, teacher))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    precompute_seconds = time.perf_counter() - prep_started

    def evaluate() -> list[dict]:
        model.eval()
        rows = []
        with torch.no_grad():
            for case in val_ids:
                coarse, fine, fdesc, mdesc, mask, teacher = cache[case]
                old, new = one(coarse, fine), model(coarse, fine)
                rows.append({
                    "case": case,
                    "old_map_rmse_unit": float(
                        (old - teacher).square().sum(-1).mean().sqrt()),
                    "new_map_rmse_unit": float(
                        (new - teacher).square().sum(-1).mean().sqrt()),
                    "old_interior_map_rmse_unit": float(
                        (old[:, 1:-1, 1:-1] - teacher[:, 1:-1, 1:-1])
                        .square().sum(-1).mean().sqrt()),
                    "new_interior_map_rmse_unit": float(
                        (new[:, 1:-1, 1:-1] - teacher[:, 1:-1, 1:-1])
                        .square().sum(-1).mean().sqrt()),
                    "old_image": float(structural_loss(fdesc, mdesc, mask, old)),
                    "new_image": float(structural_loss(fdesc, mdesc, mask, new)),
                    "old_strain": float(strain_penalty(old)),
                    "new_strain": float(strain_penalty(new)),
                })
        return rows

    before = evaluate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-4)
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        selected = rng.choice(train_ids, size=batch_size, replace=True)
        examples = [cache[int(case)] for case in selected]
        coarse, fine, fdesc, mdesc, mask, teacher = (
            torch.cat([ex[index] for ex in examples], dim=0)
            for index in range(6))
        optimizer.zero_grad(set_to_none=True)
        mapped = model(coarse, fine)
        map_mse = (mapped[:, 1:-1, 1:-1] - teacher[:, 1:-1, 1:-1]).square().sum(-1).mean()
        image = structural_loss(fdesc, mdesc, mask, mapped)
        strain = strain_penalty(mapped)
        total = map_weight * map_mse + image_weight * image + strain_weight * strain
        total.backward()
        if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                   for parameter in model.parameters() if parameter.requires_grad):
            raise FloatingPointError(f"nonfinite teacher-head VJP at {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "map_mse": float(map_mse.detach()),
                          "image": float(image.detach()),
                          "strain": float(strain.detach()),
                          "total": float(total.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    after = evaluate()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "passes": 4,
        "train_ids": train_ids, "validation_ids": val_ids,
        "teacher": "DHR_full_field", "affine_source": "DHR",
        "steps": steps, "map_weight": map_weight,
        "image_weight": image_weight, "strain_weight": strain_weight,
    }, output)
    report = {
        "method": "DHR pseudo-field supervised 4-pass safe257 recurrent head",
        "source_case_count": len(ids), "train_count": len(train_ids),
        "validation_count": len(val_ids), "steps": steps,
        "batch_size": batch_size, "map_weight": map_weight,
        "image_weight": image_weight, "strain_weight": strain_weight,
        "precompute_seconds": precompute_seconds,
        "training_seconds": train_seconds,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "validation_initial": before, "validation_final": after,
        "trace": trace, "teacher_fields_used_as_training_targets": True,
        "teacher_fields_used_as_inference_inputs": False,
        "anatomical_landmarks_used": False,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "selection", "base_checkpoint",
                 "one_pass_checkpoint", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = train(args.root, args.selection, args.base_checkpoint,
                   args.one_pass_checkpoint, args.output,
                   steps=args.steps, batch_size=args.batch,
                   device_name=args.device)
    print(json.dumps({key: report[key] for key in (
        "train_count", "validation_count", "training_seconds",
        "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
