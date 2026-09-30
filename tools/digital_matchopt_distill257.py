"""Distill offline image/match-optimized safe maps into one fast 4-pass head.

Only image-derived affines and matches enter inference. Offline optimized
maps supervise training; held-out case IDs are used only for error reporting.
No anatomical landmarks or DHR full fields are read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_amortized_network import image_features, split_ids, structural_loss
from tools.digital_mind_dense257_head import DenseSafeHead, dense_features, load_global_base
from tools.digital_mind_match_conditioned257 import gaussian_match_features
from tools.digital_image_force_feature257 import image_force_features
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_mind_sparse_match_finetune import p1_at_points, robust_match_loss


def train(root: Path, selection: Path, matches: Path, teachers: Path,
          base_checkpoint: Path, initial_checkpoint: Path, output: Path, *,
          steps: int, batch_size: int, device_name: str,
          map_weight: float = 1000., image_weight: float = .1,
          match_weight: float = .01, strain_weight: float = .02,
          kernel_maps: Path | None = None,
          frozen_maps: Path | None = None,
          fit_one_case: int | None = None,
          trunk_from_onehead: Path | None = None,
          context: str = "local",
          match_channels: int = 6,
          evidence_mode: str = "gaussian",
          teacher_suffix: str = "matchopt4_safe257") -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if steps < 1 or batch_size < 1:
        raise ValueError("positive steps and batch required")
    if kernel_maps is not None and frozen_maps is not None:
        raise ValueError("choose one initial map type")
    if evidence_mode not in ("gaussian", "gaussian_force", "gaussian_zeros",
                             "force_only", "force_recurrent",
                             "gaussian_force_recurrent") or (
        evidence_mode in ("gaussian_force", "gaussian_zeros",
                          "gaussian_force_recurrent")
        and match_channels != 12
    ) or (evidence_mode in ("force_only", "force_recurrent")
          and match_channels != 6
    ) or (evidence_mode in ("force_recurrent", "gaussian_force_recurrent")
          and context == "multilevel_unet"
    ):
        raise ValueError("evidence mode and channel count disagree")
    ids = json.loads(selection.read_text(encoding="utf-8"))["combined_train_ids"]
    train_ids, val_ids = split_ids(ids)
    device = torch.device(device_name)
    torch.manual_seed(20261005)
    rng = np.random.default_rng(20261005)
    base = load_global_base(base_checkpoint, device)
    model = RecurrentDenseSafeHead(passes=4, match_channels=match_channels,
                                   context=context).to(device)
    if trunk_from_onehead is None:
        residual_feature_geometry = "legacy_align_corners"
        saved = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        if (int(saved.get("match_channels", 0)) != match_channels or
                int(saved["passes"]) != 4):
            raise ValueError("initial checkpoint's match channels or pass count differ")
        model.load_state_dict(saved["model_state_dict"])
    else:
        one_saved = torch.load(trunk_from_onehead, map_location=device,
                               weights_only=False)
        residual_feature_geometry = one_saved.get(
            "feature_geometry", "legacy_align_corners")
        one = DenseSafeHead().to(device)
        one.load_state_dict(one_saved["head_state_dict"])
        model.initialize_from_one_pass(one)
    if kernel_maps is not None or frozen_maps is not None:
        with torch.no_grad():
            for head in model.heads:
                head.weight.zero_()
                head.bias.zero_()
    cache = {}
    started = time.perf_counter()
    with torch.no_grad():
        for case in ids:
            match_path = matches / f"{case}_alignedSG_matches.npz"
            teacher_path = teachers / f"{case}_{teacher_suffix}.npz"
            affine_path = root / f"{case}_directSG_affine.npz"
            if not (match_path.exists() and teacher_path.exists()
                    and affine_path.exists()):
                continue
            kernel_path = (None if kernel_maps is None else
                           kernel_maps / f"{case}_kernelmatch4_safe257.npz")
            frozen_path = (None if frozen_maps is None else
                           frozen_maps / f"{case}_old_safe257.npz")
            initial_path = kernel_path or frozen_path
            if initial_path is not None and not initial_path.exists():
                continue
            with np.load(match_path) as data:
                source = torch.from_numpy(data["source_fixed_unit"].copy()).to(device)
                target = torch.from_numpy(data["target_aligned_unit"].copy()).to(device)
            if len(source) < 16:
                continue
            with np.load(teacher_path) as data:
                teacher = torch.from_numpy(data["vertices"].copy()).to(device)
                teacher_m = data["post_affine_matrix"]
                teacher_b = data["post_affine_offset"]
            with np.load(affine_path) as data:
                if not (np.array_equal(teacher_m, data["post_affine_matrix"])
                        and np.array_equal(teacher_b, data["post_affine_offset"])):
                    raise ValueError(f"teacher/initializer affine mismatch for {case}")
            if teacher.shape != (1, 257, 257, 2):
                raise ValueError(f"invalid teacher grid for {case}")
            item = load_case_inputs(root, case, device, affine_source="image_only")
            feature, _, _, _ = image_features(
                F.interpolate(item["fixed"], size=(128, 128), mode="area"),
                F.interpolate(item["prewarped"], size=(128, 128), mode="area"))
            coarse = base(feature, final_side=257)
            if initial_path is not None:
                with np.load(initial_path) as data:
                    if not (np.array_equal(data["post_affine_matrix"], teacher_m)
                            and np.array_equal(data["post_affine_offset"], teacher_b)):
                        raise ValueError(f"initial-map/teacher affine mismatch for {case}")
                    coarse = torch.from_numpy(data["vertices"].copy()).to(device)
            fine, fdesc, mdesc, mask = dense_features(
                feature, item["fixed"], item["prewarped"], coarse,
                feature_geometry=residual_feature_geometry)
            raster = (gaussian_match_features(source, target)
                      if match_channels and evidence_mode not in (
                          "force_only", "force_recurrent") else None)
            if evidence_mode == "gaussian_force":
                force = image_force_features(fdesc, mdesc, mask, coarse)
                raster = torch.cat((raster, force), dim=1)
            elif evidence_mode == "gaussian_zeros":
                raster = torch.cat((raster, torch.zeros_like(raster)), dim=1)
            elif evidence_mode == "force_only":
                raster = image_force_features(fdesc, mdesc, mask, coarse)
            cache[case] = tuple(x.detach() if x is not None else None for x in (
                coarse, fine, fdesc, mdesc, mask, raster, teacher,
                source, target))
    available_train = [case for case in train_ids if case in cache]
    available_val = [case for case in val_ids if case in cache]
    if fit_one_case is not None:
        if fit_one_case not in available_train:
            raise ValueError("fit-one-case must be available in training split")
        available_train = [fit_one_case]
    if not available_train or not available_val:
        raise ValueError("no train or validation pseudo-teachers")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    precompute_seconds = time.perf_counter() - started

    def run_model(coarse, fine, fdesc, mdesc, mask, raster):
        if evidence_mode == "force_recurrent":
            return model(coarse, fine, evidence_fn=lambda current:
                         image_force_features(fdesc, mdesc, mask, current))
        if evidence_mode == "gaussian_force_recurrent":
            return model(coarse, fine, evidence_fn=lambda current:
                         torch.cat((raster, image_force_features(
                             fdesc, mdesc, mask, current)), dim=1))
        return model(coarse, fine, match_feature=raster)

    def evaluate() -> list[dict]:
        model.eval()
        rows = []
        with torch.no_grad():
            for case in available_val:
                coarse, fine, fdesc, mdesc, mask, raster, teacher, source, target = cache[case]
                mapped = run_model(coarse, fine, fdesc, mdesc, mask, raster)
                rows.append({
                    "case": case, "match_count": len(source),
                    "teacher_map_rmse_from_model_unit": float(
                        (mapped - teacher).square().sum(-1).mean().sqrt()),
                    "model_image": float(structural_loss(fdesc, mdesc, mask, mapped)),
                    "teacher_image": float(structural_loss(fdesc, mdesc, mask, teacher)),
                    "model_match_robust_px": float(robust_match_loss(
                        p1_at_points(mapped, source), target)),
                    "teacher_match_robust_px": float(robust_match_loss(
                        p1_at_points(teacher, source), target)),
                    "model_strain": float(strain_penalty(mapped)),
                    "teacher_strain": float(strain_penalty(teacher)),
                })
        return rows

    before = evaluate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-4)
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        chosen = rng.choice(available_train, size=batch_size, replace=True)
        map_losses, image_losses, match_losses, strain_losses = [], [], [], []
        optimizer.zero_grad(set_to_none=True)
        for case in chosen:
            coarse, fine, fdesc, mdesc, mask, raster, teacher, source, target = cache[int(case)]
            mapped = run_model(coarse, fine, fdesc, mdesc, mask, raster)
            map_losses.append((mapped - teacher).square().sum(-1).mean())
            image_losses.append(structural_loss(fdesc, mdesc, mask, mapped))
            match_losses.append(robust_match_loss(p1_at_points(mapped, source), target))
            strain_losses.append(strain_penalty(mapped))
        map_mse = torch.stack(map_losses).mean()
        image = torch.stack(image_losses).mean()
        match = torch.stack(match_losses).mean()
        strain = torch.stack(strain_losses).mean()
        total = (map_weight * map_mse + image_weight * image
                 + match_weight * match + strain_weight * strain)
        total.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
            raise FloatingPointError(f"invalid distillation VJP at {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "map_mse": float(map_mse.detach()),
                          "image": float(image.detach()),
                          "match": float(match.detach()),
                          "strain": float(strain.detach()),
                          "total": float(total.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    after = evaluate()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "passes": 4,
        "match_channels": match_channels,
        "evidence_mode": evidence_mode,
        "context": context,
        "residual_feature_geometry": residual_feature_geometry,
        "train_ids": available_train, "validation_ids": available_val,
        "fit_one_case": fit_one_case,
        "trunk_from_onehead": (None if trunk_from_onehead is None
                               else str(trunk_from_onehead)),
        "steps": steps, "teacher": teacher_suffix,
        "kernel_maps": None if kernel_maps is None else str(kernel_maps),
        "frozen_maps": None if frozen_maps is None else str(frozen_maps),
        "zero_residual_head_initialization": (
            kernel_maps is not None or frozen_maps is not None),
        "map_weight": map_weight, "image_weight": image_weight,
        "match_weight": match_weight, "strain_weight": strain_weight,
    }, output)
    report = {
        "method": "offline safe-map pseudo-teacher to fast P1 residual distillation",
        "source_case_count": len(ids), "train_count": len(available_train),
        "fit_one_case": fit_one_case,
        "trunk_from_onehead": (None if trunk_from_onehead is None
                               else str(trunk_from_onehead)),
        "context": context,
        "residual_feature_geometry": residual_feature_geometry,
        "match_channels": match_channels,
        "evidence_mode": evidence_mode,
        "teacher_suffix": teacher_suffix,
        "validation_count": len(available_val), "steps": steps,
        "batch_size": batch_size, "map_weight": map_weight,
        "image_weight": image_weight, "match_weight": match_weight,
        "strain_weight": strain_weight,
        "precompute_seconds": precompute_seconds,
        "training_seconds": train_seconds,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "validation_initial": before, "validation_final": after,
        "trace": trace, "anatomical_labels_used": False,
        "DHR_full_fields_used": False,
        "offline_teacher_optimization_not_in_inference": True,
        "kernel_map_is_inference_input": kernel_maps is not None,
        "frozen_onehead_map_is_inference_input": frozen_maps is not None,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "selection", "matches", "teachers",
                 "base_checkpoint", "initial_checkpoint", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kernel-maps", type=Path)
    parser.add_argument("--frozen-maps", type=Path)
    parser.add_argument("--map-weight", type=float, default=1000.)
    parser.add_argument("--image-weight", type=float, default=.1)
    parser.add_argument("--match-weight", type=float, default=.01)
    parser.add_argument("--strain-weight", type=float, default=.02)
    parser.add_argument("--fit-one-case", type=int)
    parser.add_argument("--trunk-from-onehead", type=Path)
    parser.add_argument("--context", choices=("local", "dilated", "unet",
                                              "multilevel_unet"), default="local")
    parser.add_argument("--match-channels", type=int, choices=(0, 6, 12), default=6)
    parser.add_argument("--evidence-mode",
                        choices=("gaussian", "gaussian_force", "gaussian_zeros",
                                 "force_only", "force_recurrent",
                                 "gaussian_force_recurrent"),
                        default="gaussian")
    parser.add_argument("--teacher-suffix", default="matchopt4_safe257")
    args = parser.parse_args()
    if min(args.map_weight, args.image_weight, args.match_weight,
           args.strain_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    if sum((args.map_weight, args.image_weight,
            args.match_weight, args.strain_weight)) == 0:
        raise ValueError("at least one loss weight must be positive")
    result = train(
        args.root, args.selection, args.matches, args.teachers,
        args.base_checkpoint, args.initial_checkpoint, args.output,
        steps=args.steps, batch_size=args.batch, device_name=args.device,
        kernel_maps=args.kernel_maps, frozen_maps=args.frozen_maps,
        map_weight=args.map_weight, image_weight=args.image_weight,
        match_weight=args.match_weight, strain_weight=args.strain_weight,
        fit_one_case=args.fit_one_case,
        trunk_from_onehead=args.trunk_from_onehead,
        context=args.context, match_channels=args.match_channels,
        evidence_mode=args.evidence_mode,
        teacher_suffix=args.teacher_suffix)
    print(json.dumps({key: result[key] for key in (
        "train_count", "validation_count", "training_seconds",
        "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
