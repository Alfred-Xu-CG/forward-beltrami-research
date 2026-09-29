"""Train a correspondence-conditioned four-pass safe P1 image layer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_amortized_network import (
    image_features, split_ids, structural_loss,
)
from tools.digital_mind_dense257_head import (
    DenseSafeHead, dense_features, load_global_base,
)
from tools.digital_mind_recurrent257 import RecurrentDenseSafeHead
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_mind_sparse_match_finetune import (
    p1_at_points, robust_match_loss,
)


def gaussian_match_features(source: torch.Tensor, target: torch.Tensor,
                            *, side: int = 256,
                            sigmas: tuple[float, float] = (.04, .12)
                            ) -> torch.Tensor:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or (
        len(source) < 1 or not bool(torch.isfinite(source).all()) or
        not bool(torch.isfinite(target).all())
    ):
        raise ValueError("matching nonempty finite Nx2 points required")
    axis = (torch.arange(side, device=source.device, dtype=source.dtype) + .5) / side
    motion = target - source
    fields = []
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError("positive Gaussian width required")
        xw = torch.exp(-((axis[:, None] - source[None, :, 0]).square()) /
                       (2 * sigma * sigma))
        yw = torch.exp(-((axis[:, None] - source[None, :, 1]).square()) /
                       (2 * sigma * sigma))
        mass = yw @ xw.T
        xnum = (yw * motion[None, :, 0]) @ xw.T
        ynum = (yw * motion[None, :, 1]) @ xw.T
        fields.extend((512 * xnum / (mass + 1e-8),
                       512 * ynum / (mass + 1e-8),
                       torch.log1p(mass)))
    return torch.stack(fields, dim=0)[None]


def split_matches(source: torch.Tensor, target: torch.Tensor, case: int):
    order = np.random.default_rng(290929 + int(case)).permutation(len(source))
    cut = max(1, int(.8 * len(order)))
    input_idx = torch.as_tensor(order[:cut].copy(), device=source.device)
    held_idx = torch.as_tensor(order[cut:].copy(), device=source.device)
    return source[input_idx], target[input_idx], source[held_idx], target[held_idx]


def point_mean_px(mapped: torch.Tensor, source: torch.Tensor,
                  target: torch.Tensor) -> float:
    difference = 512 * (p1_at_points(mapped, source) - target)
    return float(torch.sqrt(difference.square().sum(-1)).mean())


def train(root: Path, selection: Path, matches: Path,
          base_checkpoint: Path, one_pass_checkpoint: Path, output: Path, *,
          steps: int, batch_size: int, device_name: str) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    ids = json.loads(selection.read_text(encoding="utf-8"))["combined_train_ids"]
    train_ids, val_ids = split_ids(ids)
    device = torch.device(device_name)
    torch.manual_seed(20261003)
    rng = np.random.default_rng(20261003)
    base = load_global_base(base_checkpoint, device)
    old_saved = torch.load(one_pass_checkpoint, map_location=device, weights_only=False)
    one = DenseSafeHead().to(device)
    one.load_state_dict(old_saved["head_state_dict"])
    one.eval()
    model = RecurrentDenseSafeHead(passes=4, match_channels=6).to(device)
    model.initialize_from_one_pass(one)
    cache = {}
    started = time.perf_counter()
    with torch.no_grad():
        for case in ids:
            path = matches / f"{case}_alignedSG_matches.npz"
            if not path.exists() or not (root / f"{case}_directSG_affine.npz").exists():
                continue
            with np.load(path) as data:
                source = torch.from_numpy(data["source_fixed_unit"].copy()).to(device)
                target = torch.from_numpy(data["target_aligned_unit"].copy()).to(device)
            if len(source) < 16:
                continue
            source_in, target_in, source_held, target_held = split_matches(
                source, target, case)
            item = load_case_inputs(root, case, device, affine_source="image_only")
            feature, _, _, _ = image_features(
                F.interpolate(item["fixed"], size=(128, 128), mode="area"),
                F.interpolate(item["prewarped"], size=(128, 128), mode="area"))
            coarse = base(feature, final_side=257)
            fine, fdesc, mdesc, mask = dense_features(
                feature, item["fixed"], item["prewarped"], coarse,
                feature_geometry="legacy_align_corners")
            raster = gaussian_match_features(source_in, target_in)
            cache[case] = tuple(x.detach() for x in (
                coarse, fine, fdesc, mdesc, mask, raster,
                source_in, target_in, source_held, target_held))
    available_train = [case for case in train_ids if case in cache]
    available_val = [case for case in val_ids if case in cache]
    if not available_train or not available_val:
        raise ValueError("empty train or validation match cases")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    precompute_seconds = time.perf_counter() - started

    def validation() -> list[dict]:
        model.eval()
        rows = []
        with torch.no_grad():
            for case in available_val:
                coarse, fine, fdesc, mdesc, mask, raster, si, ti, sh, th = cache[case]
                old = one(coarse, fine)
                mapped = model(coarse, fine, match_feature=raster)
                rows.append({
                    "case": case, "input_match_count": len(si),
                    "held_match_count": len(sh),
                    "old_image": float(structural_loss(fdesc, mdesc, mask, old)),
                    "new_image": float(structural_loss(fdesc, mdesc, mask, mapped)),
                    "old_strain": float(strain_penalty(old)),
                    "new_strain": float(strain_penalty(mapped)),
                    "old_held_match_mean_px": point_mean_px(old, sh, th),
                    "new_held_match_mean_px": point_mean_px(mapped, sh, th),
                    "old_input_match_mean_px": point_mean_px(old, si, ti),
                    "new_input_match_mean_px": point_mean_px(mapped, si, ti),
                })
        return rows

    before = validation()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-4)
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        chosen = rng.choice(available_train, size=batch_size, replace=True)
        optimizer.zero_grad(set_to_none=True)
        image_losses, match_losses, strain_losses = [], [], []
        for case in chosen:
            coarse, fine, fdesc, mdesc, mask, raster, si, ti, _, _ = cache[int(case)]
            mapped = model(coarse, fine, match_feature=raster)
            image_losses.append(structural_loss(fdesc, mdesc, mask, mapped))
            match_losses.append(robust_match_loss(p1_at_points(mapped, si), ti))
            strain_losses.append(strain_penalty(mapped))
        image = torch.stack(image_losses).mean()
        match = torch.stack(match_losses).mean()
        strain = torch.stack(strain_losses).mean()
        loss = image + .1 * strain + .02 * match
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
            raise FloatingPointError(f"invalid match-conditioned VJP at {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "image": float(image.detach()),
                          "match_robust_px": float(match.detach()),
                          "strain": float(strain.detach()), "total": float(loss.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    after = validation()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "passes": 4,
                "match_channels": 6, "base_checkpoint": str(base_checkpoint),
                "one_pass_checkpoint": str(one_pass_checkpoint),
                "train_ids": available_train, "validation_ids": available_val,
                "steps": steps, "strain_weight": .1, "match_weight": .02,
                "match_input_fraction": .8}, output)
    report = {"method": "Gaussian-match-conditioned four-pass safe257 CNN",
              "case_count": len(ids), "train_count": len(available_train),
              "validation_count": len(available_val), "steps": steps,
              "batch_size": batch_size, "strain_weight": .1,
              "match_weight": .02, "match_input_fraction": .8,
              "sigma_unit": [.04, .12],
              "precompute_seconds": precompute_seconds,
              "training_seconds": training_seconds,
              "peak_torch_cuda_allocated_bytes": (
                  None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
              "validation_initial": before, "validation_final": after,
              "trace": trace, "anatomical_labels_used": False,
              "held_match_not_independent_of_RANSAC": True}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "selection", "matches", "base_checkpoint",
                 "one_pass_checkpoint", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = train(args.root, args.selection, args.matches,
                   args.base_checkpoint, args.one_pass_checkpoint, args.output,
                   steps=args.steps, batch_size=args.batch, device_name=args.device)
    print(json.dumps({key: result[key] for key in (
        "train_count", "validation_count", "precompute_seconds",
        "training_seconds", "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
