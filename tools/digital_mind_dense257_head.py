"""A genuine 257² image-conditioned safe F1 head over a frozen global base.

Unlike a 65² map merely Q1-refined to 257², every strict interior 257²
vertex receives its own spatially predicted two-component latent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import AdaptiveSoftRadialQ1Relaxation
from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_amortized_network import (
    MindSafeImageNetwork, correlation_volume, image_features, structural_loss,
)
from tools.digital_mind_objective_probe import self_similarity
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_q1_dhr_distill import identity_vertices


def dense_features(base_feature: torch.Tensor, fixed: torch.Tensor,
                   prewarped: torch.Tensor, coarse: torch.Tensor, *,
                   feature_geometry: str = "pixel_center"
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if base_feature.ndim != 4 or base_feature.shape[1:] != (43, 128, 128):
        raise ValueError("base feature must be Bx43x128x128")
    if fixed.shape != prewarped.shape or fixed.shape[1:] != (1, 512, 512):
        raise ValueError("matching Bx1x512x512 images required")
    if coarse.shape != (fixed.shape[0], 257, 257, 2):
        raise ValueError("coarse map must be Bx257x257x2")
    fixed_small = F.interpolate(fixed, size=(256, 256), mode="area")
    moving_small = F.interpolate(prewarped, size=(256, 256), mode="area")
    fdesc, fscale = self_similarity(fixed_small)
    mdesc, _ = self_similarity(moving_small)
    cost = correlation_volume(fdesc, mdesc)
    coarse_bchw = coarse.permute(0, 3, 1, 2)
    axis = (torch.arange(256, device=fixed.device, dtype=fixed.dtype) + .5) / 256
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x, y), dim=0)[None].expand(fixed.shape[0], -1, -1, -1)
    if feature_geometry == "pixel_center":
        # P1 value at each 257-grid cell center, which lies on the fixed
        # top-left-to-bottom-right diagonal used by the image objective.
        displacement = .5 * (coarse[:, :-1, :-1] + coarse[:, 1:, 1:])
        displacement = displacement.permute(0, 3, 1, 2)
    elif feature_geometry == "legacy_align_corners":
        # Preserve the old checkpoint's feature semantics for reproducibility.
        displacement = F.interpolate(coarse_bchw, size=(256, 256),
                                     mode="bilinear", align_corners=True)
    else:
        raise ValueError("unknown feature geometry")
    displacement = displacement - coords
    feature = torch.cat((
        F.interpolate(base_feature, size=(256, 256), mode="bilinear",
                      align_corners=False),
        fdesc, mdesc, cost, displacement, coords,
    ), dim=1)
    assert feature.shape[1] == 88
    mask = ((fixed_small > .04) & (fscale > 1e-4)).to(fixed.dtype)
    return feature, fdesc, mdesc, mask


class DenseSafeHead(nn.Module):
    def __init__(self, *, width: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(88, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, 2, 1),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        self.update = AdaptiveSoftRadialQ1Relaxation(
            257, raw_span=8., safety_fraction=.75, minimum_jacobian=.05)

    def forward(self, coarse: torch.Tensor,
                fine_feature: torch.Tensor) -> torch.Tensor:
        if coarse.ndim != 4 or coarse.shape[1:] != (257, 257, 2):
            raise ValueError("coarse map must be Bx257x257x2")
        if fine_feature.shape != (coarse.shape[0], 88, 256, 256):
            raise ValueError("fine feature must be Bx88x256x256")
        proposal = F.interpolate(self.encoder(fine_feature), size=(257, 257),
                                 mode="bilinear", align_corners=False)
        logits = proposal[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
        return self.update(coarse, logits)


def load_global_base(checkpoint: Path, device: torch.device) -> MindSafeImageNetwork:
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("architecture") != "global":
        raise ValueError("a global-context base checkpoint is required")
    model = MindSafeImageNetwork(architecture="global").to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def train(root: Path, selection: Path, base_checkpoint: Path,
          checkpoint: Path, *, steps: int, batch_size: int,
          device_name: str, feature_geometry: str = "pixel_center",
          affine_source: str = "DHR") -> dict:
    if checkpoint.exists() or checkpoint.with_suffix(".json").exists():
        raise FileExistsError(checkpoint)
    if steps < 1 or batch_size < 1:
        raise ValueError("positive settings required")
    device = torch.device(device_name)
    torch.manual_seed(20260930)
    rng = np.random.default_rng(20261001)
    ids = json.loads(selection.read_text(encoding="utf-8"))["combined_train_ids"]
    from tools.digital_mind_amortized_network import split_ids
    train_ids, val_ids = split_ids(ids)
    missing_affine_ids = []
    if affine_source == "image_only":
        missing_affine_ids = [case for case in ids
                              if not (root / f"{case}_directSG_affine.npz").exists()]
        available = set(ids) - set(missing_affine_ids)
        train_ids = [case for case in train_ids if case in available]
        val_ids = [case for case in val_ids if case in available]
        if not train_ids or not val_ids:
            raise ValueError("image-only initializer leaves empty train/validation split")
    used_ids = train_ids + val_ids
    base = load_global_base(base_checkpoint, device)
    cached = {}
    precompute_start = time.perf_counter()
    with torch.no_grad():
        for case_id in used_ids:
            item = load_case_inputs(root, case_id, device,
                                    affine_source=affine_source)
            base_input, _, _, _ = image_features(
                F.interpolate(item["fixed"], size=(128, 128), mode="area"),
                F.interpolate(item["prewarped"], size=(128, 128), mode="area"),
            )
            coarse = base(base_input, final_side=257)
            cached[case_id] = tuple(x.detach() for x in (
                coarse, *dense_features(base_input, item["fixed"],
                                        item["prewarped"], coarse,
                                        feature_geometry=feature_geometry)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    precompute_seconds = time.perf_counter() - precompute_start
    head = DenseSafeHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=1e-4)
    trace = []
    train_start = time.perf_counter()
    for step in range(steps):
        selected = rng.choice(train_ids, size=batch_size, replace=True)
        coarse, feature, fdesc, mdesc, mask = (
            torch.cat([cached[int(case_id)][index] for case_id in selected], dim=0)
            for index in range(5)
        )
        optimizer.zero_grad(set_to_none=True)
        mapped = head(coarse, feature)
        image = structural_loss(fdesc, mdesc, mask, mapped)
        strain = strain_penalty(mapped)
        loss = image + strain
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in head.parameters()):
            raise FloatingPointError(f"invalid dense head VJP at {step}")
        nn.utils.clip_grad_norm_(head.parameters(), 2.)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "image": float(image.detach()),
                          "strain": float(strain.detach()),
                          "total": float(loss.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_start
    head.eval()
    validation = []
    with torch.no_grad():
        for case_id in val_ids:
            coarse, feature, fdesc, mdesc, mask = cached[case_id]
            dense = head(coarse, feature)
            identity = identity_vertices(257, device=device)
            validation.append({
                "case": case_id,
                "affine_image": float(structural_loss(fdesc, mdesc, mask, identity)),
                "base_image": float(structural_loss(fdesc, mdesc, mask, coarse)),
                "dense_image": float(structural_loss(fdesc, mdesc, mask, dense)),
                "base_total": float(structural_loss(fdesc, mdesc, mask, coarse)
                                    + strain_penalty(coarse)),
                "dense_total": float(structural_loss(fdesc, mdesc, mask, dense)
                                     + strain_penalty(dense)),
            })
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": head.state_dict(),
                "base_checkpoint": str(base_checkpoint),
                "train_ids": train_ids, "validation_ids": val_ids,
                "steps": steps, "feature_geometry": feature_geometry,
                "affine_source": affine_source}, checkpoint)
    report = {"method": "frozen global 65² base + learned independent 257² F1 head",
              "base_checkpoint": str(base_checkpoint),
              "checkpoint": str(checkpoint),
              "feature_geometry": feature_geometry,
              "affine_source": affine_source,
              "case_count": len(used_ids), "source_case_count": len(ids),
              "missing_affine_ids": missing_affine_ids,
              "train_ids": train_ids,
              "validation_ids": val_ids,
              "steps": steps, "batch_size": batch_size,
              "learning_rate": .001,
              "precompute_seconds": precompute_seconds,
              "training_seconds": train_seconds,
              "peak_torch_cuda_allocated_bytes": (
                  None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
              "trace": trace, "validation": validation,
              "landmarks_full_DHR_displacement_machine_matches_loaded": False,
              "DHR_derived_initial_affine_loaded": affine_source == "DHR"}
    checkpoint.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "selection", "base_checkpoint", "checkpoint"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-geometry", default="pixel_center",
                        choices=("pixel_center", "legacy_align_corners"))
    parser.add_argument("--affine-source", default="DHR",
                        choices=("DHR", "image_only"))
    args = parser.parse_args()
    result = train(args.root, args.selection, args.base_checkpoint,
                   args.checkpoint, steps=args.steps, batch_size=args.batch,
                   device_name=args.device, feature_geometry=args.feature_geometry,
                   affine_source=args.affine_source)
    print(json.dumps({key: result[key] for key in (
        "case_count", "steps", "precompute_seconds", "training_seconds",
        "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
