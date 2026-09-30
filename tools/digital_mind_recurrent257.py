"""Four exact sequential image-conditioned safe-F1 updates on 257² vertices.

Each pass updates the same indexed P1 vertex table using current geometric
edges. The output remains P1 on the original fixed triangulation; no warped
map is resampled from an independently triangulated domain.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import AdaptiveSoftRadialQ1Relaxation
from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_amortized_network import (
    image_features, split_ids, structural_loss,
)
from tools.digital_mind_dense257_head import (
    DenseSafeHead, dense_features, load_global_base,
)
from tools.digital_mind_safe_optimize import strain_penalty


def p1_cell_centers(vertices: torch.Tensor) -> torch.Tensor:
    if vertices.ndim != 4 or vertices.shape[1:] != (257, 257, 2):
        raise ValueError("Bx257x257x2 map required")
    return .5 * (vertices[:, :-1, :-1] + vertices[:, 1:, 1:])


class GlobalMatchTrunk(nn.Module):
    """Four-scale context propagation with a local stem and symmetric skips."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.down = nn.ModuleList(nn.Sequential(
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        ) for _ in range(4))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.up = nn.ModuleList(nn.Sequential(
            nn.Conv2d(2 * width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        ) for _ in range(4))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        current = self.stem(features)
        skips = [current]
        for stage in self.down:
            current = stage(current)
            skips.append(current)
        current = self.bottleneck(current)
        for stage, skip in zip(self.up, reversed(skips[:-1]), strict=True):
            current = F.interpolate(current, size=skip.shape[-2:],
                                    mode="bilinear", align_corners=False)
            current = stage(torch.cat((current, skip), dim=1))
        return current


class RecurrentDenseSafeHead(nn.Module):
    def __init__(self, *, width: int = 32, passes: int = 4,
                 match_channels: int = 0,
                 context: str = "local") -> None:
        super().__init__()
        if passes < 1:
            raise ValueError("at least one pass required")
        self.passes = passes
        self.match_channels = match_channels
        if context not in ("local", "dilated", "unet", "multilevel_unet"):
            raise ValueError("unknown residual context")
        self.context = context
        if context in ("unet", "multilevel_unet"):
            self.trunk = GlobalMatchTrunk(90 + match_channels, width)
        else:
            dilations = (1, 1, 1) if context == "local" else (1, 2, 4, 8, 16, 32)
            layers = []
            for index, dilation in enumerate(dilations):
                layers.extend((
                    nn.Conv2d(90 + match_channels if index == 0 else width,
                              width, 3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                ))
            self.trunk = nn.Sequential(*layers)
        self.levels = (17, 33, 65) if context == "multilevel_unet" else ()
        head_count = len(self.levels) if self.levels else passes
        self.heads = nn.ModuleList(nn.Conv2d(width, 2, 1)
                                   for _ in range(head_count))
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.update = AdaptiveSoftRadialQ1Relaxation(
            257, raw_span=8., safety_fraction=.75, minimum_jacobian=.05)

    def initialize_from_one_pass(self, source: DenseSafeHead) -> None:
        stem = (self.trunk.stem if self.context in ("unet", "multilevel_unet")
                else self.trunk)
        if len(self.heads) < 1 or stem[0].out_channels != source.encoder[0].out_channels:
            raise ValueError("incompatible source head")
        with torch.no_grad():
            stem[0].weight[:, :88].copy_(source.encoder[0].weight)
            stem[0].weight[:, 88:].zero_()
            stem[0].bias.copy_(source.encoder[0].bias)
            for index in (2, 4):
                stem[index].weight.copy_(source.encoder[index].weight)
                stem[index].bias.copy_(source.encoder[index].bias)
            self.heads[0].weight.copy_(source.encoder[-1].weight)
            self.heads[0].bias.copy_(source.encoder[-1].bias)

    def forward(self, coarse: torch.Tensor,
                fine_feature: torch.Tensor, *,
                match_feature: torch.Tensor | None = None,
                evidence_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
                return_passes: bool = False) -> torch.Tensor | list[torch.Tensor]:
        if coarse.ndim != 4 or coarse.shape[1:] != (257, 257, 2) or (
            fine_feature.shape != (coarse.shape[0], 88, 256, 256)
        ):
            raise ValueError("matching coarse map and Bx88x256x256 features required")
        if evidence_fn is not None and (match_feature is not None or
                                        self.context == "multilevel_unet" or
                                        not self.match_channels):
            raise ValueError("dynamic evidence requires non-multilevel evidence head")
        if self.match_channels and evidence_fn is None:
            if match_feature is None or match_feature.shape != (
                coarse.shape[0], self.match_channels, 256, 256
            ):
                raise ValueError("matching rasterized match features required")
        elif not self.match_channels and match_feature is not None:
            raise ValueError("this model has no match input")
        if self.context == "multilevel_unet":
            feedback = torch.zeros_like(p1_cell_centers(coarse)).permute(0, 3, 1, 2)
            channels = ((fine_feature, feedback) if match_feature is None else
                        (fine_feature, match_feature, feedback))
            hidden = self.trunk(torch.cat(channels, dim=1))
            residual = torch.zeros_like(coarse).permute(0, 3, 1, 2)
            for side, head in zip(self.levels, self.heads, strict=True):
                control = .04 * torch.tanh(head(F.adaptive_avg_pool2d(
                    hidden, (side - 2, side - 2))))
                residual = residual + F.interpolate(
                    F.pad(control, (1, 1, 1, 1)), size=(257, 257),
                    mode="bilinear", align_corners=True)
            goal = coarse + residual.permute(0, 2, 3, 1)
            current = coarse
            outputs = []
            for remaining in range(self.passes, 0, -1):
                horizontal = (current[:, 1:-1, 2:] - current[:, 1:-1, :-2]) / 2
                vertical = (current[:, 2:, 1:-1] - current[:, :-2, 1:-1]) / 2
                desired = (goal[:, 1:-1, 1:-1] - current[:, 1:-1, 1:-1]) / remaining
                det = (horizontal[..., 0] * vertical[..., 1]
                       - horizontal[..., 1] * vertical[..., 0])
                valid = det.abs() > 1e-10
                scale = torch.where(valid, 8 * det, torch.ones_like(det))
                u = (desired[..., 0] * vertical[..., 1]
                     - desired[..., 1] * vertical[..., 0]) / scale
                v = (horizontal[..., 0] * desired[..., 1]
                     - horizontal[..., 1] * desired[..., 0]) / scale
                logits = torch.atanh(torch.stack((u, v), dim=-1).clamp(-.98, .98))
                logits = torch.where(valid[..., None], logits,
                                     torch.zeros_like(logits))
                current = self.update(current, logits)
                if return_passes:
                    outputs.append(current)
            return outputs if return_passes else current
        current = coarse
        base_centers = p1_cell_centers(coarse).permute(0, 3, 1, 2)
        outputs = []
        for head in self.heads:
            evidence = evidence_fn(current) if evidence_fn is not None else match_feature
            if evidence_fn is not None and evidence.shape != (
                coarse.shape[0], self.match_channels, 256, 256
            ):
                raise ValueError("dynamic evidence has wrong shape")
            feedback = p1_cell_centers(current).permute(0, 3, 1, 2) - base_centers
            channels = ((fine_feature, feedback) if evidence is None else
                        (fine_feature, evidence, feedback))
            hidden = self.trunk(torch.cat(channels, dim=1))
            proposed = F.interpolate(head(hidden), size=(257, 257),
                                     mode="bilinear", align_corners=False)
            logits = proposed[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            current = self.update(current, logits)
            if return_passes:
                outputs.append(current)
        return outputs if return_passes else current


def train(root: Path, selection: Path, base_checkpoint: Path,
          one_pass_checkpoint: Path, output: Path, *, steps: int,
          batch_size: int, device_name: str, passes: int = 4,
          strain_weight: float = 1.) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if steps < 1 or batch_size < 1 or strain_weight < 0:
        raise ValueError("positive training settings required")
    ids = json.loads(selection.read_text(encoding="utf-8"))["combined_train_ids"]
    train_ids, val_ids = split_ids(ids)
    device = torch.device(device_name)
    torch.manual_seed(20260930)
    rng = np.random.default_rng(20261002)
    base = load_global_base(base_checkpoint, device)
    saved_one = torch.load(one_pass_checkpoint, map_location=device, weights_only=False)
    if saved_one.get("feature_geometry", "legacy_align_corners") != "legacy_align_corners":
        raise ValueError("this warm start requires original feature geometry")
    one_pass = DenseSafeHead().to(device)
    one_pass.load_state_dict(saved_one["head_state_dict"])
    one_pass.eval()
    model = RecurrentDenseSafeHead(passes=passes).to(device)
    model.initialize_from_one_pass(one_pass)
    cache = {}
    precompute_started = time.perf_counter()
    with torch.no_grad():
        for case in ids:
            for source in ("DHR", "image_only"):
                if source == "image_only" and not (
                    root / f"{case}_directSG_affine.npz").exists():
                    continue
                item = load_case_inputs(root, case, device, affine_source=source)
                feature, _, _, _ = image_features(
                    F.interpolate(item["fixed"], size=(128, 128), mode="area"),
                    F.interpolate(item["prewarped"], size=(128, 128), mode="area"))
                coarse = base(feature, final_side=257)
                fine, fdesc, mdesc, mask = dense_features(
                    feature, item["fixed"], item["prewarped"], coarse,
                    feature_geometry="legacy_align_corners")
                cache[(case, source)] = tuple(t.detach() for t in (
                    coarse, fine, fdesc, mdesc, mask))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    precompute_seconds = time.perf_counter() - precompute_started

    def validate() -> list[dict]:
        model.eval()
        rows = []
        with torch.no_grad():
            for case in val_ids:
                for source in ("DHR", "image_only"):
                    if (case, source) not in cache:
                        continue
                    coarse, fine, fdesc, mdesc, mask = cache[(case, source)]
                    old = one_pass(coarse, fine)
                    new = model(coarse, fine)
                    old_image = structural_loss(fdesc, mdesc, mask, old)
                    new_image = structural_loss(fdesc, mdesc, mask, new)
                    old_strain = strain_penalty(old)
                    new_strain = strain_penalty(new)
                    rows.append({
                        "case": case, "affine_source": source,
                        "one_pass_image": float(old_image),
                        "recurrent_image": float(new_image),
                        "one_pass_strain": float(old_strain),
                        "recurrent_strain": float(new_strain),
                        "one_pass_total": float(old_image + strain_weight * old_strain),
                        "recurrent_total": float(new_image + strain_weight * new_strain),
                    })
        return rows

    before = validate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-4)
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        chosen = rng.choice(train_ids, size=batch_size, replace=True)
        examples = []
        for case in chosen:
            options = ("DHR", "image_only") if (int(case), "image_only") in cache else ("DHR",)
            source = options[int(rng.integers(len(options)))]
            examples.append(cache[(int(case), source)])
        coarse, fine, fdesc, mdesc, mask = (
            torch.cat([example[index] for example in examples], dim=0)
            for index in range(5))
        optimizer.zero_grad(set_to_none=True)
        mapped = model(coarse, fine)
        image = structural_loss(fdesc, mdesc, mask, mapped)
        strain = strain_penalty(mapped)
        total = image + strain_weight * strain
        total.backward()
        if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                   for parameter in model.parameters() if parameter.requires_grad):
            raise FloatingPointError(f"invalid recurrent parameter VJP at {step}")
        nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "image": float(image.detach()),
                          "strain": float(strain.detach()), "total": float(total.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    after = validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "passes": passes,
                "strain_weight": strain_weight,
                "base_checkpoint": str(base_checkpoint),
                "one_pass_checkpoint": str(one_pass_checkpoint),
                "train_ids": train_ids, "validation_ids": val_ids,
                "steps": steps, "image_only_augmentation": True}, output)
    report = {
        "method": "four-pass image-conditioned current-edge safe257 recurrent head",
        "passes": passes, "steps": steps, "batch_size": batch_size,
        "strain_weight": strain_weight,
        "case_count": len(ids), "train_count": len(train_ids),
        "validation_count": len(val_ids),
        "available_image_only_views": sum((case, "image_only") in cache for case in ids),
        "precompute_seconds": precompute_seconds,
        "training_seconds": training_seconds,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "validation_initial": before, "validation_final": after,
        "trace": trace, "anatomical_landmarks_used": False,
        "DHR_and_image_only_affine_training_views_but_no_DHR_full_field": True,
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
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--strain-weight", type=float, default=1.)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = train(args.root, args.selection, args.base_checkpoint,
                   args.one_pass_checkpoint, args.output, steps=args.steps,
                   batch_size=args.batch, passes=args.passes,
                   strain_weight=args.strain_weight,
                   device_name=args.device)
    print(json.dumps({key: result[key] for key in (
        "passes", "steps", "train_count", "validation_count",
        "available_image_only_views", "precompute_seconds",
        "training_seconds", "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
