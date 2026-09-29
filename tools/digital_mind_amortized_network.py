"""Train an image-only structural-feature CNN to emit safe Q1/P1 latents.

This does not use DHR full deformation teachers, machine matches, or
anatomical landmarks. It does use a saved DHR-derived initial affine to put
the moving image in the fixed residual frame. The output is a 257-square
vertex table with effective nonrigid degrees of freedom through 65-square.
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

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveSoftRadialQ1Relaxation, q1_dyadic_refine,
)
from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_objective_probe import self_similarity, sampled_p1_descriptor
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_q1_dhr_distill import identity_vertices


SHIFTS = (-3, -1, 0, 1, 3)


def correlation_volume(fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 8:
        raise ValueError("matching Bx8xHxW descriptor arrays required")
    height, width = fixed.shape[-2:]
    padded = F.pad(moving, (3, 3, 3, 3), mode="replicate")
    costs = []
    for dy in SHIFTS:
        for dx in SHIFTS:
            shifted = padded[:, :, 3 + dy:3 + dy + height,
                             3 + dx:3 + dx + width]
            costs.append((fixed - shifted).abs().mean(dim=1, keepdim=True))
    return torch.cat(costs, dim=1)


def image_features(fixed: torch.Tensor, moving: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1:] != (1, 128, 128):
        raise ValueError("matching Bx1x128x128 image arrays required")
    fixed_descriptor, fixed_scale = self_similarity(fixed)
    moving_descriptor, _ = self_similarity(moving)
    volume = correlation_volume(fixed_descriptor, moving_descriptor)
    axis = (torch.arange(128, device=fixed.device, dtype=fixed.dtype) + .5) / 128
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x, y), dim=0)[None].expand(fixed.shape[0], -1, -1, -1)
    feature = torch.cat((fixed_descriptor, moving_descriptor, volume, coords), dim=1)
    mask = ((fixed > .04) & (fixed_scale > 1e-4)).to(fixed.dtype)
    return feature, fixed_descriptor, moving_descriptor, mask


class MindSafeImageNetwork(nn.Module):
    def __init__(self, *, width: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(43, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.fuse = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.sides = (17, 33, 65)
        self.heads = nn.ModuleList(nn.Conv2d(width, 2, 1) for _ in self.sides)
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.updates = nn.ModuleList(
            AdaptiveSoftRadialQ1Relaxation(side, raw_span=8.,
                                           safety_fraction=.75, minimum_jacobian=.05)
            for side in self.sides
        )

    def forward(self, feature: torch.Tensor, *, final_side: int = 65) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1:] != (43, 128, 128):
            raise ValueError("feature must be Bx43x128x128")
        if final_side not in (65, 257):
            raise ValueError("supported final sides are 65 or 257")
        coarse = self.stem(feature)
        context = self.down(coarse)
        fused = self.fuse(coarse + F.interpolate(context, size=(128, 128),
                                                 mode="bilinear", align_corners=False))
        current = identity_vertices(17, device=feature.device).to(feature.dtype)
        current = current.expand(feature.shape[0], -1, -1, -1)
        for index, (side, head, update) in enumerate(zip(self.sides, self.heads, self.updates)):
            if index:
                current = q1_dyadic_refine(current)
            local = F.interpolate(fused, size=(side, side), mode="bilinear",
                                  align_corners=False)
            logits = head(local)[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            current = update(current, logits)
        if final_side == 257:
            current = q1_dyadic_refine(q1_dyadic_refine(current))
        return current


def structural_loss(fixed_descriptor: torch.Tensor, moving_descriptor: torch.Tensor,
                    mask: torch.Tensor, mapped: torch.Tensor) -> torch.Tensor:
    warped = sampled_p1_descriptor(moving_descriptor, mapped)
    return ((fixed_descriptor - warped).abs() * mask).sum() / (
        mask.sum() * fixed_descriptor.shape[1] + 1e-8)


def split_ids(case_ids: list[int], *, seed: int = 20260929) -> tuple[list[int], list[int]]:
    order = np.random.default_rng(seed).permutation(case_ids)
    cut = int(.8 * len(order))
    return order[:cut].tolist(), order[cut:].tolist()


def train(root: Path, selection: Path, checkpoint: Path, *, steps: int,
          device_name: str, learning_rate: float = .001, batch_size: int = 4) -> dict:
    if checkpoint.exists() or checkpoint.with_suffix(".json").exists():
        raise FileExistsError(checkpoint)
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("positive training settings required")
    case_ids = json.loads(selection.read_text(encoding="utf-8"))["combined_train_ids"]
    train_ids, val_ids = split_ids(case_ids)
    device = torch.device(device_name)
    torch.manual_seed(20260929)
    rng = np.random.default_rng(20260930)
    cases = {}
    precompute_start = time.perf_counter()
    with torch.no_grad():
        for case_id in case_ids:
            case = load_case_inputs(root, case_id, device)
            fixed = F.interpolate(case["fixed"], size=(128, 128), mode="area")
            moving = F.interpolate(case["prewarped"], size=(128, 128), mode="area")
            cases[case_id] = tuple(item.detach() for item in image_features(fixed, moving))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    precompute_seconds = time.perf_counter() - precompute_start
    model = MindSafeImageNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    identity = identity_vertices(65, device=device)
    loss_trace = []
    training_start = time.perf_counter()
    model.train()
    for step in range(steps):
        selected = rng.choice(train_ids, size=batch_size, replace=True)
        feat, fdesc, mdesc, mask = (
            torch.cat([cases[int(case_id)][index] for case_id in selected], dim=0)
            for index in range(4)
        )
        optimizer.zero_grad(set_to_none=True)
        mapped = model(feat)
        image = structural_loss(fdesc, mdesc, mask, mapped)
        regularity = strain_penalty(mapped)
        loss = image + regularity
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.parameters() if p.requires_grad):
            raise FloatingPointError(f"missing/nonfinite network VJP at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            loss_trace.append({"step": step, "image_loss": float(image.detach()),
                               "strain_penalty": float(regularity.detach()),
                               "total_loss": float(loss.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_start
    model.eval()
    validation = []
    with torch.no_grad():
        for case_id in val_ids:
            feat, fdesc, mdesc, mask = cases[case_id]
            mapped = model(feat)
            validation.append({
                "case": case_id,
                "affine_only_image_loss": float(structural_loss(fdesc, mdesc, mask, identity)),
                "predicted_image_loss": float(structural_loss(fdesc, mdesc, mask, mapped)),
                "predicted_strain_penalty": float(strain_penalty(mapped)),
            })
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "train_ids": train_ids,
                "validation_ids": val_ids, "steps": steps}, checkpoint)
    result = {
        "method": "MIND-like 5x5 local cost-volume CNN -> safe 17/33/65 F1 -> 257 Q1/P1",
        "case_count": len(case_ids), "train_case_count": len(train_ids),
        "validation_case_count": len(val_ids), "train_ids": train_ids,
        "validation_ids": val_ids, "steps": steps, "batch_size": batch_size,
        "learning_rate": learning_rate, "precompute_seconds": precompute_seconds,
        "training_seconds": training_seconds,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "loss_trace": loss_trace,
        "validation": validation,
        "landmarks_machine_matches_full_DHR_displacement_loaded": False,
        "DHR_derived_initial_affine_loaded": True,
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = train(args.root, args.selection, args.checkpoint, steps=args.steps,
                   device_name=args.device, learning_rate=args.learning_rate,
                   batch_size=args.batch)
    print(json.dumps({key: result[key] for key in (
        "case_count", "train_case_count", "validation_case_count", "steps",
        "precompute_seconds", "training_seconds", "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
