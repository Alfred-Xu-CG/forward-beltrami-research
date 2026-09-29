"""Fine-tune a safe 257² head with image-only sparse residual correspondences.

The old global base is frozen; the old dense head is a warm start. Machine
matches and 512-square image descriptors supply losses, never anatomical CSVs.
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

from tools.digital_acrobat_teacher_probe import load_case_inputs
from tools.digital_mind_amortized_network import (
    image_features, split_ids, structural_loss,
)
from tools.digital_mind_dense257_head import (
    DenseSafeHead, dense_features, load_global_base,
)
from tools.digital_mind_safe_optimize import strain_penalty


def p1_at_points(vertices: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Differentiable vertex-value interpolation at fixed normalized queries."""
    if vertices.ndim != 4 or vertices.shape[0] != 1 or vertices.shape[-1] != 2:
        raise ValueError("one BxNxNx2 vertex map required")
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("Kx2 fixed query required")
    cells = vertices.shape[1] - 1
    scaled = query * cells
    ij = torch.floor(scaled).long().clamp(0, cells - 1)
    x, y = ij[:, 0], ij[:, 1]
    s, t = scaled[:, 0] - x, scaled[:, 1] - y
    a, b = vertices[0, y, x], vertices[0, y, x + 1]
    c, d = vertices[0, y + 1, x + 1], vertices[0, y + 1, x]
    lower = (1 - s)[:, None] * a + (s - t)[:, None] * b + t[:, None] * c
    upper = (1 - t)[:, None] * a + s[:, None] * c + (t - s)[:, None] * d
    return torch.where((t <= s)[:, None], lower, upper)


def robust_match_loss(predicted: torch.Tensor, target: torch.Tensor,
                      *, side: int = 512, radius: float = 4.) -> torch.Tensor:
    delta_px = side * (predicted - target)
    square = (delta_px * delta_px).sum(dim=-1)
    return (torch.sqrt(square + radius * radius) - radius).mean()


def train(root: Path, selection: Path, match_dir: Path,
          base_checkpoint: Path, initial_head: Path, output: Path, *,
          steps: int, batch_size: int, match_weight: float,
          device_name: str) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if steps < 1 or batch_size < 1 or match_weight < 0:
        raise ValueError("invalid train settings")
    ids = json.loads(selection.read_text(encoding="utf-8"))["combined_train_ids"]
    train_ids, val_ids = split_ids(ids)
    available = {case for case in ids if (root / f"{case}_directSG_affine.npz").exists()}
    train_ids = [case for case in train_ids if case in available]
    val_ids = [case for case in val_ids if case in available]
    device = torch.device(device_name)
    torch.manual_seed(20260930)
    rng = np.random.default_rng(20261001)
    base = load_global_base(base_checkpoint, device)
    saved_head = torch.load(initial_head, map_location=device, weights_only=False)
    geometry = saved_head.get("feature_geometry", "legacy_align_corners")
    head = DenseSafeHead().to(device)
    head.load_state_dict(saved_head["head_state_dict"])
    cached = {}
    with torch.no_grad():
        for case in train_ids + val_ids:
            item = load_case_inputs(root, case, device, affine_source="image_only")
            feature, _, _, _ = image_features(
                F.interpolate(item["fixed"], size=(128, 128), mode="area"),
                F.interpolate(item["prewarped"], size=(128, 128), mode="area"))
            coarse = base(feature, final_side=257)
            fine, fdesc, mdesc, mask = dense_features(
                feature, item["fixed"], item["prewarped"], coarse,
                feature_geometry=geometry)
            matches_path = match_dir / f"{case}_alignedSG_matches.npz"
            if matches_path.exists():
                with np.load(matches_path) as points:
                    source = torch.from_numpy(points["source_fixed_unit"].copy()).to(device)
                    target = torch.from_numpy(points["target_aligned_unit"].copy()).to(device)
            else:
                source = target = None
            cached[case] = tuple(v.detach() for v in (coarse, fine, fdesc, mdesc, mask)) + (
                source, target)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def validation_rows() -> list[dict]:
        rows = []
        head.eval()
        with torch.no_grad():
            for case in val_ids:
                coarse, fine, fdesc, mdesc, mask, source, target = cached[case]
                mapped = head(coarse, fine)
                row = {"case": case, "match_count": 0 if source is None else len(source),
                       "image": float(structural_loss(fdesc, mdesc, mask, mapped)),
                       "strain": float(strain_penalty(mapped))}
                if source is not None:
                    difference = 512 * (p1_at_points(mapped, source) - target)
                    row["match_robust_px"] = float(robust_match_loss(
                        p1_at_points(mapped, source), target))
                    row["match_mean_euclidean_px"] = float(torch.sqrt(
                        difference.square().sum(-1)).mean())
                rows.append(row)
        return rows

    baseline_rows = validation_rows()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0005, weight_decay=1e-4)
    trace = []
    started = time.perf_counter()
    head.train()
    for step in range(steps):
        chosen = rng.choice(train_ids, size=batch_size, replace=True)
        optimizer.zero_grad(set_to_none=True)
        batch_losses = []
        match_losses = []
        for case in chosen:
            coarse, fine, fdesc, mdesc, mask, source, target = cached[int(case)]
            mapped = head(coarse, fine)
            image = structural_loss(fdesc, mdesc, mask, mapped)
            strain = strain_penalty(mapped)
            match = (robust_match_loss(p1_at_points(mapped, source), target)
                     if source is not None else mapped.sum() * 0.)
            batch_losses.append(image + strain + match_weight * match)
            match_losses.append(match.detach())
        loss = torch.stack(batch_losses).mean()
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in head.parameters()):
            raise FloatingPointError(f"nonfinite VJP at step {step}")
        nn.utils.clip_grad_norm_(head.parameters(), 2.)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "total": float(loss.detach()),
                          "mean_match_loss_px": float(torch.stack(match_losses).mean())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    rows = validation_rows()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": head.state_dict(),
                "base_checkpoint": str(base_checkpoint),
                "initial_head": str(initial_head),
                "feature_geometry": geometry, "affine_source": "image_only",
                "match_weight": match_weight, "train_ids": train_ids,
                "validation_ids": val_ids, "steps": steps}, output)
    report = {"method": "frozen global base + safe257 head fine-tuned on image and aligned SG matches",
              "match_weight": match_weight, "steps": steps,
              "batch_size": batch_size, "training_seconds": train_seconds,
              "feature_geometry": geometry,
              "train_count": len(train_ids), "validation_count": len(val_ids),
              "train_match_count": sum(cached[c][5] is not None for c in train_ids),
              "validation_match_count": sum(cached[c][5] is not None for c in val_ids),
              "peak_torch_cuda_allocated_bytes": (
                  None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
              "trace": trace, "validation_before": baseline_rows,
              "validation": rows,
              "anatomical_landmarks_used": False}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "selection", "match_dir", "base_checkpoint",
                 "initial_head", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--match-weight", type=float, default=.02)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = train(args.root, args.selection, args.match_dir, args.base_checkpoint,
                   args.initial_head, args.output, steps=args.steps,
                   batch_size=args.batch, match_weight=args.match_weight,
                   device_name=args.device)
    print(json.dumps({"train": report["train_count"],
                      "validation": report["validation_count"],
                      "train_matches": report["train_match_count"],
                      "val_matches": report["validation_match_count"],
                      "seconds": report["training_seconds"]}))


if __name__ == "__main__":
    main()
