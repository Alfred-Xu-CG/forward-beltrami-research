"""Diagnostic: fit the same CNN to one unlabeled physical-canvas pair.

Scores on this same pair are *not* generalization evidence. This isolates
network capacity/optimization from ACROBAT-to-BIRL transfer.
"""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_mind_amortized_network import (
    MindSafeImageNetwork, image_features, structural_loss,
)
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail
from tools.digital_affine_prewarp import warp_moving_to_fixed


def fit(fixed_path: Path, moving_path: Path, affine_path: Path,
        output: Path, *, steps: int, device_name: str) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if steps < 1:
        raise ValueError("positive step count required")
    device = torch.device(device_name)
    torch.manual_seed(20260929)
    fixed, _ = _read_gray_thumbnail(fixed_path, 512)
    moving, _ = _read_gray_thumbnail(moving_path, 512)
    fixed, moving = fixed.to(device), moving.to(device)
    with np.load(affine_path) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    if matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError("invalid initial affine")
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine required")
    with torch.no_grad():
        prewarp = warp_moving_to_fixed(
            moving, torch.tensor(matrix, device=device),
            torch.tensor(offset, device=device), height=512, width=512,
        )
        fixed_small = F.interpolate(fixed, size=(128, 128), mode="area")
        moving_small = F.interpolate(prewarp, size=(128, 128), mode="area")
        features, fixed_descriptor, moving_descriptor, mask = image_features(
            fixed_small, moving_small,
        )
    model = MindSafeImageNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    initial = float(structural_loss(
        fixed_descriptor, moving_descriptor, mask,
        identity_vertices(65, device=device)))
    trace = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mapped = model(features)
        image = structural_loss(fixed_descriptor, moving_descriptor, mask, mapped)
        strain = strain_penalty(mapped)
        loss = image + strain
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
            raise FloatingPointError(f"invalid network VJP at step {step}")
        nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "image_loss": float(image.detach()),
                          "strain": float(strain.detach()),
                          "total": float(loss.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    with torch.no_grad():
        output_map = model(features, final_side=257)
        final_image = float(structural_loss(
            fixed_descriptor, moving_descriptor, mask, output_map))
        validity = validate_q1_map(output_map, identity_vertices(257, device=device))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=output_map.cpu().numpy(),
                        boundary_reference=identity_vertices(257, device=device).cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not binary["valid"] or not validity["valid"]:
        raise RuntimeError("saved map did not certify")
    report = {"mode": "single-pair same-image CNN-capacity diagnostic, not generalization",
              "fixed": str(fixed_path), "moving": str(moving_path),
              "affine": str(affine_path), "map": str(output),
              "steps": steps, "initial_image_loss": initial,
              "last_exported_P1_image_loss": final_image,
              "training_seconds": train_seconds,
              "peak_torch_cuda_allocated_bytes": (
                  None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
              "trace": trace, "saved_binary_certificate": binary,
              "in_memory_validity": validity,
              "landmarks_full_DHR_teacher_machine_matches_loaded": False}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-image", type=Path, required=True)
    parser.add_argument("--moving-image", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = fit(args.fixed_image, args.moving_image, args.affine, args.output,
                 steps=args.steps, device_name=args.device)
    print(json.dumps({key: result[key] for key in (
        "steps", "initial_image_loss", "last_exported_P1_image_loss",
        "training_seconds", "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
