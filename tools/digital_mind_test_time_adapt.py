"""Adapt an ACROBAT-trained safe network to one image pair without labels.

The saved map is selected by the structural-image objective, never by anatomy.
This is a test-time optimizer, not a one-pass inference timing claim.
"""

from __future__ import annotations

import argparse
import copy
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


def adapt(checkpoint: Path, fixed_path: Path, moving_path: Path,
          affine_path: Path, output: Path, *, steps: int,
          device_name: str, learning_rate: float = .001) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and rate must be positive")
    device = torch.device(device_name)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MindSafeImageNetwork().to(device)
    model.load_state_dict(saved["state_dict"])
    model.train()
    fixed, _ = _read_gray_thumbnail(fixed_path, 512)
    moving, _ = _read_gray_thumbnail(moving_path, 512)
    fixed, moving = fixed.to(device), moving.to(device)
    with np.load(affine_path) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    if matrix.shape != (2, 2) or offset.shape != (2,) or (
        not np.isfinite(matrix).all() or not np.isfinite(offset).all()
    ):
        raise ValueError("invalid affine")
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine required")
    with torch.no_grad():
        prewarp = warp_moving_to_fixed(
            moving, torch.tensor(matrix, device=device),
            torch.tensor(offset, device=device), height=512, width=512,
        )
        feature, fixed_desc, moving_desc, mask = image_features(
            F.interpolate(fixed, size=(128, 128), mode="area"),
            F.interpolate(prewarp, size=(128, 128), mode="area"),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                   weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    trace = []
    with torch.no_grad():
        initial_map = model(feature)
        initial_image = float(structural_loss(fixed_desc, moving_desc,
                                               mask, initial_map))
        initial_strain = float(strain_penalty(initial_map))
    best_total = initial_image + initial_strain
    best_step = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        mapped = model(feature)
        image = structural_loss(fixed_desc, moving_desc, mask, mapped)
        strain = strain_penalty(mapped)
        loss = image + strain
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
            raise FloatingPointError(f"nonfinite VJP at step {step}")
        nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        with torch.no_grad():
            after = model(feature)
            image_after = float(structural_loss(fixed_desc, moving_desc,
                                                 mask, after))
            strain_after = float(strain_penalty(after))
            total_after = image_after + strain_after
        if total_after < best_total:
            best_total = total_after
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
        if step in (1, 5, 10, 25, 50, steps):
            trace.append({"step": step, "image": image_after,
                          "strain": strain_after, "total": total_after})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final = model(feature, final_side=257)
        final_image = float(structural_loss(fixed_desc, moving_desc, mask,
                                            final))
        valid = validate_q1_map(final, identity_vertices(257, device=device))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=final.cpu().numpy(),
                        boundary_reference=identity_vertices(257, device=device).cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not valid["valid"] or not binary["valid"]:
        raise RuntimeError("saved map invalid")
    result = {
        "method": "image-only test-time adaptation of ACROBAT-trained safe network",
        "checkpoint": str(checkpoint), "fixed": str(fixed_path),
        "moving": str(moving_path), "affine": str(affine_path),
        "map": str(output), "steps": steps, "learning_rate": learning_rate,
        "best_step_by_65_P1_image_plus_strain": best_step,
        "initial_65_P1_image_loss": initial_image,
        "initial_65_P1_strain": initial_strain,
        "best_65_P1_total": best_total,
        "exported_257_P1_image_loss": final_image,
        "adaptation_seconds_excluding_feature_and_io": seconds,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "trace": trace, "saved_binary_certificate": binary,
        "in_memory_validity": valid,
        "landmarks_full_DHR_teacher_machine_matches_loaded": False,
        "DHR_derived_initial_affine_used": True,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n",
                                           encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "fixed_image", "moving_image", "affine", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = adapt(args.checkpoint, args.fixed_image, args.moving_image,
                   args.affine, args.output, steps=args.steps,
                   device_name=args.device, learning_rate=args.learning_rate)
    print(json.dumps({key: result[key] for key in (
        "best_step_by_65_P1_image_plus_strain", "initial_65_P1_image_loss",
        "best_65_P1_total", "exported_257_P1_image_loss",
        "adaptation_seconds_excluding_feature_and_io",
    )}))


if __name__ == "__main__":
    main()
