"""Same-objective per-pair safe-F1 optimization atop a frozen neural map.

This is a diagnostic upper-bound experiment, not a fast amortized layer.
No anatomical landmark or DHR full field is loaded; checkpointing is by the
same 256-square structural-descriptor-plus-strain objective as training.
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

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveSoftRadialQ1Relaxation, validate_q1_map,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_mind_amortized_network import structural_loss
from tools.digital_mind_objective_probe import self_similarity
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def optimize(fixed_path: Path, moving_path: Path, baseline_path: Path,
             output: Path, *, steps: int, device_name: str) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if steps < 1:
        raise ValueError("positive step count required")
    device = torch.device(device_name)
    with np.load(baseline_path) as archive:
        vertices = np.asarray(archive["vertices"], dtype=np.float32)
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    if vertices.shape != (1, 257, 257, 2) or matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError("genuine 257-square map and affine required")
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive saved affine required")
    baseline = torch.from_numpy(vertices.copy()).to(device)
    identity = identity_vertices(257, device=device)
    if not validate_q1_map(baseline, identity)["valid"]:
        raise ValueError("baseline map invalid")
    fixed, _ = _read_gray_thumbnail(fixed_path, 512)
    moving, _ = _read_gray_thumbnail(moving_path, 512)
    fixed, moving = fixed.to(device), moving.to(device)
    with torch.no_grad():
        aligned = warp_moving_to_fixed(moving, torch.tensor(matrix, device=device),
                                      torch.tensor(offset, device=device),
                                      height=512, width=512)
        fixed256 = F.interpolate(fixed, size=(256, 256), mode="area")
        moving256 = F.interpolate(aligned, size=(256, 256), mode="area")
        fdesc, fscale = self_similarity(fixed256)
        mdesc, _ = self_similarity(moving256)
        mask = ((fixed256 > .04) & (fscale > 1e-4)).to(fixed256.dtype)
    logits = nn.Parameter(torch.zeros((1, 255, 255, 2), device=device))
    update = AdaptiveSoftRadialQ1Relaxation(
        257, raw_span=8., safety_fraction=.75, minimum_jacobian=.05).to(device)
    optimizer = torch.optim.Adam([logits], lr=.02)
    best_total = float("inf")
    best = baseline.detach().clone()
    best_step = -1
    trace = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mapped = update(baseline, logits)
        image = structural_loss(fdesc, mdesc, mask, mapped)
        strain = strain_penalty(mapped)
        total = image + strain
        value = float(total.detach())
        if value < best_total:
            best_total = value
            best = mapped.detach().clone()
            best_step = step
            best_image = float(image.detach())
            best_strain = float(strain.detach())
        total.backward()
        if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
            raise FloatingPointError(f"invalid latent VJP at step {step}")
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "image": float(image.detach()),
                          "strain": float(strain.detach()), "total": value})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    validity = validate_q1_map(best, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=best.cpu().numpy(),
                        boundary_reference=identity.cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not validity["valid"] or not binary["valid"]:
        raise RuntimeError("optimized map topology failure")
    report = {
        "method": "slow same-objective one-more-safe-F1 per-pair diagnostic",
        "baseline": str(baseline_path), "output": str(output),
        "steps": steps, "learning_rate": .02,
        "baseline_image": trace[0]["image"],
        "baseline_strain": trace[0]["strain"],
        "best_step": best_step, "best_total": best_total,
        "best_image": best_image, "best_strain": best_strain,
        "training_seconds": elapsed,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "trace": trace, "saved_certificate": binary,
        "anatomical_landmarks_used": False,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "moving", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = optimize(args.fixed, args.moving, args.baseline, args.output,
                      steps=args.steps, device_name=args.device)
    print(json.dumps({key: result[key] for key in (
        "baseline_image", "best_image", "best_step", "training_seconds",
        "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
