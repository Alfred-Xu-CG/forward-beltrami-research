"""Slow label-free match/image objective optimization of repeated safe F1 passes.

This measures whether informative sparse image correspondences can activate
the certified P1 decoder. It is not an amortized fast inference result.
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
from tools.digital_mind_sparse_match_finetune import p1_at_points, robust_match_loss
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def optimize(fixed_path: Path, moving_path: Path, baseline_path: Path,
             matches_path: Path, output: Path, *, steps: int, passes: int,
             match_weight: float, image_weight: float, strain_weight: float,
             device_name: str, minimum_jacobian: float = .05) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if (steps < 1 or passes < 1 or min(match_weight, image_weight, strain_weight) < 0
            or not 0 < minimum_jacobian < 1):
        raise ValueError("invalid settings")
    device = torch.device(device_name)
    with np.load(baseline_path) as archive:
        vertices = np.asarray(archive["vertices"], dtype=np.float32)
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    with np.load(matches_path) as archive:
        source = torch.from_numpy(archive["source_fixed_unit"].copy()).to(device)
        target = torch.from_numpy(archive["target_aligned_unit"].copy()).to(device)
        match_affine = (archive["post_affine_matrix"].copy()
                        if "post_affine_matrix" in archive.files else None)
        match_offset = (archive["post_affine_offset"].copy()
                        if "post_affine_offset" in archive.files else None)
    if vertices.shape != (1, 257, 257, 2) or len(source) < 16:
        raise ValueError("genuine 257 map and at least 16 matches required")
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2
            or not bool(torch.isfinite(source).all())
            or not bool(torch.isfinite(target).all())
            or bool((source < 0).any()) or bool((source > 1).any())
            or bool((target < 0).any()) or bool((target > 1).any())):
        raise ValueError("finite paired unit-square matches required")
    if (match_affine is None) != (match_offset is None):
        raise ValueError("partial match affine metadata")
    if match_affine is not None and not (
        np.array_equal(match_affine, matrix) and np.array_equal(match_offset, offset)
    ):
        raise ValueError("match archive uses a different affine frame")
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive baseline affine required")
    baseline = torch.from_numpy(vertices.copy()).to(device)
    identity = identity_vertices(257, device=device)
    if not validate_q1_map(baseline, identity)["valid"]:
        raise ValueError("baseline invalid")
    fixed, _ = _read_gray_thumbnail(fixed_path, 512)
    moving, _ = _read_gray_thumbnail(moving_path, 512)
    fixed, moving = fixed.to(device), moving.to(device)
    with torch.no_grad():
        aligned = warp_moving_to_fixed(
            moving, torch.tensor(matrix, device=device),
            torch.tensor(offset, device=device), height=512, width=512)
        fixed256 = F.interpolate(fixed, size=(256, 256), mode="area")
        moving256 = F.interpolate(aligned, size=(256, 256), mode="area")
        fdesc, fscale = self_similarity(fixed256)
        mdesc, _ = self_similarity(moving256)
        mask = ((fixed256 > .04) & (fscale > 1e-4)).to(fixed256.dtype)
    logits = nn.ParameterList(nn.Parameter(torch.zeros(
        (1, 255, 255, 2), device=device)) for _ in range(passes))
    update = AdaptiveSoftRadialQ1Relaxation(
        257, raw_span=8., safety_fraction=.75,
        minimum_jacobian=minimum_jacobian).to(device)
    optimizer = torch.optim.Adam(logits, lr=.02)
    best_total = float("inf")
    best = baseline.detach().clone()
    trace = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mapped = baseline
        for proposal in logits:
            mapped = update(mapped, proposal)
        image = structural_loss(fdesc, mdesc, mask, mapped)
        match = robust_match_loss(p1_at_points(mapped, source), target)
        strain = strain_penalty(mapped)
        total = image_weight * image + match_weight * match + strain_weight * strain
        value = float(total.detach())
        if value < best_total:
            best_total = value
            best = mapped.detach().clone()
            best_step = step
            best_image, best_match, best_strain = (
                float(image.detach()), float(match.detach()), float(strain.detach()))
        total.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in logits):
            raise FloatingPointError(f"invalid match-guided VJP at {step}")
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "image": float(image.detach()),
                          "match_robust_px": float(match.detach()),
                          "strain": float(strain.detach()), "total": value})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=best.cpu().numpy(),
                        boundary_reference=identity.cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not validate_q1_map(best, identity)["valid"] or not binary["valid"]:
        raise RuntimeError("saved map invalid")
    report = {
        "method": "slow image/machine-match-guided repeated safe-F1 pair optimization",
        "baseline": str(baseline_path), "matches": str(matches_path),
        "match_count": len(source), "output": str(output),
        "passes": passes, "steps": steps, "learning_rate": .02,
        "minimum_jacobian": minimum_jacobian,
        "image_weight": image_weight, "match_weight": match_weight,
        "strain_weight": strain_weight, "best_step": best_step,
        "best_total": best_total, "best_image": best_image,
        "best_match_robust_px": best_match, "best_strain": best_strain,
        "training_seconds": elapsed,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "trace": trace, "saved_certificate": binary,
        "anatomical_landmarks_or_DHR_full_field_used": False,
        "not_amortized_fast_inference": True,
        "match_affine_frame_machine_checked": match_affine is not None,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "moving", "baseline", "matches", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--match-weight", type=float, default=.05)
    parser.add_argument("--image-weight", type=float, default=1.)
    parser.add_argument("--strain-weight", type=float, default=.1)
    parser.add_argument("--minimum-jacobian", type=float, default=.05)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = optimize(args.fixed, args.moving, args.baseline, args.matches,
                      args.output, steps=args.steps, passes=args.passes,
                      match_weight=args.match_weight,
                      image_weight=args.image_weight,
                      strain_weight=args.strain_weight,
                      device_name=args.device,
                      minimum_jacobian=args.minimum_jacobian)
    print(json.dumps({key: result[key] for key in (
        "passes", "steps", "match_count", "best_image",
        "best_match_robust_px", "best_strain", "training_seconds",
        "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
