"""Low-dimensional multilevel residual proposals steered through a safe 257² P1 decoder.

Teacher-map mode is only a representation-capacity diagnostic. Image mode
uses images and image-derived matches, never anatomical landmarks.
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
from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_kernel_match_safe257 import steer
from tools.digital_mind_amortized_network import structural_loss
from tools.digital_mind_objective_probe import self_similarity
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_mind_sparse_match_finetune import p1_at_points, robust_match_loss
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def multilevel_goal(baseline: torch.Tensor,
                    controls: nn.ParameterList) -> torch.Tensor:
    if baseline.shape != (1, 257, 257, 2):
        raise ValueError("one 257-square baseline required")
    residual = torch.zeros_like(baseline).permute(0, 3, 1, 2)
    for control in controls:
        if control.ndim != 4 or control.shape[:2] != (1, 2) or (
            control.shape[2] != control.shape[3]
        ):
            raise ValueError("control must be 1x2x(s-2)x(s-2)")
        padded = F.pad(control, (1, 1, 1, 1))
        residual = residual + F.interpolate(
            padded, size=(257, 257), mode="bilinear", align_corners=True)
    return baseline + residual.permute(0, 2, 3, 1)


def optimize(baseline_path: Path, output: Path, *,
             teacher_path: Path | None, fixed_path: Path | None,
             moving_path: Path | None, matches_path: Path | None,
             levels: tuple[int, ...] = (17, 33, 65),
             steps: int = 50, learning_rate: float = .002,
             passes: int = 4, strain_weight: float = 1.,
             minimum_jacobian: float = .05,
             device_name: str = "cuda:0") -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if (not levels or any(s < 3 or s > 257 for s in levels)
            or len(set(levels)) != len(levels) or
            steps < 1 or learning_rate <= 0 or passes < 1 or
            strain_weight < 0 or not 0 < minimum_jacobian < 1):
        raise ValueError("invalid multilevel settings")
    teacher_mode = teacher_path is not None
    if teacher_mode and any(p is not None for p in
                            (fixed_path, moving_path, matches_path)):
        raise ValueError("teacher diagnostic does not read image inputs")
    if not teacher_mode and any(p is None for p in
                                (fixed_path, moving_path, matches_path)):
        raise ValueError("image mode needs fixed/moving/matches")
    device = torch.device(device_name)
    with np.load(baseline_path) as data:
        initial = data["vertices"].astype(np.float32)
        matrix = data["post_affine_matrix"].astype(np.float32)
        offset = data["post_affine_offset"].astype(np.float32)
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine required")
    baseline = torch.from_numpy(initial.copy()).to(device)
    identity = identity_vertices(257, device=device)
    if not validate_q1_map(baseline, identity)["valid"]:
        raise ValueError("baseline must be certified")
    teacher = None
    source = target = fdesc = mdesc = mask = None
    if teacher_mode:
        with np.load(teacher_path) as data:
            if not (np.array_equal(data["post_affine_matrix"], matrix)
                    and np.array_equal(data["post_affine_offset"], offset)):
                raise ValueError("teacher and baseline affine differ")
            teacher = torch.from_numpy(data["vertices"].copy()).to(device)
    else:
        with np.load(matches_path) as data:
            source = torch.from_numpy(data["source_fixed_unit"].copy()).to(device)
            target = torch.from_numpy(data["target_aligned_unit"].copy()).to(device)
            if "post_affine_matrix" in data and not (
                np.array_equal(data["post_affine_matrix"], matrix) and
                np.array_equal(data["post_affine_offset"], offset)
            ):
                raise ValueError("match affine frame differs")
        if (len(source) < 16 or source.shape != target.shape or
                not bool(torch.isfinite(source).all()) or
                not bool(torch.isfinite(target).all())):
            raise ValueError("at least 16 finite paired matches required")
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
    controls = nn.ParameterList([
        nn.Parameter(torch.zeros((1, 2, s - 2, s - 2), device=device))
        for s in levels
    ])
    optimizer = torch.optim.Adam(controls, lr=learning_rate)
    best = baseline.detach().clone()
    best_total = float("inf")
    best_step = 0
    best_values = {}
    trace = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        goal = multilevel_goal(baseline, controls)
        mapped = steer(baseline, goal, passes=passes,
                       minimum_jacobian=minimum_jacobian)
        if teacher_mode:
            map_mse = (mapped - teacher).square().sum(-1).mean()
            total = map_mse
            values = {"teacher_map_rmse_unit": float(map_mse.detach().sqrt())}
        else:
            image = structural_loss(fdesc, mdesc, mask, mapped)
            match = robust_match_loss(p1_at_points(mapped, source), target)
            strain = strain_penalty(mapped)
            total = image + .05 * match + strain_weight * strain
            values = {"image": float(image.detach()),
                      "match_robust_px": float(match.detach()),
                      "strain": float(strain.detach())}
        value = float(total.detach())
        if value < best_total:
            best_total = value
            best = mapped.detach().clone()
            best_step = step
            best_values = values
        total.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in controls):
            raise FloatingPointError(f"invalid multilevel VJP at step {step}")
        optimizer.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            trace.append({"step": step, "total": value, **values})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=best.cpu().numpy(),
                        boundary_reference=identity.cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not binary["valid"] or not validate_q1_map(best, identity)["valid"]:
        raise RuntimeError("saved multilevel map invalid")
    report = {
        "method": "multilevel proposed goal plus four-pass safe P1 steering",
        "mode": "teacher-capacity-only" if teacher_mode else "unlabeled-image-match",
        "baseline": str(baseline_path),
        "teacher": None if teacher_path is None else str(teacher_path),
        "fixed": None if fixed_path is None else str(fixed_path),
        "moving": None if moving_path is None else str(moving_path),
        "matches": None if matches_path is None else str(matches_path),
        "levels": levels, "control_parameter_count": sum(2 * (s - 2) ** 2 for s in levels),
        "steps": steps, "learning_rate": learning_rate,
        "passes": passes, "strain_weight": strain_weight,
        "minimum_jacobian": minimum_jacobian,
        "best_step": best_step, "best_total": best_total,
        "best_values": best_values, "training_seconds": elapsed,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "saved_certificate": binary, "trace": trace,
        "anatomical_landmarks_or_DHR_full_field_loaded": False,
        "not_amortized_fast_inference": True,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--fixed", type=Path)
    parser.add_argument("--moving", type=Path)
    parser.add_argument("--matches", type=Path)
    parser.add_argument("--levels", default="17,33,65")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--strain-weight", type=float, default=1.)
    parser.add_argument("--minimum-jacobian", type=float, default=.05)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = optimize(
        args.baseline, args.output, teacher_path=args.teacher,
        fixed_path=args.fixed, moving_path=args.moving, matches_path=args.matches,
        levels=tuple(int(s) for s in args.levels.split(",")),
        steps=args.steps, learning_rate=args.learning_rate,
        passes=args.passes, strain_weight=args.strain_weight,
        minimum_jacobian=args.minimum_jacobian, device_name=args.device)
    print(json.dumps({key: result[key] for key in (
        "mode", "control_parameter_count", "best_step", "best_total",
        "training_seconds", "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
