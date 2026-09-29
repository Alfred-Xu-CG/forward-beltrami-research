"""Image-only per-pair MIND-like objective test with a safe F1 decoder.

This deliberately measures whether structural image evidence helps registration.
It is a 200-step per-pair optimizer, not the desired amortized neural layer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveSoftRadialQ1Relaxation, q1_dyadic_refine, validate_q1_map,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_mind_objective_probe import self_similarity, sampled_p1_descriptor
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_real_optimize import _read_gray_thumbnail


class SafeThreeLevelF1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sides = (17, 33, 65)
        self.updates = nn.ModuleList(
            AdaptiveSoftRadialQ1Relaxation(side, raw_span=8.,
                                           safety_fraction=.75,
                                           minimum_jacobian=.05)
            for side in self.sides
        )
        self.logits = nn.ParameterList(
            nn.Parameter(torch.zeros((1, side - 2, side - 2, 2)))
            for side in self.sides
        )

    def forward(self) -> torch.Tensor:
        current = identity_vertices(17, device=self.logits[0].device).to(self.logits[0].dtype)
        for i, layer in enumerate(self.updates):
            if i:
                current = q1_dyadic_refine(current)
            current = layer(current, self.logits[i])
        return current


def strain_penalty(mapped: torch.Tensor) -> torch.Tensor:
    side = mapped.shape[1]
    reference = identity_vertices(side, device=mapped.device).to(mapped.dtype)
    residual = mapped - reference
    dx = (residual[:, :, 1:] - residual[:, :, :-1]) * (side - 1)
    dy = (residual[:, 1:] - residual[:, :-1]) * (side - 1)
    return .5 * (dx.square().sum(-1).mean() + dy.square().sum(-1).mean())


def optimize(fixed_path: Path, moving_path: Path, affine_path: Path,
             output: Path, *, steps: int, device_name: str) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if steps < 1:
        raise ValueError("positive step count required")
    with np.load(affine_path) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    if matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError("2x2 affine and 2-vector required")
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine determinant required")
    device = torch.device(device_name)
    fixed, _ = _read_gray_thumbnail(fixed_path, 128)
    moving, _ = _read_gray_thumbnail(moving_path, 128)
    fixed, moving = fixed.to(device), moving.to(device)
    with torch.no_grad():
        fixed_feature, fixed_scale = self_similarity(fixed)
        moving_feature, _ = self_similarity(moving)
        mask = ((fixed > .04) & (fixed_scale > 1e-4)).to(fixed.dtype)
    affine_matrix = torch.tensor(matrix.copy(), device=device)
    affine_offset = torch.tensor(offset.copy(), device=device)
    model = SafeThreeLevelF1().to(device)
    optimizer = torch.optim.Adam(model.logits, lr=.02)
    best_objective = float("inf")
    best_map = None
    trace = []
    durations = []
    finite_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        mapped = model()
        effective = mapped @ affine_matrix.T + affine_offset
        warped_feature = sampled_p1_descriptor(moving_feature, effective)
        image_loss = ((fixed_feature - warped_feature).abs() * mask).sum() / (
            mask.sum() * fixed_feature.shape[1] + 1e-8)
        penalty = strain_penalty(mapped)
        loss = image_loss + penalty  # predeclared lambda = 1
        value = float(loss.detach())
        if value < best_objective:
            best_objective = value
            best_map = mapped.detach().clone()
            best_step = step
            best_image = float(image_loss.detach())
            best_penalty = float(penalty.detach())
        loss.backward()
        finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                     for p in model.logits)
        finite_steps += int(finite)
        if not finite:
            raise FloatingPointError(f"nonfinite latent VJP at step {step}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            trace.append({"step": step, "image_loss": float(image_loss.detach()),
                          "strain_penalty": float(penalty.detach()),
                          "total_loss": value})
    assert best_map is not None
    output_map = best_map
    while output_map.shape[1] < 257:
        output_map = q1_dyadic_refine(output_map)
    validity = validate_q1_map(output_map, identity_vertices(257, device=device))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=output_map.cpu().numpy(),
                        boundary_reference=identity_vertices(257, device=device).cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not validity["valid"] or not binary["valid"]:
        raise RuntimeError("saved optimized map failed topology check")
    report = {"method": "per-pair MIND-like descriptor + safe 17/33/65 F1",
              "fixed": str(fixed_path), "moving": str(moving_path),
              "affine": str(affine_path), "map": str(output),
              "steps": steps, "learning_rate": .02, "strain_lambda": 1.,
              "best_pre_update_step": best_step, "best_regularized_loss": best_objective,
              "best_image_loss": best_image, "best_strain_penalty": best_penalty,
              "initial_image_loss": trace[0]["image_loss"],
              "median_step_seconds": statistics.median(durations),
              "finite_gradient_steps": finite_steps,
              "peak_torch_cuda_allocated_bytes": (
                  None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
              "in_memory_validity": validity,
              "saved_binary_certificate": binary,
              "loss_trace": trace,
              "landmarks_or_full_DHR_displacement_loaded": False}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-image", type=Path, required=True)
    parser.add_argument("--moving-image", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = optimize(args.fixed_image, args.moving_image, args.affine, args.output,
                      steps=args.steps, device_name=args.device)
    print(json.dumps({key: result[key] for key in (
        "steps", "initial_image_loss", "best_image_loss", "best_strain_penalty",
        "best_pre_update_step", "median_step_seconds", "finite_gradient_steps",
    )}))


if __name__ == "__main__":
    main()
