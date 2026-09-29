"""Small-kernel correspondence latent to a certified 257-square P1 map.

The input matcher and initial affine are external. This forward layer solves an
N-by-N positive-definite kernel system (N is the match count), not a system on
the dense vertex grid, then steers four analytic current-edge safe updates.
"""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveSoftRadialQ1Relaxation, validate_q1_map,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_mind_sparse_match_finetune import p1_at_points, robust_match_loss
from tools.digital_q1_dhr_distill import identity_vertices


def kernel_target(baseline: torch.Tensor, source: torch.Tensor,
                  target: torch.Tensor, *, sigma: float = .1,
                  ridge: float = .1) -> torch.Tensor:
    """Return B plus a Gaussian kernel-ridge displacement on all 257² nodes."""
    if baseline.shape != (1, 257, 257, 2) or source.shape != target.shape or (
        source.ndim != 2 or source.shape[1] != 2 or len(source) < 16
    ):
        raise ValueError("one 257-square map and at least 16 paired matches required")
    if not 0 < sigma < 1 or ridge <= 0:
        raise ValueError("positive kernel bandwidth and ridge required")
    if not (torch.isfinite(source).all() and torch.isfinite(target).all()
            and ((source >= 0) & (source <= 1)).all()
            and ((target >= 0) & (target <= 1)).all()):
        raise ValueError("finite unit-square matches required")
    s, t = source.to(torch.float64), target.to(torch.float64)
    delta = t - p1_at_points(baseline, source).to(torch.float64)
    sq = torch.cdist(s, s).square()
    kernel = torch.exp(-sq / (2 * sigma * sigma))
    system = kernel + ridge * torch.eye(len(s), device=s.device, dtype=s.dtype)
    coefficients = torch.linalg.solve(system, delta)
    x = identity_vertices(257, device=baseline.device).reshape(-1, 2).to(torch.float64)
    values = []
    for q in x.split(8192):
        radial = torch.exp(-torch.cdist(q, s).square() / (2 * sigma * sigma))
        values.append(radial @ coefficients)
    displacement = torch.cat(values, dim=0).reshape(1, 257, 257, 2)
    return baseline + displacement.to(baseline.dtype)


def steer(baseline: torch.Tensor, goal: torch.Tensor, *, passes: int = 4,
          minimum_jacobian: float = .05) -> torch.Tensor:
    if passes < 1:
        raise ValueError("passes must be positive")
    if goal.shape != baseline.shape:
        raise ValueError("baseline and goal shapes differ")
    update = AdaptiveSoftRadialQ1Relaxation(
        257, raw_span=8., safety_fraction=.75,
        minimum_jacobian=minimum_jacobian).to(baseline.device)
    current = baseline
    for remaining in range(passes, 0, -1):
        horizontal = (current[:, 1:-1, 2:] - current[:, 1:-1, :-2]) / 2
        vertical = (current[:, 2:, 1:-1] - current[:, :-2, 1:-1]) / 2
        desired = (goal[:, 1:-1, 1:-1] - current[:, 1:-1, 1:-1]) / remaining
        det = horizontal[..., 0] * vertical[..., 1] - horizontal[..., 1] * vertical[..., 0]
        valid = det.abs() > 1e-10
        scale = torch.where(valid, 8 * det, torch.ones_like(det))
        u = (desired[..., 0] * vertical[..., 1]
             - desired[..., 1] * vertical[..., 0]) / scale
        v = (horizontal[..., 0] * desired[..., 1]
             - horizontal[..., 1] * desired[..., 0]) / scale
        logits = torch.atanh(torch.stack((u, v), dim=-1).clamp(-.98, .98))
        logits = torch.where(valid[..., None], logits, torch.zeros_like(logits))
        current = update(current, logits)
    return current


def run(baseline_path: Path, matches_path: Path, output: Path, *,
        passes: int = 4, sigma: float = .1, ridge: float = .1,
        minimum_jacobian: float = .05, device_name: str = "cuda:0") -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    device = torch.device(device_name)
    with np.load(baseline_path) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float32)
        matrix = np.asarray(data["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(data["post_affine_offset"], dtype=np.float32)
    with np.load(matches_path) as data:
        source = torch.from_numpy(data["source_fixed_unit"].copy()).to(device)
        target = torch.from_numpy(data["target_aligned_unit"].copy()).to(device)
        m_matrix = data["post_affine_matrix"].copy() if "post_affine_matrix" in data else None
        m_offset = data["post_affine_offset"].copy() if "post_affine_offset" in data else None
    if (m_matrix is None) != (m_offset is None):
        raise ValueError("partial match affine metadata")
    if m_matrix is not None and not (
        np.array_equal(m_matrix, matrix) and np.array_equal(m_offset, offset)
    ):
        raise ValueError("match archive and baseline affine frames differ")
    a, b, c, d = (Fraction.from_float(float(z)) for z in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine required")
    baseline = torch.from_numpy(vertices.copy()).to(device).requires_grad_(True)
    source.requires_grad_(True)
    reference = identity_vertices(257, device=device)
    if not validate_q1_map(baseline, reference)["valid"]:
        raise ValueError("invalid baseline map")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    target.requires_grad_(True)
    start = time.perf_counter()
    goal = kernel_target(baseline, source, target, sigma=sigma, ridge=ridge)
    mapped = steer(baseline, goal, passes=passes,
                   minimum_jacobian=minimum_jacobian)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start
    match_loss = robust_match_loss(p1_at_points(mapped, source), target)
    vjp_start = time.perf_counter()
    baseline_vjp, source_vjp, target_vjp = torch.autograd.grad(
        match_loss, (baseline, source, target))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    vjp_seconds = time.perf_counter() - vjp_start
    if not all(torch.isfinite(g).all() for g in
               (baseline_vjp, source_vjp, target_vjp)):
        raise FloatingPointError("nonfinite baseline/source/target VJP")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=mapped.detach().cpu().numpy(),
                        boundary_reference=reference.cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not binary["valid"] or not validate_q1_map(mapped.detach(), reference)["valid"]:
        raise RuntimeError("saved map invalid")
    result = {
        "method": "Gaussian kernel-ridge match latent plus four safe current-edge F1 passes",
        "baseline": str(baseline_path), "matches": str(matches_path),
        "output": str(output), "match_count": len(source),
        "sigma": sigma, "ridge": ridge, "passes": passes,
        "minimum_jacobian": minimum_jacobian,
        "input_match_robust_px": float(match_loss.detach()),
        "forward_seconds": forward_seconds,
        "baseline_source_target_vjp_seconds": vjp_seconds,
        "finite_baseline_source_target_vjp": True,
        "peak_torch_cuda_allocated_bytes": (
            None if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
        "saved_certificate": binary,
        "anatomical_landmarks_or_DHR_full_field_used": False,
        "external_matcher_is_not_part_of_differentiable_layer": True,
        "match_affine_frame_machine_checked": m_matrix is not None,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n",
                                           encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "matches", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=.1)
    parser.add_argument("--ridge", type=float, default=.1)
    parser.add_argument("--minimum-jacobian", type=float, default=.05)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.baseline, args.matches, args.output, passes=args.passes,
                 sigma=args.sigma, ridge=args.ridge,
                 minimum_jacobian=args.minimum_jacobian,
                 device_name=args.device)
    print(json.dumps({key: result[key] for key in (
        "match_count", "input_match_robust_px", "forward_seconds",
        "baseline_source_target_vjp_seconds", "peak_torch_cuda_allocated_bytes",
    )}))


if __name__ == "__main__":
    main()
