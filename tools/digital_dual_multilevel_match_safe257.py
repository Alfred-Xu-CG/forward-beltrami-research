"""Dual ridge fit of sparse matches to multilevel controls, then safe 257² P1 steering.

The N-by-N solve scales with machine-match count, not the dense mesh size.
No anatomical landmark or full DHR field is read. Correspondence selection and
affine estimation remain external discrete operations.
"""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_kernel_match_safe257 import steer
from tools.digital_mind_sparse_match_finetune import p1_at_points
from tools.digital_q1_dhr_distill import identity_vertices


def multilevel_design(query: torch.Tensor,
                      levels: tuple[int, ...] = (17, 33, 65)) -> torch.Tensor:
    """Exact fixed-diagonal 257-P1 evaluation of prolonged interior hat bases."""
    if query.ndim != 2 or query.shape[1] != 2 or len(query) < 1:
        raise ValueError("nonempty Nx2 query required")
    if not bool(torch.isfinite(query).all()) or bool(((query < 0) | (query > 1)).any()):
        raise ValueError("query must lie in the reference square")
    scaled = 256 * query
    ij = torch.floor(scaled).long().clamp(0, 255)
    x, y = ij[:, 0], ij[:, 1]
    s, t = scaled[:, 0] - x, scaled[:, 1] - y
    lower = t <= s
    vx = (x, torch.where(lower, x + 1, x), x + 1)
    vy = (y, torch.where(lower, y, y + 1), y + 1)
    bary = (torch.where(lower, 1 - s, 1 - t),
            torch.where(lower, s - t, t - s),
            torch.where(lower, t, s))
    width = sum((side - 2) ** 2 for side in levels)
    design = torch.zeros((len(query), width), device=query.device, dtype=query.dtype)
    offset = 0
    for side in levels:
        stride = side - 2
        for px, py, vertex_weight in zip(vx, vy, bary, strict=True):
            u = px.to(query.dtype) * ((side - 1) / 256)
            v = py.to(query.dtype) * ((side - 1) / 256)
            cx = torch.floor(u).long().clamp(0, side - 2)
            cy = torch.floor(v).long().clamp(0, side - 2)
            fx, fy = u - cx, v - cy
            for ox, oy, weight in (
                (0, 0, (1 - fx) * (1 - fy)),
                (1, 0, fx * (1 - fy)),
                (0, 1, (1 - fx) * fy),
                (1, 1, fx * fy),
            ):
                ix, iy = cx + ox, cy + oy
                interior = ((ix > 0) & (ix < side - 1)
                            & (iy > 0) & (iy < side - 1))
                column = offset + (iy - 1) * stride + (ix - 1)
                column = column.clamp(offset, offset + stride * stride - 1)
                design.scatter_add_(1, column[:, None],
                                    (vertex_weight * weight * interior)[:, None])
        offset += stride * stride
    return design


def dual_fit_map(baseline: torch.Tensor, source: torch.Tensor,
                 target: torch.Tensor, *, ridge: float,
                 levels: tuple[int, ...] = (17, 33, 65),
                 passes: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("positive finite ridge required")
    if baseline.shape != (1, 257, 257, 2) or source.shape != target.shape:
        raise ValueError("baseline and matched Nx2 points required")
    design = multilevel_design(source, levels)
    desired = target - p1_at_points(baseline, source)
    gram = design @ design.T
    system = gram + ridge * torch.eye(len(source), device=source.device,
                                      dtype=source.dtype)
    dual = torch.linalg.solve(system, desired)
    coefficients = design.T @ dual
    residual = torch.zeros_like(baseline).permute(0, 3, 1, 2)
    offset = 0
    for side in levels:
        count = (side - 2) ** 2
        control = coefficients[offset:offset + count].T.reshape(
            1, 2, side - 2, side - 2)
        residual = residual + F.interpolate(
            F.pad(control, (1, 1, 1, 1)), size=(257, 257),
            mode="bilinear", align_corners=True)
        offset += count
    goal = baseline + residual.permute(0, 2, 3, 1)
    return steer(baseline, goal, passes=passes), coefficients


def run(baseline_path: Path, matches_path: Path, output: Path, *,
        ridge: float = .1, passes: int = 4,
        device_name: str = "cuda:0") -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    device = torch.device(device_name)
    with np.load(baseline_path) as data:
        baseline = torch.from_numpy(data["vertices"].copy()).to(device)
        matrix = data["post_affine_matrix"].copy()
        offset = data["post_affine_offset"].copy()
    a, b, c, d = (Fraction.from_float(float(x)) for x in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine required")
    with np.load(matches_path) as data:
        source = torch.from_numpy(data["source_fixed_unit"].copy()).to(device)
        target = torch.from_numpy(data["target_aligned_unit"].copy()).to(device)
        if "post_affine_matrix" in data and not (
            np.array_equal(data["post_affine_matrix"], matrix) and
            np.array_equal(data["post_affine_offset"], offset)
        ):
            raise ValueError("match affine frame differs")
    if len(source) < 8 or source.shape != target.shape:
        raise ValueError("at least eight matched points required")
    reference = identity_vertices(257, device=device)
    if not validate_q1_map(baseline, reference)["valid"]:
        raise ValueError("baseline is not a valid square map")
    source.requires_grad_(True)
    target.requires_grad_(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    mapped, controls = dual_fit_map(baseline, source, target,
                                    ridge=ridge, passes=passes)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    forward_peak = (None if device.type != "cuda" else
                    int(torch.cuda.max_memory_allocated(device)))
    loss = (mapped - baseline).square().mean()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - started
    vjp_peak = (None if device.type != "cuda" else
                int(torch.cuda.max_memory_allocated(device)))
    finite = bool(torch.isfinite(source.grad).all() and
                  torch.isfinite(target.grad).all())
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=mapped.detach().cpu().numpy(),
                        boundary_reference=reference.cpu().numpy(),
                        post_affine_matrix=matrix, post_affine_offset=offset)
    binary = certify_q1_binary_map(output)
    if not binary["valid"] or not validate_q1_map(mapped.detach(), reference)["valid"]:
        raise RuntimeError("saved map invalid")
    report = {
        "method": "dual ridge multilevel hat-basis fit plus safe four-pass P1 steering",
        "baseline": str(baseline_path), "matches": str(matches_path),
        "match_count": len(source), "ridge": ridge, "passes": passes,
        "control_scalar_count": 2 * len(controls),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "forward_peak_torch_cuda_allocated_bytes": forward_peak,
        "vjp_peak_torch_cuda_allocated_bytes": vjp_peak,
        "finite_match_coordinate_vjp": finite,
        "saved_certificate": binary,
        "anatomical_landmarks_or_DHR_full_field_loaded": False,
        "timing_excludes_match_generation_and_file_io": True,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                          encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=.1)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.baseline, args.matches, args.output,
                 ridge=args.ridge, passes=args.passes,
                 device_name=args.device)
    print(json.dumps({key: result[key] for key in (
        "match_count", "ridge", "forward_seconds", "backward_seconds",
        "finite_match_coordinate_vjp")}))


if __name__ == "__main__":
    main()
