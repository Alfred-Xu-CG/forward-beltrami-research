"""One-pass Gaussian sparse-motion proposals through safe coarse-to-fine Q1.

This is a forward feature-to-map decoder. Image matching and the initial
affine are external; this module does not claim image-to-map CNN inference.
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

from qcopt.neural_bijection.dense.digital_q1 import (
    SafeColoredQ1Relaxation, q1_dyadic_refine, validate_q1_map,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_match_fit import q1_map_at_points


def _gaussian_displacement_direct(
    vertices: torch.Tensor, source: torch.Tensor, target: torch.Tensor, sigma: float,
) -> torch.Tensor:
    """Reference Gaussian sum; materializes one weight per vertex and match."""
    delta = vertices[0, :, :, None, :] - source[None, None, :, :]
    squared = delta.square().sum(-1)
    weights = torch.exp(-squared / (2 * sigma * sigma))
    motion = target - source
    return ((weights[..., None] * motion).sum(-2) /
            (weights.sum(-1)[..., None] + 1e-8))[None]


def _gaussian_displacement(
    vertices: torch.Tensor, source: torch.Tensor, target: torch.Tensor, sigma: float,
) -> torch.Tensor:
    """Exact separable Gaussian sum on the rectangular reference grid."""
    x = vertices[0, 0, :, 0]
    y = vertices[0, :, 0, 1]
    x_weights = torch.exp(-((x[:, None] - source[None, :, 0]).square()) /
                          (2 * sigma * sigma))
    y_weights = torch.exp(-((y[:, None] - source[None, :, 1]).square()) /
                          (2 * sigma * sigma))
    x_transpose = x_weights.transpose(0, 1)
    mass = y_weights @ x_transpose
    motion = target - source
    numerator_x = (y_weights * motion[None, :, 0]) @ x_transpose
    numerator_y = (y_weights * motion[None, :, 1]) @ x_transpose
    return (torch.stack((numerator_x, numerator_y), dim=-1) /
            (mass[..., None] + 1e-8))[None]


def _gaussian_displacement_at_points(
    queries: torch.Tensor, source: torch.Tensor, target: torch.Tensor, sigma: float,
) -> torch.Tensor:
    """Evaluate the same Gaussian field at a small calibration point set."""
    squared = (queries[:, None, :] - source[None, :, :]).square().sum(-1)
    weights = torch.exp(-squared / (2 * sigma * sigma))
    return (weights @ (target - source)) / (weights.sum(-1, keepdim=True) + 1e-8)


def forward_kernel_map(
    source_points: torch.Tensor, target_points: torch.Tensor, *,
    final_side: int = 257, sigma: float = .12,
    update_sigmas: tuple[float, ...] | None = None,
    proposal_mode: str = "absolute",
    gain_mode: str = "fixed",
    calibration_source: torch.Tensor | None = None,
    calibration_target: torch.Tensor | None = None,
    gain_trace: list[float] | None = None,
) -> torch.Tensor:
    """Generate a positive-Q1 table by safe multilevel F1-D updates."""
    if source_points.shape != target_points.shape or source_points.ndim != 2 or (
        source_points.shape[1] != 2 or len(source_points) < 1 or
        source_points.dtype != target_points.dtype or
        source_points.dtype not in (torch.float32, torch.float64) or
        source_points.device != target_points.device or
        not bool(torch.isfinite(source_points).all()) or
        not bool(torch.isfinite(target_points).all()) or
        not np.isfinite(sigma) or sigma <= 0
    ):
        raise ValueError("matching finite Nx2 points and positive sigma required")
    if final_side < 17:
        raise ValueError("final_side must be a dyadic refinement of 17")
    if proposal_mode not in {"absolute", "residual"}:
        raise ValueError("proposal_mode must be absolute or residual")
    if gain_mode not in {"fixed", "calibrated"}:
        raise ValueError("gain_mode must be fixed or calibrated")
    if gain_mode == "calibrated" and (
        proposal_mode != "residual" or calibration_source is None or
        calibration_target is None or
        calibration_source.shape != calibration_target.shape or
        calibration_source.ndim != 2 or calibration_source.shape[1] != 2 or
        len(calibration_source) < 1 or
        calibration_source.dtype != source_points.dtype or
        calibration_source.device != source_points.device or
        calibration_target.dtype != source_points.dtype or
        calibration_target.device != source_points.device or
        not bool(torch.isfinite(calibration_source).all()) or
        not bool(torch.isfinite(calibration_target).all())
    ):
        raise ValueError("calibrated residual gain needs matching finite calibration points")
    sides = [17]
    while sides[-1] < final_side:
        sides.append(2 * sides[-1] - 1)
    if sides[-1] != final_side:
        raise ValueError("final_side must be a dyadic refinement of 17")
    sigma_floor = float(np.sqrt(torch.finfo(source_points.dtype).tiny))
    widths = ((sigma,) * min(3, len(sides)) if update_sigmas is None
              else tuple(update_sigmas))
    if not widths or len(widths) > len(sides) or any(
        not np.isfinite(width) or width < sigma_floor for width in widths
    ):
        raise ValueError("numerically valid update widths required for a prefix of grid levels")
    current = identity_vertices(17, device=source_points.device).to(source_points.dtype)
    for index, side in enumerate(sides):
        if index:
            current = q1_dyadic_refine(current)
        if index >= len(widths):
            continue
        reference = identity_vertices(side, device=source_points.device).to(source_points.dtype)
        if proposal_mode == "absolute":
            desired = reference + _gaussian_displacement(
                reference, source_points, target_points, widths[index],
            )
        else:
            predicted = q1_map_at_points(current, source_points[None])[0]
            residual_target = source_points + (target_points - predicted)
            field = _gaussian_displacement(
                reference, source_points, residual_target, widths[index],
            )
            if gain_mode == "calibrated":
                assert calibration_source is not None and calibration_target is not None
                residual_calibration = calibration_target - q1_map_at_points(
                    current, calibration_source[None])[0]
                field_calibration = _gaussian_displacement_at_points(
                    calibration_source, source_points, residual_target,
                    widths[index])
                numerator = (residual_calibration * field_calibration).sum()
                denominator = field_calibration.square().sum() + 1e-8
                gain = (numerator / denominator).clamp(0, 1)
                if gain_trace is not None:
                    gain_trace.append(float(gain.detach().cpu()))
                desired = current + gain * field
            else:
                desired = current + field
        h = 1 / (side - 1)
        normalized = ((desired - current)[:, 1:-1, 1:-1] / (2 * h)).clamp(-.95, .95)
        logits = torch.atanh(normalized)
        current = SafeColoredQ1Relaxation(side).to(current.device)(current, logits)
    return current


def _match_rmse(mapped: torch.Tensor, points: torch.Tensor,
                targets: torch.Tensor) -> float:
    return float((q1_map_at_points(mapped, points[None]) - targets[None])
                 .square().sum(-1).mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--affine-map", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--final-side", type=int, default=257)
    parser.add_argument("--sigma", type=float, default=.12)
    parser.add_argument("--update-sigmas", type=float, nargs="*",
                        help="one positive Gaussian width for each moving level")
    parser.add_argument("--proposal-mode", choices=("absolute", "residual"),
                        default="absolute")
    parser.add_argument("--gain-mode", choices=("fixed", "calibrated"),
                        default="fixed")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    update_sigmas = (None if args.update_sigmas is None
                     else tuple(args.update_sigmas))
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    matches = json.loads(args.matches.read_text(encoding="utf-8"))
    if matches.get("status") != "ok" or matches.get("ransac_inliers", 0) < 16:
        raise ValueError("forward decoder requires at least 16 accepted matches")
    source_all = np.asarray(matches["source_points_unit"], dtype=np.float32)
    target_all = np.asarray(matches["target_points_unit"], dtype=np.float32)
    permutation = np.random.default_rng(290929).permutation(len(source_all))
    train_count = int(.8 * len(source_all))
    device = torch.device(args.device)
    source = torch.from_numpy(source_all[permutation[:train_count]].copy()).to(device)
    target = torch.from_numpy(target_all[permutation[:train_count]].copy()).to(device)
    source_val = torch.from_numpy(source_all[permutation[train_count:]].copy()).to(device)
    target_val = torch.from_numpy(target_all[permutation[train_count:]].copy()).to(device)
    extra = {"gain_mode": args.gain_mode,
             "calibration_source": source_val if args.gain_mode == "calibrated" else None,
             "calibration_target": target_val if args.gain_mode == "calibrated" else None}
    with np.load(args.affine_map) as archive:
        matrix = np.asarray(archive["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(archive["post_affine_offset"], dtype=np.float32)
    if matrix.shape != (2, 2) or offset.shape != (2,) or (
        not np.isfinite(matrix).all() or not np.isfinite(offset).all()
    ):
        raise ValueError("finite affine factors required")
    a, b, c, d = (Fraction.from_float(float(value)) for value in matrix.flat)
    if a * d - b * c <= 0:
        raise ValueError("positive affine determinant required")
    identity = identity_vertices(args.final_side, device=device)
    with torch.no_grad():
        for _ in range(3):
            forward_kernel_map(source, target, final_side=args.final_side,
                               sigma=args.sigma, update_sigmas=update_sigmas,
                               proposal_mode=args.proposal_mode, **extra)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    forward_times = []
    with torch.no_grad():
        for _ in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            mapped = forward_kernel_map(source, target, final_side=args.final_side,
                                        sigma=args.sigma, update_sigmas=update_sigmas,
                                        proposal_mode=args.proposal_mode, **extra)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_times.append(time.perf_counter() - started)
        val = _match_rmse(mapped, source_val, target_val)
        train = _match_rmse(mapped, source, target)
        zero_val = _match_rmse(identity, source_val, target_val)
        zero_train = _match_rmse(identity, source, target)
        validity = validate_q1_map(mapped, identity)
        saved = mapped.cpu().numpy()
    forward_peak = (None if device.type != "cuda" else
                    int(torch.cuda.max_memory_allocated(device)))
    vjp_times = []
    finite_vjp = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        src = source.detach().clone().requires_grad_(True)
        dst = target.detach().clone().requires_grad_(True)
        if args.gain_mode == "calibrated":
            src_cal = source_val.detach().clone().requires_grad_(True)
            dst_cal = target_val.detach().clone().requires_grad_(True)
            extra_vjp = {"gain_mode": "calibrated",
                         "calibration_source": src_cal,
                         "calibration_target": dst_cal}
        else:
            extra_vjp = extra
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = forward_kernel_map(src, dst, final_side=args.final_side,
                                    sigma=args.sigma, update_sigmas=update_sigmas,
                                    proposal_mode=args.proposal_mode, **extra_vjp)
        (output - identity).square().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        vjp_times.append(time.perf_counter() - started)
        finite_vjp &= bool(torch.isfinite(src.grad).all() and
                           torch.isfinite(dst.grad).all())
        if args.gain_mode == "calibrated":
            finite_vjp &= bool(torch.isfinite(src_cal.grad).all() and
                               torch.isfinite(dst_cal.grad).all())
    vjp_peak = (None if device.type != "cuda" else
                int(torch.cuda.max_memory_allocated(device)))
    gain_trace: list[float] = []
    if args.gain_mode == "calibrated":
        with torch.no_grad():
            forward_kernel_map(source, target, final_side=args.final_side,
                               sigma=args.sigma, update_sigmas=update_sigmas,
                               proposal_mode=args.proposal_mode, gain_trace=gain_trace,
                               **extra)
    reference = identity_vertices(args.final_side, device=torch.device("cpu")).numpy()
    np.savez_compressed(args.output_map, vertices=saved,
                        boundary_reference=reference,
                        post_affine_matrix=matrix, post_affine_offset=offset)
    certificate = certify_q1_binary_map(args.output_map)
    result = {
        "mode": "forward_gaussian_matches_to_safe_Q1_not_image_encoder",
        "proposal_mode": args.proposal_mode,
        "gain_mode": args.gain_mode,
        "calibrated_gain_by_level": gain_trace if args.gain_mode == "calibrated" else None,
        "calibration_matches_used_for_inference": args.gain_mode == "calibrated",
        "matches": str(args.matches),
        "fixed_affine_archive": str(args.affine_map),
        "saved_map": str(args.output_map),
        "control_side": args.final_side,
        "sigma_unit": args.sigma,
        "update_sides": [17, 33, 65, 129, 257][:len(update_sigmas)] if update_sigmas is not None else [17, 33, 65],
        "update_sigma_units": list(update_sigmas) if update_sigmas is not None else [args.sigma] * 3,
        "train_matches": train_count,
        "heldback_matches": (len(source_all) - train_count
                             if args.gain_mode == "fixed" else 0),
        "calibration_matches": (len(source_all) - train_count
                                if args.gain_mode == "calibrated" else 0),
        "split_seed": 290929,
        "zero_train_match_vector_rmse_unit": zero_train,
        "forward_train_match_vector_rmse_unit": train,
        "zero_heldback_match_vector_rmse_unit": zero_val if args.gain_mode == "fixed" else None,
        "forward_heldback_match_vector_rmse_unit": val if args.gain_mode == "fixed" else None,
        "identity_calibration_match_vector_rmse_unit": zero_val if args.gain_mode == "calibrated" else None,
        "forward_calibration_match_vector_rmse_unit": val if args.gain_mode == "calibrated" else None,
        "decoder_forward_seconds_median": statistics.median(forward_times),
        "decoder_forward_vjp_seconds_median": statistics.median(vjp_times),
        "finite_input_match_vjp": finite_vjp,
        "decoder_forward_peak_allocated_bytes": forward_peak,
        "decoder_vjp_peak_allocated_bytes": vjp_peak,
        "in_memory_nonpositive_corners": validity["nonpositive_corners"],
        "in_memory_boundary_max_error": validity["boundary_max_error"],
        "saved_binary_residual_valid": bool(certificate["valid"]),
        "saved_binary_nonpositive_corners": int(certificate["nonpositive_corners"]),
        "stored_affine_det_positive_exact": True,
        "saved_factorization_valid": bool(certificate["valid"]),
        "external_feature_extraction_seconds": matches.get(
            "total_decode_align_model_match_seconds"),
    }
    args.output_report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
