"""Predeclared high-frequency test of genuine fine-level safe kernel proposals."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_forward_kernel import forward_kernel_map
from tools.digital_q1_match_fit import q1_map_at_points


WIDTHS = (.12, .06, .03, .015, .01)


def analytic_map(points: torch.Tensor, amplitude: float = .02) -> torch.Tensor:
    x, y = points[..., 0], points[..., 1]
    bump = amplitude * torch.sin(8 * torch.pi * x) * torch.sin(8 * torch.pi * y)
    bump = torch.where((x == 0) | (x == 1) | (y == 0) | (y == 1),
                       torch.zeros_like(bump), bump)
    return torch.stack((x + bump, y), dim=-1)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    source: torch.Tensor, target: torch.Tensor, reference: torch.Tensor,
    truth: torch.Tensor, queries: torch.Tensor, *,
    widths: tuple[float, ...] | None, repeats: int, proposal_mode: str,
    gain_mode: str, calibration_source: torch.Tensor | None,
    calibration_target: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    device = source.device
    def decode(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None,
               d: torch.Tensor | None) -> torch.Tensor:
        return forward_kernel_map(a, b, final_side=257, sigma=.12,
                                  update_sigmas=widths,
                                  proposal_mode=proposal_mode, gain_mode=gain_mode,
                                  calibration_source=c, calibration_target=d)
    with torch.no_grad():
        for _ in range(2):
            decode(source, target, calibration_source, calibration_target)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times = []
    with torch.no_grad():
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            mapped = decode(source, target, calibration_source, calibration_target)
            _sync(device)
            forward_times.append(time.perf_counter() - started)
        forward_peak = (torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda" else None)
        vertex_rmse = float((mapped - truth).square().sum(-1).mean().sqrt())
        predicted = q1_map_at_points(mapped, queries[None])[0]
        query_truth = analytic_map(queries)
        query_rmse = float((predicted - query_truth).square().sum(-1).mean().sqrt())
        validity = validate_q1_map(mapped, reference)
        saved = mapped.cpu().numpy()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vjp_times = []
    finite = True
    for _ in range(repeats):
        a = source.detach().clone().requires_grad_(True)
        b = target.detach().clone().requires_grad_(True)
        c = (None if calibration_source is None else
             calibration_source.detach().clone().requires_grad_(True))
        d = (None if calibration_target is None else
             calibration_target.detach().clone().requires_grad_(True))
        _sync(device)
        started = time.perf_counter()
        result = decode(a, b, c, d)
        (result - truth).square().mean().backward()
        _sync(device)
        vjp_times.append(time.perf_counter() - started)
        finite &= bool(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())
        if gain_mode == "calibrated":
            finite &= bool(torch.isfinite(c.grad).all() and torch.isfinite(d.grad).all())
    vjp_peak = (torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else None)
    gains: list[float] = []
    if gain_mode == "calibrated":
        with torch.no_grad():
            forward_kernel_map(source, target, final_side=257, sigma=.12,
                               update_sigmas=widths, proposal_mode=proposal_mode,
                               gain_mode=gain_mode,
                               calibration_source=calibration_source,
                               calibration_target=calibration_target,
                               gain_trace=gains)
    return saved, {
        "calibrated_gain_by_level": gains if gain_mode == "calibrated" else None,
        "update_sigma_units": list(widths) if widths is not None else [.12] * 3,
        "update_sides": [17, 33, 65, 129, 257][:len(widths)] if widths is not None else [17, 33, 65],
        "vertex_vector_rmse_unit": vertex_rmse,
        "offgrid_query_vector_rmse_unit": query_rmse,
        "in_memory_nonpositive_corners": validity["nonpositive_corners"],
        "in_memory_boundary_max_error": validity["boundary_max_error"],
        "finite_match_coordinate_vjp": finite,
        "forward_seconds_median": statistics.median(forward_times),
        "forward_vjp_seconds_median": statistics.median(vjp_times),
        "forward_peak_allocated_bytes": forward_peak,
        "vjp_peak_allocated_bytes": vjp_peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--proposal-mode", choices=("absolute", "residual"),
                        default="absolute")
    parser.add_argument("--gain-mode", choices=("fixed", "calibrated"),
                        default="fixed")
    parser.add_argument("--split-matches", action="store_true",
                        help="use deterministic 80/20 proposal/calibration split")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("positive repeats required")
    if args.gain_mode == "calibrated" and (
        args.proposal_mode != "residual" or not args.split_matches
    ):
        raise ValueError("calibrated gain requires residual proposal and split")
    device = torch.device(args.device)
    reference = identity_vertices(257, device=device)
    truth = analytic_map(reference)
    source_axis = (torch.arange(16, device=device, dtype=torch.float32) + .5) / 16
    sy, sx = torch.meshgrid(source_axis, source_axis, indexing="ij")
    source = torch.stack((sx.ravel(), sy.ravel()), dim=-1)
    target = analytic_map(source)
    if args.split_matches:
        permutation = torch.from_numpy(
            np.random.default_rng(290929).permutation(len(source))).to(device)
        train_count = int(.8 * len(source))
        calibration_source = source[permutation[train_count:]]
        calibration_target = target[permutation[train_count:]]
        source = source[permutation[:train_count]]
        target = target[permutation[:train_count]]
    else:
        calibration_source = calibration_target = None
    generator = torch.Generator(device="cpu").manual_seed(290929)
    queries = torch.rand((10000, 2), generator=generator).to(device)
    target_validity = validate_q1_map(truth, reference)
    if target_validity["nonpositive_corners"] or target_validity["boundary_max_error"]:
        raise ValueError("analytic target sampled Q1 map is not positive")
    report = {
        "mode": "synthetic_dense_detail_no_image_network",
        "proposal_mode": args.proposal_mode,
        "gain_mode": args.gain_mode,
        "split_matches": args.split_matches,
        "device": str(device),
        "control_side": 257,
        "control_vertices": 257 * 257,
        "target_formula": "(x+.02*sin(8*pi*x)*sin(8*pi*y),y)",
        "analytic_continuum_determinant_lower_bound": 1 - 8 * np.pi * .02,
        "proposal_match_count": len(source),
        "calibration_match_count": (0 if calibration_source is None else
                                    len(calibration_source)),
        "match_locations": "16x16 offset centers (i+0.5)/16",
        "offgrid_query_count": len(queries),
        "query_seed": 290929,
        "target_sample_nonpositive_Q1_corners": target_validity["nonpositive_corners"],
        "repeats_after_two_warmups": args.repeats,
        "cases": {},
    }
    reference_cpu = reference.cpu().numpy()
    for name, widths in (("coarse_65", None),
                         ("matched_coarse_65", WIDTHS[:3]),
                         ("multiband_257", WIDTHS)):
        saved, result = _measure(source, target, reference, truth, queries,
                                 widths=widths, repeats=args.repeats,
                                 proposal_mode=args.proposal_mode,
                                 gain_mode=args.gain_mode,
                                 calibration_source=calibration_source,
                                 calibration_target=calibration_target)
        output = args.output_prefix.with_name(args.output_prefix.name + "_" + name + ".npz")
        np.savez_compressed(output, vertices=saved, boundary_reference=reference_cpu,
                            post_affine_matrix=np.eye(2, dtype=np.float32),
                            post_affine_offset=np.zeros(2, dtype=np.float32))
        certificate = certify_q1_binary_map(output)
        result["saved_map"] = str(output)
        result["saved_binary_residual_valid"] = certificate["valid"]
        result["saved_binary_nonpositive_corners"] = certificate["nonpositive_corners"]
        report["cases"][name] = result
    report_path = args.output_prefix.with_name(args.output_prefix.name + "_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
