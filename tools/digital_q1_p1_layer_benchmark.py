"""Batch and 512²-query cost of the factorized 257² safe P1 geometry layer."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_p1_layer import (
    SafeSparseMatchP1Layer, evaluate_factorized_p1,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _inputs(batch: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(290929)
    source = (.08 + .84 * torch.rand((batch, 96, 2), generator=generator)).to(device)
    phase = torch.sin(torch.pi * source[..., 0]) * torch.sin(torch.pi * source[..., 1])
    amplitudes = torch.tensor([.018, -.014], device=device)[:batch]
    motion = torch.stack((amplitudes[:, None] * phase,
                          -.55 * amplitudes[:, None] * phase), dim=-1)
    target = source + motion
    matrix = torch.tensor([[[1.02, .03], [-.01, .97]],
                           [[.98, -.02], [.01, 1.01]]],
                          device=device)[:batch]
    offset = torch.tensor([[.01, -.02], [-.005, .014]], device=device)[:batch]
    axis = (torch.arange(512, device=device) + .5) / 512
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    queries = torch.stack((x.flatten(), y.flatten()), dim=-1)[None].expand(
        batch, -1, -1)
    return source, target, matrix, offset, queries


def _measure(batch: int, device: torch.device, repeats: int,
             output_prefix: Path) -> dict:
    source, target, matrix, offset, queries = _inputs(batch, device)
    layer = SafeSparseMatchP1Layer(side=257,
                                   update_sigmas=(.12, .06, .03),
                                   proposal_mode="residual").to(device)
    with torch.no_grad():
        for _ in range(2):
            mapped = layer(source, target, post_affine_matrix=matrix,
                           post_affine_offset=offset)
            evaluate_factorized_p1(mapped, queries)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_times = []
    with torch.no_grad():
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            mapped = layer(source, target, post_affine_matrix=matrix,
                           post_affine_offset=offset)
            values = evaluate_factorized_p1(mapped, queries)
            _sync(device)
            forward_times.append(time.perf_counter() - started)
        saved = mapped.residual_vertices.cpu().numpy()
        forward_finite = bool(torch.isfinite(values).all())
    forward_peak = (None if device.type != "cuda" else
                    int(torch.cuda.max_memory_allocated(device)))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vjp_times = []
    finite_vjp = True
    for _ in range(repeats):
        a = source.detach().clone().requires_grad_(True)
        b = target.detach().clone().requires_grad_(True)
        m = matrix.detach().clone().requires_grad_(True)
        t = offset.detach().clone().requires_grad_(True)
        _sync(device)
        started = time.perf_counter()
        mapped = layer(a, b, post_affine_matrix=m, post_affine_offset=t)
        values = evaluate_factorized_p1(mapped, queries)
        values.square().mean().backward()
        _sync(device)
        vjp_times.append(time.perf_counter() - started)
        finite_vjp &= all(bool(torch.isfinite(item.grad).all())
                          for item in (a, b, m, t))
    vjp_peak = (None if device.type != "cuda" else
                int(torch.cuda.max_memory_allocated(device)))
    reference = identity_vertices(257, device=torch.device("cpu")).numpy()
    output = output_prefix.with_name(output_prefix.name + f"_batch{batch}_map.npz")
    np.savez_compressed(output, vertices=saved,
                        boundary_reference=np.repeat(reference, batch, axis=0),
                        post_affine_matrix=matrix.cpu().numpy(),
                        post_affine_offset=offset.cpu().numpy(),
                        interpolation="P1 SW-NE fixed diagonal, positive affine postcomposition")
    certificate = certify_q1_binary_map(output)
    affine_positive_exact = []
    for item in matrix.cpu().numpy():
        aa, bb, cc, dd = (Fraction.from_float(float(value)) for value in item.flat)
        affine_positive_exact.append(bool(aa * dd - bb * cc > 0))
    return {
        "batch": batch,
        "control_side": 257,
        "control_vertices_per_sample": 257 * 257,
        "p1_triangles_per_sample": 2 * 256 * 256,
        "query_count_per_sample": 512 * 512,
        "latent_matches_per_sample": 96,
        "saved_map": str(output),
        "saved_binary_residual_valid": certificate["valid"],
        "saved_binary_nonpositive_corners": certificate["nonpositive_corners"],
        "saved_affine_positive_exact_by_sample": affine_positive_exact,
        "forward_finite": forward_finite,
        "finite_match_and_affine_vjp": finite_vjp,
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
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("positive repeats required")
    device = torch.device(args.device)
    report = {
        "mode": "safe_P1_geometry_layer_no_image_encoder_or_intensity_sampling",
        "precision": "float32",
        "device": str(device),
        "repeats_after_two_warmups": args.repeats,
        "cases": [_measure(batch, device, args.repeats, args.output_prefix)
                  for batch in (1, 2)],
    }
    report_path = args.output_prefix.with_name(args.output_prefix.name + "_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
