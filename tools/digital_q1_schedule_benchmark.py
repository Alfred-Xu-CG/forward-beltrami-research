"""Measure full F1/F2 Q1 schedule forward and first-order VJP at dense control scale."""

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
from qcopt.neural_bijection.dense.scheduled_q1_pyramid import ScheduledQ1Pyramid


def benchmark_schedule(
    *, schedule: str, seed_side: int = 17, final_side: int = 257,
    batch: int = 1, device: str = "cuda:0", repeats: int = 10,
    output_map: Path | None = None, seed_repeats: int = 1,
) -> dict:
    if batch < 1 or repeats < 1:
        raise ValueError("positive batch and repeats required")
    torch.manual_seed(290929)
    target_device = torch.device(device)
    model = ScheduledQ1Pyramid(
        seed_side, final_side, schedule, seed_repeats=seed_repeats,
    ).to(target_device)
    stages = tuple(tuple(torch.nn.Parameter(
        .3 * torch.randn(batch, side - 2, side - 2, 2, device=target_device)
    ) for _ in range(count)) for side, count in zip(model.sides, model.passes_per_stage))
    latents = tuple(field for stage in stages for field in stage)
    axis = torch.arange(final_side, device=target_device) / (final_side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    reference = torch.stack((xx, yy), dim=-1)[None].expand(batch, -1, -1, -1)

    def sync() -> None:
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

    for _ in range(2):
        for field in latents:
            field.grad = None
        output = model(stages)
        (output - reference).square().mean().backward()
    sync()
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    inference_times = []
    with torch.no_grad():
        for _ in range(repeats):
            sync()
            started = time.perf_counter()
            output = model(stages)
            sync()
            inference_times.append(time.perf_counter() - started)
    complete_times = []
    for _ in range(repeats):
        for field in latents:
            field.grad = None
        sync()
        started = time.perf_counter()
        output = model(stages)
        (output - reference).square().mean().backward()
        sync()
        complete_times.append(time.perf_counter() - started)
    gradients_finite = all(
        field.grad is not None and bool(torch.isfinite(field.grad).all())
        for field in latents
    )
    with torch.no_grad():
        mapped = model(stages)
    validity = validate_q1_map(mapped, reference)
    certificate = None
    if output_map is not None:
        np.savez_compressed(
            output_map, vertices=mapped.cpu().numpy(),
            boundary_reference=reference.cpu().numpy(),
        )
        certificate = certify_q1_binary_map(output_map)
    return {
        "schedule": schedule, "sides": list(model.sides),
        "seed_repeats": seed_repeats,
        "passes_per_stage": list(model.passes_per_stage),
        "latent_scalar_count": sum(field.numel() for field in latents),
        "structurally_active_latent_scalar_count": (
            batch * model.active_latent_scalar_count_per_sample()
        ),
        "batch": batch, "device": str(target_device), "dtype": "float32",
        "repeats_after_two_warmups": repeats,
        "inference_forward_seconds_median": statistics.median(inference_times),
        "complete_forward_vjp_seconds_median": statistics.median(complete_times),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda" else None
        ),
        "all_latent_gradients_finite": gradients_finite,
        "tensor_nonpositive_corners": validity["nonpositive_corners"],
        "saved_binary_valid": None if certificate is None else certificate["valid"],
        "saved_binary_nonpositive_corners": None if certificate is None else certificate["nonpositive_corners"],
        "output_map": None if output_map is None else str(output_map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--seed-side", type=int, default=17)
    parser.add_argument("--final-side", type=int, default=257)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-repeats", type=int, default=1)
    parser.add_argument("--output-map", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    report = benchmark_schedule(
        schedule=args.schedule, seed_side=args.seed_side, final_side=args.final_side,
        batch=args.batch, device=args.device, repeats=args.repeats,
        output_map=args.output_map, seed_repeats=args.seed_repeats,
    )
    if args.output_report is not None:
        args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
