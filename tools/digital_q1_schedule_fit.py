"""Fit a declared F1/F2 Q1 schedule to one external affine-factored field.

This is a representation and cost ablation, not image-to-latent inference.
No landmarks are accepted. The saved best residual remains on one Q1 grid;
the teacher's positive affine is kept as a separate postcomposition factor.
"""

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
from tools.digital_q1_dhr_distill import _load_target_archive, identity_vertices


def fit_schedule(
    teacher: torch.Tensor, *, schedule: str, steps: int, learning_rate: float,
    device: str, seed_repeats: int = 1,
) -> tuple[torch.Tensor, dict]:
    if teacher.ndim != 4 or teacher.shape[0] != 1 or teacher.shape[-1] != 2 or (
        teacher.shape[1] != teacher.shape[2]
    ) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("one finite square teacher map required")
    if steps < 1 or learning_rate <= 0:
        raise ValueError("positive steps and learning rate required")
    target_device = torch.device(device)
    torch.manual_seed(290929)
    teacher = teacher.to(device=target_device, dtype=torch.float32)
    side = teacher.shape[1]
    model = ScheduledQ1Pyramid(
        17, side, schedule, seed_repeats=seed_repeats,
    ).to(target_device)
    stages = tuple(tuple(torch.nn.Parameter(torch.zeros(
        1, level - 2, level - 2, 2, device=target_device,
    )) for _ in range(count)) for level, count in zip(model.sides, model.passes_per_stage))
    latents = tuple(field for stage in stages for field in stage)
    optimizer = torch.optim.Adam(latents, lr=learning_rate)
    reference = identity_vertices(side, device=target_device)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    def sync() -> None:
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

    best_loss = float("inf")
    best_map = None
    initial_loss = None
    last_loss = None
    checkpoint_100 = None
    durations = []
    finite_steps = 0
    for index in range(steps):
        sync()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        mapped = model(stages)
        loss = (mapped[:, 1:-1, 1:-1] - teacher[:, 1:-1, 1:-1]).square().sum(-1).mean()
        loss.backward()
        finite = all(field.grad is not None and bool(torch.isfinite(field.grad).all())
                     for field in latents)
        finite_steps += int(finite)
        if not finite:
            raise FloatingPointError("nonfinite or missing latent gradient")
        value = float(loss.detach())
        initial_loss = value if initial_loss is None else initial_loss
        last_loss = value
        if index == 99:
            checkpoint_100 = value ** .5
        if value < best_loss:
            best_loss = value
            best_map = mapped.detach().clone()
        optimizer.step()
        sync()
        durations.append(time.perf_counter() - started)
    assert best_map is not None and initial_loss is not None and last_loss is not None
    validity = validate_q1_map(best_map, reference)
    return best_map, {
        "mode": "external_teacher_F1_F2_schedule_representation_ablation_not_G2",
        "schedule": schedule, "sides": list(model.sides),
        "seed_repeats": seed_repeats,
        "passes_per_stage": list(model.passes_per_stage),
        "latent_scalar_count": sum(field.numel() for field in latents),
        "structurally_active_latent_scalar_count": model.active_latent_scalar_count_per_sample(),
        "steps": steps, "learning_rate": learning_rate,
        "initial_interior_vector_rmse": initial_loss ** .5,
        "step100_interior_vector_rmse": checkpoint_100,
        "last_interior_vector_rmse": last_loss ** .5,
        "best_interior_vector_rmse": best_loss ** .5,
        "median_complete_step_seconds": statistics.median(durations),
        "total_measured_step_seconds": sum(durations),
        "finite_gradient_steps": finite_steps,
        "nonpositive_corners": validity["nonpositive_corners"],
        "boundary_max_error": validity["boundary_max_error"],
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=.04)
    parser.add_argument("--seed-repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    target, matrix, offset = _load_target_archive(args.target)
    if matrix is None or offset is None:
        raise ValueError("schedule comparison requires a positive-affine factored target")
    mapped, report = fit_schedule(
        target, schedule=args.schedule, steps=args.steps,
        learning_rate=args.learning_rate, device=args.device,
        seed_repeats=args.seed_repeats,
    )
    reference = identity_vertices(mapped.shape[1], device=torch.device("cpu"))
    np.savez_compressed(
        args.output_map, vertices=mapped.cpu().numpy(),
        boundary_reference=reference.numpy(),
        post_affine_matrix=matrix, post_affine_offset=offset,
    )
    certificate = certify_q1_binary_map(args.output_map)
    report.update({
        "target": str(args.target), "saved_map": str(args.output_map),
        "saved_binary_residual_valid": certificate["valid"],
        "saved_binary_nonpositive_corners": certificate["nonpositive_corners"],
        "post_affine_det": float(np.linalg.det(matrix.astype(np.float64))),
    })
    if args.output_report is not None:
        args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
