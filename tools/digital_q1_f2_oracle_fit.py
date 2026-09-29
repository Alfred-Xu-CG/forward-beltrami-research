"""Oracle-latent target fit for repeated fixed-h/current-edge F2 patch layers.

The target coordinates are directly visible to Adam. This measures a local
parameterization/optimization bottleneck, not an image-to-latent network.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    StaggeredPatchQ1Layer, q1_corner_determinants,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_q1_f2_current_edge_benchmark import _map


def run(side: int, rounds: int, patch_cells: int, steps: int,
        learning_rate: float,
        device: str, prefix: Path) -> dict:
    if steps < 1 or rounds < 1 or not 0 < learning_rate <= .1:
        raise ValueError("steps/rounds positive and learning_rate in (0,.1] required")
    target_device = torch.device(device)
    base = _map(side, target_device, .1)
    target = _map(side, target_device, .07)
    reference = _map(side, target_device, 0.)
    report = {
        "question": "can oracle full-field logits fit an analytic target using repeated F2",
        "control_vertices": side * side, "q1_cells": (side - 1) ** 2,
        "rounds": rounds, "patch_cells": patch_cells, "steps": steps,
        "batch": 1, "dtype": "float32", "device": device,
        "raw_span": .5, "learning_rate": learning_rate,
        "source": "x+0.1*sin(2*pi*x)*sin(pi*y), y",
        "target": "x+0.07*sin(2*pi*x)*sin(pi*y), y",
        "objective": "mean of squared two-coordinate error at every control vertex",
        "arms": {},
    }
    for mode in ("fixed_h", "current_edge"):
        layer = StaggeredPatchQ1Layer(
            side, patch_cells=patch_cells, proposal_mode=mode, raw_span=.5,
        ).to(target_device)
        latent = torch.nn.Parameter(torch.zeros(
            rounds, 4, 1, side - 2, side - 2, 2, device=target_device,
        ))
        optimizer = torch.optim.Adam([latent], lr=learning_rate)
        durations, checkpoints = [], []
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)
        for step in range(steps):
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            mapped = base
            for round_index in range(rounds):
                mapped = layer(mapped, tuple(latent[round_index]))
            loss = (mapped - target).square().sum(-1).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite F2 oracle loss at {step}")
            loss.backward()
            if latent.grad is None or not bool(torch.isfinite(latent.grad).all()):
                raise FloatingPointError(f"nonfinite F2 oracle VJP at {step}")
            optimizer.step()
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            durations.append(time.perf_counter() - started)
            if step in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
                checkpoints.append({"step": step + 1,
                                    "coordinate_rmse": float((loss.detach() / 2).sqrt())})
        with torch.no_grad():
            mapped = base
            for round_index in range(rounds):
                mapped = layer(mapped, tuple(latent[round_index]))
        archive = prefix.with_name(prefix.name + f"_{mode}.npz")
        np.savez_compressed(archive,
                            vertices=mapped.cpu().numpy().astype(np.float32),
                            boundary_reference=reference.cpu().numpy().astype(np.float32),
                            latent=latent.detach().cpu().numpy().astype(np.float32))
        certificate = certify_q1_binary_map(archive)
        if not certificate["valid"]:
            raise ArithmeticError(f"invalid oracle F2 output for {mode}")
        report["arms"][mode] = {
            "initial_coordinate_rmse": float((base - target).square().mean().sqrt()),
            "final_coordinate_rmse": float((mapped - target).square().mean().sqrt()),
            "trace": checkpoints,
            "median_training_step_seconds": statistics.median(durations),
            "peak_torch_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(target_device)
                if target_device.type == "cuda" else None),
            "minimum_normalized_q1_corner": float(
                q1_corner_determinants(mapped).amin() * (side - 1) ** 2),
            "certificate": certificate, "saved_map": archive.name,
        }
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=.02)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.side, args.rounds, args.patch_cells,
                         args.steps, args.learning_rate,
                         args.device, args.prefix), indent=2))


if __name__ == "__main__":
    main()
