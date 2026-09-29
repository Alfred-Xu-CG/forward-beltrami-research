"""Matched analytic-target oracle fit for repeated current-edge F1 updates.

Target vertices directly train free logits. No image or CNN is involved.
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
    AdaptiveSoftRadialQ1Relaxation, q1_corner_determinants,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_q1_f2_current_edge_benchmark import _map


def run(side: int, rounds: int, steps: int, device: str, prefix: Path) -> dict:
    target_device = torch.device(device)
    base = _map(side, target_device, .1)
    target = _map(side, target_device, .07)
    reference = _map(side, target_device, 0.)
    update = AdaptiveSoftRadialQ1Relaxation(side).to(target_device)
    logits = torch.nn.Parameter(torch.zeros(
        rounds, 1, side - 2, side - 2, 2, device=target_device,
    ))
    optimizer = torch.optim.Adam([logits], lr=.02)
    durations, trace = [], []
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    for step in range(steps):
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        mapped = base
        for round_index in range(rounds):
            mapped = update(mapped, logits[round_index])
        loss = (mapped - target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite F1 oracle loss at {step}")
        loss.backward()
        if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
            raise FloatingPointError(f"nonfinite F1 oracle gradient at {step}")
        optimizer.step()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        durations.append(time.perf_counter() - started)
        if step in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
            trace.append({"step": step + 1, "coordinate_rmse": float(loss.detach().sqrt())})
    with torch.no_grad():
        mapped = base
        for round_index in range(rounds):
            mapped = update(mapped, logits[round_index])
    archive = prefix.with_suffix(".npz")
    np.savez_compressed(archive,
                        vertices=mapped.cpu().numpy().astype(np.float32),
                        boundary_reference=reference.cpu().numpy().astype(np.float32),
                        latent=logits.detach().cpu().numpy().astype(np.float32))
    certificate = certify_q1_binary_map(archive)
    if not certificate["valid"]:
        raise ArithmeticError("invalid saved F1 oracle output")
    report = {
        "question": "matched F1 oracle capacity against repeated F2 analytic target",
        "control_vertices": side * side, "rounds": rounds, "steps": steps,
        "batch": 1, "dtype": "float32", "device": device,
        "raw_span": 8, "learning_rate": .02,
        "source": "x+0.1*sin(2*pi*x)*sin(pi*y), y",
        "target": "x+0.07*sin(2*pi*x)*sin(pi*y), y",
        "initial_coordinate_rmse": float((base - target).square().mean().sqrt()),
        "final_coordinate_rmse": float((mapped - target).square().mean().sqrt()),
        "trace": trace,
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
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.side, args.rounds, args.steps,
                         args.device, args.prefix), indent=2))


if __name__ == "__main__":
    main()
