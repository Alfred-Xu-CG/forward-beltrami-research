"""Matched fixed-h/current-edge repeated F2 geometry benchmark.

This probes proposal scale and cost, not image-conditioned registration.
The saved archive is independently certifiable in its actual float32 values.
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


def _map(side: int, device: torch.device, amplitude: float) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    wave = torch.sin(2 * torch.pi * xx) * torch.sin(torch.pi * yy)
    wave = torch.where((xx > 0) & (xx < 1) & (yy > 0) & (yy < 1),
                       wave, torch.zeros_like(wave))
    return torch.stack((xx + amplitude * wave, yy), -1)[None]


def run(side: int, rounds: int, patch_cells: int, device: str,
        repeats: int, prefix: Path) -> dict:
    if side < 9 or rounds < 1 or repeats < 1 or (side - 1) % patch_cells:
        raise ValueError("side, rounds, repeats and patch compatibility required")
    target_device = torch.device(device)
    reference = _map(side, target_device, 0.0)
    base_values = _map(side, target_device, 0.1)
    target = _map(side, target_device, 0.07)
    torch.manual_seed(290930)
    shared = 2 * torch.randn(rounds, 4, 1, side - 2, side - 2, 2,
                             device=target_device)
    report = {
        "question": "current-edge versus fixed-h repeated F2 geometry cost and displacement",
        "control_vertices": side * side,
        "q1_cells": (side - 1) ** 2,
        "q1_corners": 4 * (side - 1) ** 2,
        "rounds": rounds, "patch_cells": patch_cells,
        "batch": 1, "dtype": "float32", "device": device,
        "raw_span": 0.5, "latent_seed": 290930,
        "latent_distribution": "independent Gaussian standard deviation 2",
        "source": "x+0.1*sin(2*pi*x)*sin(pi*y), y",
        "target": "x+0.07*sin(2*pi*x)*sin(pi*y), y",
        "warmup": 1, "timed_repeats": repeats, "arms": {},
    }

    for mode in ("fixed_h", "current_edge"):
        layer = StaggeredPatchQ1Layer(side, patch_cells=patch_cells,
                                      proposal_mode=mode, raw_span=.5).to(target_device)

        def evaluate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            base = base_values.detach().clone().requires_grad_()
            latents = shared.detach().clone().requires_grad_()
            mapped = base
            for round_index in range(rounds):
                mapped = layer(mapped, tuple(latents[round_index]))
            loss = (mapped - target).square().sum(-1).mean()
            grad_base, grad_latents = torch.autograd.grad(loss, (base, latents))
            return mapped, grad_base, grad_latents

        evaluate()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
            torch.cuda.reset_peak_memory_stats(target_device)
        forward_times, total_times = [], []
        last = None
        for _ in range(repeats):
            base = base_values.detach().clone().requires_grad_()
            latents = shared.detach().clone().requires_grad_()
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            start = time.perf_counter()
            mapped = base
            for round_index in range(rounds):
                mapped = layer(mapped, tuple(latents[round_index]))
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            mid = time.perf_counter()
            loss = (mapped - target).square().sum(-1).mean()
            grad_base, grad_latents = torch.autograd.grad(loss, (base, latents))
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            end = time.perf_counter()
            forward_times.append(mid - start)
            total_times.append(end - start)
            last = (mapped.detach(), grad_base.detach(), grad_latents.detach())
        assert last is not None
        mapped, grad_base, grad_latents = last
        if not bool(torch.isfinite(grad_base).all() and torch.isfinite(grad_latents).all()):
            raise FloatingPointError("nonfinite F2 VJP")
        archive = prefix.with_name(prefix.name + f"_{mode}.npz")
        np.savez_compressed(archive,
                            vertices=mapped.cpu().numpy().astype(np.float32),
                            boundary_reference=reference.cpu().numpy().astype(np.float32),
                            latent_vjp=grad_latents.cpu().numpy().astype(np.float32))
        certificate = certify_q1_binary_map(archive)
        if not certificate["valid"]:
            raise ArithmeticError("saved F2 map invalid")
        report["arms"][mode] = {
            "median_forward_seconds": statistics.median(forward_times),
            "median_forward_plus_vjp_seconds": statistics.median(total_times),
            "peak_torch_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(target_device)
                if target_device.type == "cuda" else None
            ),
            "mean_vertex_displacement_from_base": float((
                mapped - base_values).norm(dim=-1).mean()),
            "coordinate_rmse_to_target": float((mapped - target).square().mean().sqrt()),
            "minimum_normalized_q1_corner": float(
                q1_corner_determinants(mapped).amin() * (side - 1) ** 2),
            "finite_base_and_latent_vjp": True,
            "certificate": certificate,
            "saved_map": archive.name,
        }
    output = prefix.with_suffix(".json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.side, args.rounds, args.patch_cells,
                         args.device, args.repeats, args.prefix), indent=2))


if __name__ == "__main__":
    main()
