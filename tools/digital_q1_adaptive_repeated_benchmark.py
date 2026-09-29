"""Compare repeated fixed-span and current-geometry Q1 F1 passes at 257²."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveEllipsoidQ1Relaxation, AdaptiveSoftRadialQ1Relaxation,
    SafeColoredQ1Relaxation,
    q1_corner_determinants,
)


def benchmark(side: int, rounds: int, latent_scale: float, device: str,
              repeats: int, output: Path) -> list[dict]:
    if side < 3 or rounds < 1 or repeats < 1 or latent_scale <= 0:
        raise ValueError("positive side, rounds, repeats and latent scale required")
    if output.exists():
        raise FileExistsError(output)
    target_device = torch.device(device)
    axis = torch.linspace(0, 1, side, device=target_device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    base = identity.clone()
    base[..., 0] += .14 * torch.sin(2 * torch.pi * xx)
    target = identity.clone()
    target[..., 0] += .07 * torch.sin(2 * torch.pi * xx) * torch.sin(torch.pi * yy)
    if float(q1_corner_determinants(base).amin()) <= .05 / (side - 1) ** 2:
        raise AssertionError("distorted initial map is below the stated floor")
    torch.manual_seed(290929)
    initial_logits = latent_scale * torch.randn(
        rounds, 1, side - 2, side - 2, 2, device=target_device,
    )
    results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for label, layer in (
        ("fixed_h_radial", SafeColoredQ1Relaxation(side)),
        ("current_geometry_ellipsoid", AdaptiveEllipsoidQ1Relaxation(side)),
        ("current_edge_soft_radial", AdaptiveSoftRadialQ1Relaxation(side)),
    ):
        layer = layer.to(target_device)
        forward_times, full_times, memories = [], [], []
        saved = None
        clip_counts = []
        for iteration in range(repeats + 1):
            logits = initial_logits.detach().clone().requires_grad_(True)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
                torch.cuda.reset_peak_memory_stats(target_device)
            started = time.perf_counter()
            mapped = base
            for round_index in range(rounds):
                previous = mapped
                mapped = layer(mapped, logits[round_index])
                if label == "fixed_h_radial" and iteration == repeats:
                    raw = 2 / (side - 1) * torch.tanh(logits[round_index])
                    accepted = mapped[:, 1:-1, 1:-1] - previous[:, 1:-1, 1:-1]
                    raw_norm = torch.linalg.vector_norm(raw, dim=-1)
                    ratio = torch.linalg.vector_norm(accepted, dim=-1) / raw_norm.clamp_min(1e-12)
                    clip_counts.append(float((ratio < .999).float().mean().detach()))
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            after_forward = time.perf_counter()
            loss = (mapped - target).square().mean()
            loss.backward()
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            after_backward = time.perf_counter()
            if iteration:
                forward_times.append(after_forward - started)
                full_times.append(after_backward - started)
                if target_device.type == "cuda":
                    memories.append(torch.cuda.max_memory_allocated(target_device))
            if iteration == repeats:
                corners = q1_corner_determinants(mapped)
                saved = mapped.detach().cpu().numpy().astype(np.float32)
                record = {
                    "mode": label, "side": side, "rounds": rounds,
                    "latent_scale": latent_scale, "device": str(target_device),
                    "repeats_after_warmup": repeats,
                    "median_forward_seconds": statistics.median(forward_times),
                    "median_forward_vjp_seconds": statistics.median(full_times),
                    "peak_torch_allocated_bytes": max(memories) if memories else None,
                    "minimum_normalized_q1_corner": float(corners.amin() * (side - 1) ** 2),
                    "mean_vertex_motion": float(torch.linalg.vector_norm(
                        mapped - base, dim=-1).mean()),
                    "gradient_finite": bool(torch.isfinite(logits.grad).all()),
                    "mean_fixed_radial_clip_fraction_by_round": (
                        clip_counts if clip_counts else None),
                }
        assert saved is not None
        np.savez_compressed(
            output.with_name(output.stem + "_" + label + ".npz"),
            vertices=saved,
            boundary_reference=base.detach().cpu().numpy().astype(np.float32),
            post_affine_matrix=np.eye(2, dtype=np.float32),
            post_affine_offset=np.zeros(2, dtype=np.float32),
        )
        results.append(record)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--latent-scale", type=float, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.side, args.rounds, args.latent_scale,
                               args.device, args.repeats, args.output), indent=2))


if __name__ == "__main__":
    main()
