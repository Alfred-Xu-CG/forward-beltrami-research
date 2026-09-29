"""Oracle latent-fit diagnostic for one versus repeated full-vertex Q1 passes.

This deliberately optimizes latent arrays directly. It is not an image-to-latent
network and its result must not be interpreted as image-registration accuracy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveEllipsoidQ1Relaxation,
    AdaptiveSoftRadialQ1Relaxation,
    SafeColoredQ1Relaxation,
    q1_corner_determinants,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def run(side: int, steps: int, output: Path, device: str) -> list[dict]:
    if side < 5 or steps < 1 or output.exists():
        raise ValueError("side >= 5, steps >= 1 and a fresh output path required")
    torch.manual_seed(290929)
    axis = torch.linspace(0, 1, side, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    target = identity.clone()
    # This analytic target is a homeomorphism with unchanged outer boundary:
    # dT_x/dx >= 1-0.2*pi > 0. Its center displacement is O(0.2), while
    # one F1 color schedule near side=17 can move an interior point only O(h).
    target[..., 0] += 0.2 * torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
    if float(q1_corner_determinants(target).amin()) <= 0:
        raise AssertionError("the analytic target is not corner-positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for label, factory in (
        ("fixed_h_radial", SafeColoredQ1Relaxation),
        ("current_geometry_ellipsoid", AdaptiveEllipsoidQ1Relaxation),
        ("current_edge_soft_radial", AdaptiveSoftRadialQ1Relaxation),
    ):
        for rounds in (1, 4, 8):
            layer = factory(side).to(device)
            logits = torch.nn.Parameter(torch.zeros(
                rounds, 1, side - 2, side - 2, 2, device=device,
            ))
            optimizer = torch.optim.Adam([logits], lr=0.1)
            started = time.perf_counter()
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                mapped = identity
                for index in range(rounds):
                    mapped = layer(mapped, logits[index])
                loss = (mapped - target).square().mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                mapped = identity
                for index in range(rounds):
                    mapped = layer(mapped, logits[index])
                rmse = float(torch.sqrt((mapped - target).square().mean()))
                max_error = float((mapped - target).abs().amax())
                min_q = float(q1_corner_determinants(mapped).amin() * (side - 1) ** 2)
                saved = mapped.cpu().numpy().astype(np.float32)
            archive = output.with_name(f"{output.stem}_{label}_k{rounds}.npz")
            np.savez_compressed(
                archive, vertices=saved,
                boundary_reference=identity.cpu().numpy().astype(np.float32),
                post_affine_matrix=np.eye(2, dtype=np.float32),
                post_affine_offset=np.zeros(2, dtype=np.float32),
            )
            certificate = certify_q1_binary_map(archive)
            record = {
                "mode": label, "side": side, "rounds": rounds,
                "steps": steps, "optimizer": "Adam", "learning_rate": 0.1,
                "device": device, "vertex_coordinate_rmse": rmse,
                "maximum_absolute_coordinate_error": max_error,
                "minimum_normalized_q1_corner": min_q,
                "elapsed_seconds": time.perf_counter() - started,
                "certificate": certificate,
            }
            results.append(record)
            print(json.dumps(record), flush=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=17)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.side, args.steps, args.output, args.device)


if __name__ == "__main__":
    main()
