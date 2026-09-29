"""Construct exact-arithmetic waypoint latents for current-edge soft F1.

This exposes the analytic inverse of each single-color local update for one
known analytic target. It is not a learned image-to-latent inference method.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveSoftRadialQ1Relaxation, q1_corner_determinants,
)
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def _map(side: int, amplitude: float, device: torch.device) -> torch.Tensor:
    axis = torch.arange(side, device=device, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    wave = torch.sin(2 * torch.pi * xx) * torch.sin(torch.pi * yy)
    wave = torch.where((xx > 0) & (xx < 1) & (yy > 0) & (yy < 1),
                       wave, torch.zeros_like(wave))
    return torch.stack((xx + amplitude * wave, yy), -1)[None]


def _cross(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def attempt(side: int, rounds: int, device: torch.device) -> dict:
    base, target = _map(side, .1, device), _map(side, .07, device)
    layer = AdaptiveSoftRadialQ1Relaxation(side).to(device)
    floor = base.new_tensor([.05 / (side - 1) ** 2])
    current = base.reshape(1, side * side, 2)
    latent_rounds = []
    max_adverse, max_latent_coordinate = 0., 0.
    start_time = time.perf_counter()
    for round_index in range(rounds):
        waypoint = base + (round_index + 1) / rounds * (target - base)
        desired_table = waypoint.reshape_as(current)
        round_latent = base.new_zeros((1, (side - 2) ** 2, 2))
        for color in range(4):
            vertices = getattr(layer, f"_vertices_{color}")
            opposite = vertices[:, None, None] + layer._opposite_offsets[None]
            local = ((vertices // side - 1) * (side - 2) + vertices % side - 1)
            point = current[:, vertices]
            desired = desired_table[:, vertices] - point
            first = current[:, opposite[..., 0]]
            second = current[:, opposite[..., 1]]
            edge = second - first
            areas = _cross(edge, point[:, :, None] - first)
            budget = torch.minimum(.75 * areas, areas - floor[:, None, None])
            if not bool((budget > 0).all()):
                return {"rounds": rounds, "success": False,
                        "reason": "nonpositive local floor slack",
                        "failed_round": round_index + 1, "failed_color": color}
            adverse = -_cross(edge, desired[:, :, None]) / budget
            maximum = adverse.clamp_min(0).amax(dim=-1)
            max_adverse = max(max_adverse, float(maximum.amax()))
            if not bool((maximum < 1).all()):
                return {"rounds": rounds, "success": False,
                        "reason": "desired displacement exceeds local radial budget",
                        "failed_round": round_index + 1, "failed_color": color,
                        "maximum_adverse_ratio": max_adverse}
            raw = desired / (1 - maximum[..., None])
            horizontal = (current[:, vertices + 1] - current[:, vertices - 1]) / 2
            vertical = (current[:, vertices + side] - current[:, vertices - side]) / 2
            determinant = _cross(horizontal, vertical)
            if not bool((determinant > 0).all()):
                return {"rounds": rounds, "success": False,
                        "reason": "centered edge matrix singular",
                        "failed_round": round_index + 1, "failed_color": color}
            ux = _cross(raw, vertical) / determinant / layer.raw_span
            uy = _cross(horizontal, raw) / determinant / layer.raw_span
            coordinate = torch.stack((ux, uy), -1)
            max_latent_coordinate = max(max_latent_coordinate, float(coordinate.abs().amax()))
            if not bool((coordinate.abs() < 1).all()):
                return {"rounds": rounds, "success": False,
                        "reason": "raw proposal exceeds tanh range",
                        "failed_round": round_index + 1, "failed_color": color,
                        "maximum_tanh_coordinate": max_latent_coordinate}
            round_latent[:, local] = torch.atanh(coordinate)
            current = layer._update_color(current, round_latent, floor, color)
            mismatch = (current[:, vertices] - desired_table[:, vertices]).abs().amax()
            if float(mismatch) > 2e-12:
                return {"rounds": rounds, "success": False,
                        "reason": "forward-inverse numerical mismatch",
                        "failed_round": round_index + 1, "failed_color": color,
                        "maximum_active_vertex_mismatch": float(mismatch)}
        latent_rounds.append(round_latent.reshape(1, side - 2, side - 2, 2).clone())
    mapped = current.reshape_as(base)
    replay = base
    for round_latent in latent_rounds:
        replay = layer(replay, round_latent)
    return {
        "rounds": rounds, "success": True,
        "elapsed_seconds": time.perf_counter() - start_time,
        "maximum_adverse_ratio": max_adverse,
        "maximum_tanh_coordinate": max_latent_coordinate,
        "final_coordinate_rmse": float((mapped - target).square().mean().sqrt()),
        "maximum_vertex_coordinate_error": float((mapped - target).abs().amax()),
        "maximum_complete_round_replay_error": float((mapped - replay).abs().amax()),
        "minimum_normalized_q1_corner": float(
            q1_corner_determinants(mapped).amin() * (side - 1) ** 2),
        "vertices": mapped.detach().cpu().numpy(),
        "boundary_reference": _map(side, 0., device).detach().cpu().numpy(),
        "latent": torch.stack(latent_rounds).detach().cpu().numpy(),
    }


def run(side: int, device: str, prefix: Path) -> dict:
    target_device = torch.device(device)
    trials = []
    for rounds in (1, 2, 4, 8, 16, 32, 64):
        result = attempt(side, rounds, target_device)
        if result["success"]:
            archive = prefix.with_suffix(".npz")
            latent_values = result["latent"].copy()
            np.savez_compressed(archive, vertices=result.pop("vertices"),
                                boundary_reference=result.pop("boundary_reference"),
                                latent=result.pop("latent"))
            result["certificate"] = certify_q1_binary_map(archive)
            result["saved_map"] = archive.name
            # Replay the same oracle logits in the actual float32 deployment
            # dtype; this is a measured saved-map property, not an all-input
            # rounding theorem.
            layer32 = AdaptiveSoftRadialQ1Relaxation(side).to(target_device)
            with torch.no_grad():
                mapped32 = _map(side, .1, target_device).to(torch.float32)
                target32 = _map(side, .07, target_device).to(torch.float32)
                for logits64 in latent_values:
                    mapped32 = layer32(mapped32, torch.from_numpy(logits64).to(
                        device=target_device, dtype=torch.float32))
            archive32 = prefix.with_name(prefix.name + "_float32.npz")
            np.savez_compressed(
                archive32,
                vertices=mapped32.cpu().numpy().astype(np.float32),
                boundary_reference=_map(side, 0., target_device).to(
                    torch.float32).cpu().numpy(),
            )
            result["float32_replay"] = {
                "final_coordinate_rmse": float((mapped32 - target32).square().mean().sqrt()),
                "maximum_vertex_coordinate_error": float((mapped32 - target32).abs().amax()),
                "minimum_normalized_q1_corner": float(
                    q1_corner_determinants(mapped32).amin() * (side - 1) ** 2),
                "certificate": certify_q1_binary_map(archive32),
                "saved_map": archive32.name,
            }
        trials.append(result)
        if result["success"]:
            break
    report = {
        "question": "constructive latent reachability for an analytic 257-grid map",
        "method": "linear map waypoints and exact local soft-radial inverse per F1 color",
        "side": side, "device": device, "dtype": "float64",
        "source": "x+0.1*sin(2*pi*x)*sin(pi*y), y",
        "target": "x+0.07*sin(2*pi*x)*sin(pi*y), y",
        "trials": trials,
    }
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.side, args.device, args.prefix), indent=2))


if __name__ == "__main__":
    main()
