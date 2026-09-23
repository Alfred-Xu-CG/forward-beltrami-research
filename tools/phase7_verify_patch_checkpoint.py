"""Reconstruct a saved staged patch oracle independently of its fit loop."""

from __future__ import annotations

import argparse
import json
import math

import torch

from phase7_teacher_reachability import target_map
from qcopt.neural_bijection.dense import StaggeredPatchP1Layer, exact_dyadic_p1_refine


def minimum_jacobian(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]

    def cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    return float(torch.minimum(
        cross(b - a, c - a).amin(),
        cross(c - a, d - a).amin(),
    ) * (mapped.shape[1] - 1) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    side = config["seed_side"]
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    current = torch.stack((xx, yy), dim=-1)[None]
    levels = []
    with torch.no_grad():
        for saved_level in saved["latents"]:
            layer = StaggeredPatchP1Layer(side, config["patch_cells"]).to(device)
            latents = tuple(z.to(device) for z in saved_level)
            if len(latents) != len(layer.passes):
                raise ValueError(f"pass count mismatch at side {side}")
            current = layer(current, latents)
            target = target_map(side, device, current.dtype, config["target_kind"])
            difference = current - target
            levels.append({
                "side": side,
                "control_vertices": side ** 2,
                "vertex_vector_rmse": float(difference.square().sum(dim=-1).mean().sqrt()),
                "maximum_coordinate_error": float(difference.abs().amax()),
                "minimum_jacobian": minimum_jacobian(current),
                "finite": bool(torch.isfinite(current).all()),
            })
            if side != config["side"]:
                current = exact_dyadic_p1_refine(current)
                side = 2 * side - 1
    if side != config["side"]:
        raise ValueError(f"final side {side} disagrees with checkpoint {config['side']}")
    if any(not row["finite"] or row["minimum_jacobian"] <= 0 for row in levels):
        raise RuntimeError("reconstruction contains a nonfinite or nonpositive face")
    if not all(math.isfinite(row["vertex_vector_rmse"]) for row in levels):
        raise RuntimeError("nonfinite reconstruction error")
    print(json.dumps({
        "method": "phase7_independent_patch_checkpoint_reconstruction",
        "checkpoint": args.checkpoint,
        "target_kind": config["target_kind"],
        "device": str(device),
        "levels": levels,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
