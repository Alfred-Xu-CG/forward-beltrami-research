"""Known-target inverse patch latents, diagnostic rather than learned inference."""

from __future__ import annotations

import argparse
import json
import math

import torch

from phase7_teacher_reachability import area_stats, target_map
from qcopt.neural_bijection.dense import StaggeredPatchP1Layer, exact_dyadic_p1_refine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--seed-side", type=int, default=17)
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--target-kind", choices=("base", "high32", "high64", "high128", "local_swirl"),
                        default="high32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--raw-span", type=float, default=0.5)
    args = parser.parse_args()
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    side = args.seed_side
    axis = torch.arange(side, device=device, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    current = torch.stack((xx, yy), dim=-1)[None]
    stages = []
    while True:
        target = target_map(side, device, dtype, args.target_kind)
        before = float((current - target).square().sum(dim=-1).mean().sqrt())
        layer = StaggeredPatchP1Layer(side, args.patch_cells, raw_span=args.raw_span).to(device)
        new_mask = torch.zeros((side, side), dtype=torch.bool, device=device)
        if side == args.seed_side:
            new_mask[1:-1, 1:-1] = True
        else:
            new_mask[1:-1, 1:-1] = True
            new_mask[::2, ::2] = False
        remaining = new_mask.reshape(-1).clone()
        pass_records = []
        for patch_pass in layer.passes:
            ids = patch_pass.interior_ids
            chosen = ids[remaining[ids]]
            logits = torch.zeros((1, side * side, 2), device=device, dtype=dtype)
            raw_limit = args.raw_span * args.patch_cells / (side - 1)
            ratio = (target.reshape(1, -1, 2)[:, chosen]
                     - current.reshape(1, -1, 2)[:, chosen]) / raw_limit
            max_ratio = float(ratio.abs().amax()) if chosen.numel() else 0.0
            if max_ratio >= 1:
                raise RuntimeError(f"raw span too small at side={side}, ratio={max_ratio}")
            logits[:, chosen] = torch.atanh(ratio)
            current = patch_pass(current, logits.reshape(1, side, side, 2)[:, 1:-1, 1:-1])
            remaining[chosen] = False
            pass_records.append({
                "chosen_vertices": int(chosen.numel()),
                "maximum_raw_span_ratio": max_ratio,
                "maximum_target_residual": float((current - target).abs().amax()),
            })
        if remaining.any().item():
            raise AssertionError("staggered patches failed to cover every required vertex")
        minimum, invalid = area_stats(current)
        stages.append({
            "side": side,
            "before_vertex_vector_rmse": before,
            "after_vertex_vector_rmse": float((current - target).square().sum(dim=-1).mean().sqrt()),
            "after_maximum_coordinate_error": float((current - target).abs().amax()),
            "target_minimum_jacobian": area_stats(target)[0],
            "minimum_jacobian": minimum,
            "nonpositive_faces": invalid,
            "passes": pass_records,
        })
        if side == args.side:
            break
        side = 2 * side - 1
        if side > args.side:
            raise ValueError("final side must be reachable from seed by dyadic refinement")
        current = exact_dyadic_p1_refine(current)
    print(json.dumps({
        "method": "phase7_patch_teacher_inverse_latents",
        "side": args.side,
        "target_kind": args.target_kind,
        "device": args.device,
        "dtype": args.dtype,
        "seed_side": args.seed_side,
        "patch_cells": args.patch_cells,
        "raw_span": args.raw_span,
        "stages": stages,
        "note": "Known target coordinates supply latents. No image encoder is involved.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
