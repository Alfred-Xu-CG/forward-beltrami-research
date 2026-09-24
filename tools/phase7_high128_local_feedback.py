"""Test image-derived fine residual updates after a safe learned coarse P1 map.

The target map is used only for evaluation. The feedback proposal uses image
residuals and 2x2 local ridge solves, followed by safe colored P1 updates.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset
from phase7_train_forward_pyramid_image import (
    minimum_jacobian, sinusoidal_mode_coefficients,
)
from qcopt.neural_bijection.dense import (
    ForwardP1ImageEncoder, HybridPatchSeedVertexP1Pyramid,
    SafeColoredVertexRelaxation, certify_p1_or_identity,
    exact_dyadic_p1_refine, local_photometric_logits,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-side", type=int, default=257)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99317)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--passes-per-level", type=int, default=1)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-vjp", action="store_true",
                        help="Backpropagate one image loss through both fine feedback levels and encoder.")
    args = parser.parse_args()
    if min(args.count, args.batch, args.passes_per_level) < 1:
        raise ValueError("counts, batch and passes must be positive")
    device = torch.device(args.device)
    full = HybridPatchSeedVertexP1Pyramid(
        17, 1025, patch_cells=8, seed_cycles=4,
        compute_dtype=torch.float64,
    ).to(device)
    coarse = HybridPatchSeedVertexP1Pyramid(
        17, 257, patch_cells=8, seed_cycles=4,
        compute_dtype=torch.float64,
    ).to(device)
    encoder = ForwardP1ImageEncoder(
        17, full.level_sides, seed_passes=1,
        feature_side=args.feature_side, width=16, flow_hint=True,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    relax = {
        side: SafeColoredVertexRelaxation(
            side, motion_mode="radial", raw_span=2.0,
        ).to(device)
        for side in (513, 1025)
    }
    table = StructuredDenseQueryTable.from_shape(1024, 1024, height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    data = tuple(value.to(device) for value in make_dataset(
        args.count, 512, args.seed, target_family="high128",
    ))

    def metrics(control: torch.Tensor, fixed: torch.Tensor,
                moving: torch.Tensor, true_map: torch.Tensor) -> dict[str, float]:
        query = table.interpolate(control.flatten(1, 2)).float()
        warped = F.grid_sample(
            moving, 2 * query - 1,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        target_mode = sinusoidal_mode_coefficients(true_map, 128)
        predicted_mode = sinusoidal_mode_coefficients(query, 128)
        return {
            "image_mse": float((warped - fixed).square().mean()),
            "map_squared": float((query - true_map).square().sum(dim=-1).mean()),
            "mode_squared": float((predicted_mode - target_mode).square().sum(dim=-1).mean()),
            "predicted_mode_squared": float(predicted_mode.square().sum(dim=-1).mean()),
            "true_mode_squared": float(target_mode.square().sum(dim=-1).mean()),
            "minimum_jacobian": minimum_jacobian(control),
            "identity_outputs": int(torch.all(
                control == full.final_identity,
                dim=(1, 2, 3),
            ).sum()),
        }

    sums = {name: {
        "image_mse": 0., "map_squared": 0., "mode_squared": 0.,
        "predicted_mode_squared": 0., "true_mode_squared": 0.,
        "minimum_jacobian": math.inf, "identity_outputs": 0,
    } for name in ("learned_full", "coarse_plus_feedback")}
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            fixed, moving, true_map = (
                tensor[start:start + args.batch] for tensor in data
            )
            seed, levels = encoder(fixed, moving)
            baseline = full(seed[0], levels)
            current = coarse(seed[0], levels[:4])
            for side in (513, 1025):
                current = exact_dyadic_p1_refine(current)
                for _ in range(args.passes_per_level):
                    hint = local_photometric_logits(
                        fixed, moving, current.float(),
                        window=args.window, ridge=args.ridge, raw_span=2.0,
                    ).to(torch.float64)
                    floor = current.new_full((current.shape[0],), .05 / (side - 1) ** 2)
                    current = relax[side](current, hint, area_floor=floor)
            feedback, _ = certify_p1_or_identity(current, full.final_identity)
            for name, control in (("learned_full", baseline),
                                  ("coarse_plus_feedback", feedback)):
                row = metrics(control, fixed, moving, true_map)
                weight = fixed.shape[0]
                for key in ("image_mse", "map_squared", "mode_squared",
                            "predicted_mode_squared", "true_mode_squared"):
                    sums[name][key] += weight * row[key]
                sums[name]["minimum_jacobian"] = min(
                    sums[name]["minimum_jacobian"], row["minimum_jacobian"],
                )
                sums[name]["identity_outputs"] += row["identity_outputs"]
    result = {}
    for name, values in sums.items():
        result[name] = {
            "image_mse": values["image_mse"] / args.count,
            "map_rmse": math.sqrt(values["map_squared"] / args.count),
            "mode_rmse": math.sqrt(values["mode_squared"] / args.count),
            "predicted_mode_rms": math.sqrt(values["predicted_mode_squared"] / args.count),
            "true_mode_rms": math.sqrt(values["true_mode_squared"] / args.count),
            "minimum_jacobian": values["minimum_jacobian"],
            "identity_outputs": values["identity_outputs"],
        }
    report = {
        "method": "phase7_high128_local_feedback", "checkpoint": args.checkpoint,
        "feature_side": args.feature_side, "count": args.count, "seed": args.seed,
        "batch": args.batch, "passes_per_level": args.passes_per_level,
        "window": args.window, "ridge": args.ridge,
        "control_vertices": 1025 ** 2, "control_faces": 2 * 1024 ** 2,
        "image_side": 512, "results": result,
    }
    if args.check_vjp:
        fixed, moving, _ = (tensor[:1] for tensor in data)
        encoder.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        seed, levels = encoder(fixed, moving)
        current = coarse(seed[0], levels[:4])
        for side in (513, 1025):
            current = exact_dyadic_p1_refine(current)
            for _ in range(args.passes_per_level):
                hint = local_photometric_logits(
                    fixed, moving, current.float(),
                    window=args.window, ridge=args.ridge, raw_span=2.0,
                ).to(torch.float64)
                floor = current.new_full((1,), .05 / (side - 1) ** 2)
                current = relax[side](current, hint, area_floor=floor)
        current, accepted = certify_p1_or_identity(current, full.final_identity)
        query = table.interpolate(current.flatten(1, 2)).float()
        warped = F.grid_sample(
            moving, 2 * query - 1,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        loss = (warped - fixed).square().mean()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
        report["vjp"] = {
            "loss": float(loss.detach()),
            "accepted": bool(accepted.item()),
            "encoder_gradient_tensors": len(gradients),
            "encoder_gradient_max": max(float(g.abs().amax()) for g in gradients),
            "all_encoder_gradients_finite": all(bool(torch.isfinite(g).all()) for g in gradients),
            "seconds": time.perf_counter() - began,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
