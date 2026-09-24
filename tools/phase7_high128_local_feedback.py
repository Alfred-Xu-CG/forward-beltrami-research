"""Test image-derived fine residual updates after a safe learned coarse P1 map.

The target map is used only for evaluation. The feedback proposal uses image
residuals and 2x2 local ridge solves, followed by safe colored P1 updates.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset
from phase7_train_forward_pyramid_image import (
    minimum_jacobian, replace_test_appearance, sinusoidal_mode_coefficients,
)
from qcopt.neural_bijection.dense import (
    ForwardP1ImageEncoder, HybridPatchSeedVertexP1Pyramid,
    SafeColoredVertexRelaxation, certify_p1_or_identity,
    exact_dyadic_p1_refine, local_photometric_logits,
)
from qcopt.neural_bijection.dense.photometric_hint import physical_image_gradient
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
    parser.add_argument("--train-steps", type=int, default=0,
                        help="Image-only joint training of the coarse encoder and two feedback gains.")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--train-appearance-augmentation",
                        choices=("none", "crosswaves", "spots", "all"),
                        default="none",
                        help="Re-pair the same training maps with additional moving-image textures.")
    parser.add_argument("--train-learning-rate", type=float, default=0.0002)
    parser.add_argument("--train-gain-learning-rate", type=float, default=0.002)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--load-feedback-state", default=None,
                        help="Load a previously image-only trained encoder and two feedback gains.")
    parser.add_argument("--learned-residual", action="store_true",
                        help="Add a shared image-residual CNN proposal at each safe fine update.")
    parser.add_argument("--correction-only-final", action="store_true",
                        help="Apply learned residual only to 1025-square updates.")
    parser.add_argument("--freeze-coarse", action="store_true",
                        help="Train only feedback gains and/or learned correction, not the image encoder.")
    parser.add_argument("--freeze-gains", action="store_true",
                        help="Do not optimize two local-feedback gains.")
    parser.add_argument("--train-correction-learning-rate", type=float, default=0.002)
    parser.add_argument("--test-appearance", choices=("standard", "spots", "crosswaves"),
                        default="standard")
    args = parser.parse_args()
    if min(args.count, args.batch, args.passes_per_level, args.train_count) < 1 or (
        args.train_steps < 0 or args.train_learning_rate <= 0 or
        args.train_gain_learning_rate <= 0 or
        args.train_correction_learning_rate <= 0
    ):
        raise ValueError("counts, batch and passes must be positive")
    if args.correction_only_final and not args.learned_residual:
        raise ValueError("--correction-only-final requires --learned-residual")
    torch.manual_seed(20260924)
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
    log_gains = torch.nn.Parameter(torch.zeros(2, device=device, dtype=torch.float64))
    correction_net = torch.nn.Sequential(
        torch.nn.Conv2d(7, 16, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(16, 2, 1),
    ).to(device)
    torch.nn.init.zeros_(correction_net[-1].weight)
    torch.nn.init.zeros_(correction_net[-1].bias)
    if args.load_feedback_state:
        trained_state = torch.load(args.load_feedback_state, map_location=device, weights_only=True)
        encoder.load_state_dict(trained_state["encoder"])
        with torch.no_grad():
            log_gains.copy_(trained_state["log_gains"].to(device))
        if args.learned_residual and trained_state.get("correction_net") is not None:
            correction_net.load_state_dict(trained_state["correction_net"])
        elif args.learned_residual and not args.train_steps:
            raise ValueError("learned residual evaluation requires correction_net weights")
    relax = {
        side: SafeColoredVertexRelaxation(
            side, motion_mode="radial", raw_span=2.0,
        ).to(device)
        for side in (513, 1025)
    }
    table = StructuredDenseQueryTable.from_shape(1024, 1024, height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    feedback_tables = {
        side: StructuredDenseQueryTable.from_shape(
            side - 1, side - 1, height=512, width=512,
        )
        for side in (513, 1025)
    } if args.learned_residual else {}
    for feedback_table in feedback_tables.values():
        feedback_table.prepare(device=device, dtype=torch.float64)
    image_axis = torch.arange(512, device=device, dtype=torch.float32) / 511
    coord_y, coord_x = torch.meshgrid(image_axis, image_axis, indexing="ij")
    image_coordinates = torch.stack((coord_x, coord_y), dim=0)[None]
    data = tuple(value.to(device) for value in make_dataset(
        args.count, 512, args.seed, target_family="high128",
    ))
    data = replace_test_appearance(data, args.test_appearance, args.seed)

    def learned_proposal(fixed: torch.Tensor, moving: torch.Tensor,
                         current: torch.Tensor, side: int) -> torch.Tensor:
        query = feedback_tables[side].interpolate(current.flatten(1, 2)).float()
        warped = F.grid_sample(
            moving, 2 * query - 1,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        warped_gradient = F.grid_sample(
            physical_image_gradient(moving), 2 * query - 1,
            mode="bilinear", padding_mode="border", align_corners=True,
        ) / 32.0
        features = torch.cat((
            fixed, warped, 100.0 * (fixed - warped), warped_gradient,
            image_coordinates.expand(fixed.shape[0], -1, -1, -1),
        ), dim=1)
        correction = correction_net(features)
        correction = F.interpolate(
            correction, size=(side, side), mode="bilinear", align_corners=True,
        )
        return correction[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1).double()

    def feedback_forward(fixed: torch.Tensor, moving: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seed, levels = encoder(fixed, moving)
        current = coarse(seed[0], levels[:4])
        for level, side in enumerate((513, 1025)):
            current = exact_dyadic_p1_refine(current)
            for _ in range(args.passes_per_level):
                hint = local_photometric_logits(
                    fixed, moving, current.float(),
                    window=args.window, ridge=args.ridge, raw_span=2.0,
                ).to(torch.float64)
                floor = current.new_full((current.shape[0],), .05 / (side - 1) ** 2)
                gain = torch.exp(log_gains[level].clamp(-4, 4))
                proposal = gain * hint
                if args.learned_residual and (not args.correction_only_final or side == 1025):
                    proposal = proposal + learned_proposal(fixed, moving, current, side)
                current = relax[side](current, proposal, area_floor=floor)
        return certify_p1_or_identity(current, full.final_identity)

    train_report = None
    if args.train_steps:
        train = tuple(value.to(device) for value in make_dataset(
            args.train_count, 512, 55101, target_family="high128",
        ))
        if args.train_appearance_augmentation != "none":
            modes = (
                ("crosswaves", "spots")
                if args.train_appearance_augmentation == "all"
                else (args.train_appearance_augmentation,)
            )
            variants = (train,) + tuple(
                replace_test_appearance(train, mode, 55101)
                for mode in modes
            )
            train = tuple(torch.cat(
                [variant[index] for variant in variants], dim=0,
            ) for index in range(3))
        effective_train_pairs = train[0].shape[0]
        for name, parameter in encoder.named_parameters():
            parameter.requires_grad_(not args.freeze_coarse
                                     and not name.startswith("level_heads.4.")
                                     and not name.startswith("level_heads.5."))
        log_gains.requires_grad_(not args.freeze_gains)
        groups = []
        encoder_parameters = [p for p in encoder.parameters() if p.requires_grad]
        if encoder_parameters:
            groups.append({"params": encoder_parameters, "lr": args.train_learning_rate})
        if log_gains.requires_grad:
            groups.append({"params": [log_gains], "lr": args.train_gain_learning_rate})
        if args.learned_residual:
            groups.append({
                "params": list(correction_net.parameters()),
                "lr": args.train_correction_learning_rate,
            })
        if not groups:
            raise ValueError("no trainable parameter group")
        optimizer = torch.optim.Adam(groups, eps=1e-12)
        generator = torch.Generator(device="cpu").manual_seed(6019)
        encoder.train()
        correction_net.train()
        times = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(1, args.train_steps + 1):
            selected = torch.randint(effective_train_pairs, (args.batch,), generator=generator).to(device)
            fixed, moving = (tensor[selected] for tensor in train[:2])
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            control, accepted = feedback_forward(fixed, moving)
            if not bool(accepted.all()):
                raise RuntimeError(f"noncertified feedback at step {step}")
            query = table.interpolate(control.flatten(1, 2)).float()
            warped = F.grid_sample(
                moving, 2 * query - 1,
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            loss = (warped - fixed).square().mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite training loss at step {step}")
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - began)
        encoder.eval()
        correction_net.eval()
        train_report = {
            "steps": args.train_steps,
            "train_count": args.train_count,
            "train_appearance_augmentation": args.train_appearance_augmentation,
            "effective_train_pairs": effective_train_pairs,
            "train_seed": 55101,
            "training_objective": "image_mse_only",
            "median_train_step_seconds": statistics.median(times),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "learned_gains": torch.exp(log_gains.detach()).tolist(),
            "correction_parameters": (
                sum(parameter.numel() for parameter in correction_net.parameters())
                if args.learned_residual else 0
            ),
            "freeze_coarse": args.freeze_coarse,
            "freeze_gains": args.freeze_gains,
            "correction_only_final": args.correction_only_final,
            "last_training_image_mse": float(loss.detach()),
        }
        if args.save_state:
            torch.save({
                "encoder": encoder.state_dict(),
                "log_gains": log_gains.detach().cpu(),
                "correction_net": (
                    correction_net.state_dict() if args.learned_residual else None
                ),
                "config": vars(args),
            }, args.save_state)

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
            feedback, _ = feedback_forward(fixed, moving)
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
        "load_feedback_state": args.load_feedback_state,
        "learned_residual": args.learned_residual,
        "correction_only_final": args.correction_only_final,
        "test_appearance": args.test_appearance,
        "batch": args.batch, "passes_per_level": args.passes_per_level,
        "window": args.window, "ridge": args.ridge,
        "control_vertices": 1025 ** 2, "control_faces": 2 * 1024 ** 2,
        "image_side": 512, "results": result,
        "training": train_report,
        "feedback_gains": torch.exp(log_gains.detach()).tolist(),
    }
    if args.check_vjp:
        fixed, moving, _ = (tensor[:1] for tensor in data)
        encoder.zero_grad(set_to_none=True)
        correction_net.zero_grad(set_to_none=True)
        log_gains.grad = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        current, accepted = feedback_forward(fixed, moving)
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
        correction_gradients = [
            p.grad for p in correction_net.parameters() if p.grad is not None
        ] if args.learned_residual else []
        report["vjp"] = {
            "loss": float(loss.detach()),
            "accepted": bool(accepted.item()),
            "encoder_gradient_tensors": len(gradients),
            "encoder_gradient_max": max(float(g.abs().amax()) for g in gradients),
            "all_encoder_gradients_finite": all(bool(torch.isfinite(g).all()) for g in gradients),
            "gain_gradient": log_gains.grad.detach().tolist() if log_gains.grad is not None else None,
            "correction_gradient_tensors": len(correction_gradients),
            "correction_gradient_max": (
                max(float(g.abs().amax()) for g in correction_gradients)
                if correction_gradients else None
            ),
            "all_correction_gradients_finite": all(
                bool(torch.isfinite(g).all()) for g in correction_gradients
            ),
            "seconds": time.perf_counter() - began,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
