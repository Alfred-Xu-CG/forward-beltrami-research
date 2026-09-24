"""Larger fixed-P1 grids for trained local image feedback.

The image encoder was trained through 1025 square. At 2049/4097 square this
is zero-shot unless --train-extra-steps explicitly learns the two extra-level
gains from images. All extra-level proposals use current image residuals.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from phase6_train_multisample_image import make_dataset
from phase7_train_forward_pyramid_image import (
    minimum_jacobian, replace_test_appearance, sinusoidal_mode_coefficients,
)
from qcopt.neural_bijection.dense import (
    ForwardP1ImageEncoder, HybridPatchSeedVertexP1Pyramid,
    SafeColoredVertexRelaxation, certify_p1_or_identity,
    exact_dyadic_p1_refine, local_photometric_logits,
    physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-checkpoint", required=True)
    parser.add_argument("--final-side", type=int, choices=(1025, 2049, 4097),
                        default=2049)
    parser.add_argument("--extra-passes", type=int, default=1)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--target-family", choices=("high128", "high128_tri", "high128_tiles"),
                        default="high128")
    parser.add_argument("--test-appearance", choices=("standard", "crosswaves", "spots"),
                        default="standard")
    parser.add_argument("--image-channels", type=int, default=1,
                        help="Independent synthetic views of the same map for image feedback/loss.")
    parser.add_argument("--duplicate-image-channels", action="store_true",
                        help="Repeat one image instead of adding independent texture; information control.")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--check-vjp", action="store_true")
    parser.add_argument("--checkpoint-extra", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--geometry-dtype", choices=("float64", "float32"),
                        default="float64")
    parser.add_argument("--train-extra-steps", type=int, default=0)
    parser.add_argument("--train-extra-count", type=int, default=32)
    parser.add_argument("--train-extra-lr", type=float, default=.01)
    parser.add_argument("--train-extra-objective", choices=("image", "map"),
                        default="image")
    parser.add_argument("--train-loss-scale", type=float, default=1.0)
    parser.add_argument("--save-extra-gains")
    parser.add_argument("--load-extra-gains")
    parser.add_argument("--extra-spatial-correction", action="store_true")
    parser.add_argument("--extra-correction-highpass-window", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.count, args.extra_passes, args.timing_repeats) < 1:
        raise ValueError("count, extra-passes and timing-repeats must be positive")
    if args.train_extra_steps < 0 or args.train_extra_count < 1 or args.train_extra_lr <= 0:
        raise ValueError("invalid extra-level training settings")
    if args.train_loss_scale <= 0:
        raise ValueError("train loss scale must be positive")
    if args.image_channels < 1 or args.image_channels > 4:
        raise ValueError("image channels must be in [1,4]")
    if args.duplicate_image_channels and args.image_channels == 1:
        raise ValueError("duplicate channels require image-channels > 1")
    if args.image_channels > 1 and args.test_appearance != "standard":
        raise ValueError("multichannel appearance is defined for standard only")
    if args.train_extra_steps and args.final_side < 2049:
        raise ValueError("extra-level training needs final-side >= 2049")
    if args.train_extra_steps and not args.save_extra_gains:
        raise ValueError("extra-level training needs --save-extra-gains")
    if args.extra_spatial_correction and args.final_side < 2049:
        raise ValueError("extra spatial correction needs a new fine level")
    if args.extra_correction_highpass_window < 3 or args.extra_correction_highpass_window % 2 != 1:
        raise ValueError("extra correction highpass window must be odd and >= 3")
    setup_started = time.perf_counter()
    device = torch.device(args.device)
    geometry_dtype = getattr(torch, args.geometry_dtype)
    coarse = HybridPatchSeedVertexP1Pyramid(
        17, 257, patch_cells=8, seed_cycles=4,
        compute_dtype=geometry_dtype,
    ).to(device)
    encoder = ForwardP1ImageEncoder(
        17, (33, 65, 129, 257, 513, 1025), seed_passes=1,
        feature_side=257, width=16, flow_hint=True,
    ).to(device)
    feedback_state = torch.load(
        args.feedback_checkpoint, map_location=device, weights_only=True,
    )
    encoder.load_state_dict(feedback_state["encoder"])
    encoder.eval()
    log_gains = torch.nn.Parameter(
        feedback_state["log_gains"].to(device=device, dtype=geometry_dtype),
        requires_grad=not bool(args.train_extra_steps),
    )
    extra_log_gains = torch.nn.Parameter(
        log_gains.detach()[1].repeat(2).clone(),
    )
    extra_correction_net = torch.nn.Sequential(
        torch.nn.Conv2d(7, 16, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(16, 2, 1),
    ).to(device)
    torch.nn.init.zeros_(extra_correction_net[-1].weight)
    torch.nn.init.zeros_(extra_correction_net[-1].bias)
    if args.load_extra_gains:
        loaded = torch.load(args.load_extra_gains, map_location=device, weights_only=True)
        with torch.no_grad():
            extra_log_gains.copy_(loaded["extra_log_gains"].to(geometry_dtype))
        if args.extra_spatial_correction:
            if loaded.get("extra_correction_net") is None:
                if not args.train_extra_steps:
                    raise ValueError("loaded state has no extra spatial correction")
            else:
                extra_correction_net.load_state_dict(loaded["extra_correction_net"])
    if args.train_extra_steps:
        encoder.requires_grad_(False)
    sides = tuple(side for side in (513, 1025, 2049, 4097)
                  if side <= args.final_side)
    relax = {
        side: SafeColoredVertexRelaxation(
            side, motion_mode="radial", raw_span=2.0,
        ).to(device)
        for side in sides
    }
    axis = torch.arange(
        args.final_side, device=device, dtype=geometry_dtype,
    ) / (args.final_side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    final_identity = torch.stack((xx, yy), dim=-1)[None]
    table = StructuredDenseQueryTable.from_shape(
        args.final_side - 1, args.final_side - 1,
        height=512, width=512,
    )
    table.prepare(device=device, dtype=geometry_dtype)
    dataset = tuple(t.to(device) for t in make_dataset(
        args.count, 512, args.seed, target_family=args.target_family,
    ))
    dataset = replace_test_appearance(dataset, args.test_appearance, args.seed)

    def multiview(data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if args.image_channels == 1:
            return data
        fixed, moving, true_map = data
        if args.duplicate_image_channels:
            return (fixed.repeat(1, args.image_channels, 1, 1),
                    moving.repeat(1, args.image_channels, 1, 1), true_map)
        fixed_views = [fixed]
        moving_views = [moving]
        for channel in range(1, args.image_channels):
            extra = make_dataset(
                true_map.shape[0], 512, seed + 10000 * channel,
                target_family="high128",
            )[1].to(device)
            fixed_views.append(F.grid_sample(
                extra, 2 * true_map - 1,
                mode="bilinear", padding_mode="border", align_corners=True,
            ).detach())
            moving_views.append(extra)
        return torch.cat(fixed_views, dim=1), torch.cat(moving_views, dim=1), true_map

    dataset = multiview(dataset, args.seed)
    image_axis = torch.arange(512, device=device, dtype=torch.float32) / 511
    image_y, image_x = torch.meshgrid(image_axis, image_axis, indexing="ij")
    image_coordinates = torch.stack((image_x, image_y), dim=0)[None]
    correction_table = None
    if args.extra_spatial_correction:
        correction_table = StructuredDenseQueryTable.from_shape(
            args.final_side - 1, args.final_side - 1,
            height=512, width=512,
        )
        correction_table.prepare(device=device, dtype=geometry_dtype)
    setup_seconds = time.perf_counter() - setup_started
    setup_cuda_allocated = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    )

    def spatial_correction(fixed: torch.Tensor, moving: torch.Tensor,
                           current: torch.Tensor) -> torch.Tensor:
        assert correction_table is not None
        query = correction_table.interpolate(current.flatten(1, 2)).float()
        warped = F.grid_sample(
            moving, 2 * query - 1,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        gradient = F.grid_sample(
            physical_image_gradient(moving), 2 * query - 1,
            mode="bilinear", padding_mode="border", align_corners=True,
        ) / 32
        features = torch.cat((
            fixed[:, :1], warped[:, :1], 100 * (fixed[:, :1] - warped[:, :1]),
            gradient[:, :1], gradient[:, moving.shape[1]:moving.shape[1] + 1],
            image_coordinates.expand(fixed.shape[0], -1, -1, -1),
        ), dim=1)
        correction = extra_correction_net(features)
        window = args.extra_correction_highpass_window
        correction = correction - F.avg_pool2d(
            correction, window, stride=1, padding=window // 2,
            count_include_pad=False,
        )
        correction = F.interpolate(
            correction, size=(args.final_side, args.final_side),
            mode="bilinear", align_corners=True,
        )
        return correction[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1).to(geometry_dtype)

    def forward(fixed: torch.Tensor, moving: torch.Tensor, *,
                record_extra: bool = False):
        seed, levels = encoder(fixed[:, :1], moving[:, :1])
        current = coarse(seed[0], levels[:4])
        extra_motion = {}
        for level, side in enumerate(sides):
            current = exact_dyadic_p1_refine(current)
            before_extra = current if record_extra and side >= 2049 else None
            passes = 2 if side <= 1025 else args.extra_passes

            def update(base: torch.Tensor, *, current_side: int = side,
                       current_level: int = level,
                       current_passes: int = passes) -> torch.Tensor:
                updated = base
                for _ in range(current_passes):
                    hint = local_photometric_logits(
                        fixed, moving, updated.float(),
                        window=args.window, ridge=1., raw_span=2.,
                    ).to(geometry_dtype)
                    floor = updated.new_full(
                        (updated.shape[0],), .05 / (current_side - 1) ** 2,
                    )
                    gain = torch.exp(
                        log_gains[current_level] if current_level < 2
                        else extra_log_gains[current_level - 2].clamp(-4, 4)
                    )
                    proposal = gain * hint
                    if args.extra_spatial_correction and current_side == args.final_side:
                        proposal = proposal + spatial_correction(
                            fixed, moving, updated,
                        )
                    updated = relax[current_side](updated, proposal, area_floor=floor)
                return updated

            current = (
                checkpoint(update, current, use_reentrant=False)
                if args.checkpoint_extra and side >= 2049 and torch.is_grad_enabled()
                else update(current)
            )
            if before_extra is not None:
                displacement = current - before_extra
                extra_motion[str(side)] = {
                    "max_coordinate_change": float(displacement.abs().amax()),
                    "vertex_vector_rms": float(
                        displacement.square().sum(dim=-1).mean().sqrt()
                    ),
                }
        mapped, accepted = certify_p1_or_identity(current, final_identity)
        return mapped, accepted, extra_motion

    train_report = None
    if args.train_extra_steps:
        train_data = tuple(t.to(device) for t in make_dataset(
            args.train_extra_count, 512, 55101, target_family="high128",
        ))
        train_data = multiview(train_data, 55101)
        train_parameters = [extra_log_gains]
        if args.extra_spatial_correction:
            train_parameters.extend(extra_correction_net.parameters())
        optimizer = torch.optim.Adam(train_parameters, lr=args.train_extra_lr)
        step_times = []
        step_peaks = []
        accepted_steps = 0
        initial_loss = None
        first_gradient_max = None
        torch.manual_seed(20260924)
        for step in range(args.train_extra_steps):
            index = int(torch.randint(args.train_extra_count, (1,)).item())
            train_fixed, train_moving, train_true = (
                value[index:index + 1] for value in train_data
            )
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            train_mapped, train_accepted, _ = forward(train_fixed, train_moving)
            train_query = table.interpolate(train_mapped.flatten(1, 2)).float()
            train_warped = F.grid_sample(
                train_moving, 2 * train_query - 1,
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            train_raw_loss = (
                (train_warped - train_fixed).square().mean()
                if args.train_extra_objective == "image"
                else (train_query - train_true).square().sum(dim=-1).mean()
            )
            train_loss = args.train_loss_scale * train_raw_loss
            if initial_loss is None:
                initial_loss = float(train_raw_loss.detach())
            train_loss.backward()
            if not bool(torch.isfinite(extra_log_gains.grad).all()):
                raise RuntimeError(f"nonfinite extra-level VJP at step {step}")
            if step == 0:
                first_gradient_max = {
                    "extra_gain": float(extra_log_gains.grad.abs().amax()),
                    "correction": max(
                        float(parameter.grad.abs().amax())
                        for parameter in extra_correction_net.parameters()
                        if parameter.grad is not None
                    ) if args.extra_spatial_correction else None,
                }
            optimizer.step()
            accepted_steps += int(train_accepted.item())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                step_peaks.append(torch.cuda.max_memory_allocated(device))
            step_times.append(time.perf_counter() - began)
        torch.save({
            "extra_log_gains": extra_log_gains.detach().cpu(),
            "extra_correction_net": (
                extra_correction_net.state_dict()
                if args.extra_spatial_correction else None
            ),
            "feedback_checkpoint": args.feedback_checkpoint,
            "config": vars(args),
        }, args.save_extra_gains)
        train_report = {
            "steps": args.train_extra_steps,
            "count": args.train_extra_count,
            "batch": 1,
            "train_seed": 55101,
            "objective": args.train_extra_objective,
            "loss_scale": args.train_loss_scale,
            "initial_sample_loss": initial_loss,
            "last_sample_loss": float(train_raw_loss.detach()),
            "first_gradient_max": first_gradient_max,
            "accepted_steps": accepted_steps,
            "extra_gains": torch.exp(extra_log_gains.detach()).tolist(),
            "spatial_correction": args.extra_spatial_correction,
            "median_step_seconds": statistics.median(step_times),
            "peak_cuda_allocated_bytes": max(step_peaks) if step_peaks else None,
        }

    fixed, moving, true_map = (t[:1] for t in dataset)
    with torch.no_grad():
        inference_times = []
        inference_peaks = []
        for _ in range(args.timing_repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            mapped, accepted, _ = forward(fixed, moving)
            query = table.interpolate(mapped.flatten(1, 2)).float()
            warped = F.grid_sample(
                moving, 2 * query - 1,
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                inference_peaks.append(torch.cuda.max_memory_allocated(device))
            inference_times.append(time.perf_counter() - began)
        mapped, accepted, extra_motion = forward(
            fixed, moving, record_extra=True,
        )
        query = table.interpolate(mapped.flatten(1, 2)).float()
        warped = F.grid_sample(
            moving, 2 * query - 1,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        report = {
            "method": "phase7_feedback_scale_probe",
            "final_side": args.final_side,
            "control_vertices": args.final_side ** 2,
            "control_faces": 2 * (args.final_side - 1) ** 2,
            "image_queries": 512 ** 2,
            "extra_passes": args.extra_passes,
            "checkpoint_extra": args.checkpoint_extra,
            "seed": args.seed,
            "target_family": args.target_family,
            "test_appearance": args.test_appearance,
            "accepted": bool(accepted.item()),
            "minimum_jacobian": minimum_jacobian(mapped),
            "minimum_jacobian_recomputed_float64": minimum_jacobian(
                mapped.double()
            ),
            "image_mse": float((warped - fixed).square().mean()),
            "map_rmse": float(
                (query - true_map).square().sum(dim=-1).mean().sqrt()
            ),
            "mode_rmse": float(
                (sinusoidal_mode_coefficients(query, 128)
                 - sinusoidal_mode_coefficients(true_map, 128))
                .square().sum(dim=-1).mean().sqrt()
            ),
            "inference_seconds": statistics.median(inference_times),
            "inference_times": inference_times,
            "device": str(device),
            "torch_version": torch.__version__,
            "geometry_dtype": args.geometry_dtype,
            "extra_gains": torch.exp(extra_log_gains.detach()).tolist(),
            "extra_spatial_correction": args.extra_spatial_correction,
            "image_channels": args.image_channels,
            "duplicate_image_channels": args.duplicate_image_channels,
            "extra_training": train_report,
            "extra_level_actual_motion": extra_motion,
            "setup_seconds": setup_seconds,
            "setup_cuda_allocated_bytes": setup_cuda_allocated,
            "inference_peak_cuda_allocated_bytes": (
                max(inference_peaks) if inference_peaks else None
            ),
        }
    if args.check_vjp:
        vjp_times = []
        peaks = []
        for _ in range(args.timing_repeats):
            encoder.zero_grad(set_to_none=True)
            log_gains.grad = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            mapped, accepted, _ = forward(fixed, moving)
            query = table.interpolate(mapped.flatten(1, 2)).float()
            warped = F.grid_sample(
                moving, 2 * query - 1,
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            loss = (warped - fixed).square().mean()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            vjp_times.append(time.perf_counter() - began)
            peaks.append(
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else None
            )
        gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
        report["vjp"] = {
            "accepted": bool(accepted.item()),
            "seconds": statistics.median(vjp_times),
            "times": vjp_times,
            "peak_cuda_allocated_bytes": max(
                peak for peak in peaks if peak is not None
            ) if device.type == "cuda" else None,
            "encoder_gradient_tensors": len(gradients),
            "all_encoder_gradients_finite": all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            ) if gradients else None,
            "gain_gradient_finite": bool(
                torch.isfinite(log_gains.grad).all()
            ) if log_gains.grad is not None else None,
            "extra_gain_gradient_finite": bool(
                torch.isfinite(extra_log_gains.grad).all()
            ) if extra_log_gains.grad is not None else None,
            "minimum_jacobian": minimum_jacobian(mapped),
        }
    if args.count > 1:
        accepted_count = 0
        minimum_double = float("inf")
        squared_map_error = 0.0
        squared_mode_error = 0.0
        squared_image_error = 0.0
        with torch.no_grad():
            for index in range(args.count):
                case_fixed, case_moving, case_true = (
                    value[index:index + 1] for value in dataset
                )
                case_mapped, case_accepted, _ = forward(
                    case_fixed, case_moving,
                )
                accepted_count += int(case_accepted.item())
                minimum_double = min(
                    minimum_double,
                    minimum_jacobian(case_mapped.double()),
                )
                case_query = table.interpolate(
                    case_mapped.flatten(1, 2)
                ).float()
                squared_map_error += float(
                    (case_query - case_true).square().sum(dim=-1).mean()
                )
                squared_mode_error += float(
                    (sinusoidal_mode_coefficients(case_query, 128)
                     - sinusoidal_mode_coefficients(case_true, 128))
                    .square().sum(dim=-1).mean()
                )
                case_warped = F.grid_sample(
                    case_moving, 2 * case_query - 1,
                    mode="bilinear", padding_mode="border", align_corners=True,
                )
                squared_image_error += float(
                    (case_warped - case_fixed).square().mean()
                )
        report["cohort_validation"] = {
            "count": args.count,
            "accepted_count": accepted_count,
            "minimum_jacobian_recomputed_float64": minimum_double,
            "query_map_rmse": (squared_map_error / args.count) ** .5,
            "mode_rmse": (squared_mode_error / args.count) ** .5,
            "image_mse": squared_image_error / args.count,
        }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
