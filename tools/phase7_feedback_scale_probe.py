"""Zero-shot larger fixed-P1 control grids for trained local image feedback.

The image encoder was trained through 1025 square. At 2049/4097 square this
is a scale/memory probe, not a newly trained high-resolution registration
result. All extra-level proposals are obtained from current image residuals.
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
from phase7_train_forward_pyramid_image import minimum_jacobian
from qcopt.neural_bijection.dense import (
    ForwardP1ImageEncoder, HybridPatchSeedVertexP1Pyramid,
    SafeColoredVertexRelaxation, certify_p1_or_identity,
    exact_dyadic_p1_refine, local_photometric_logits,
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
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--check-vjp", action="store_true")
    parser.add_argument("--checkpoint-extra", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--geometry-dtype", choices=("float64", "float32"),
                        default="float64")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.count, args.extra_passes, args.timing_repeats) < 1:
        raise ValueError("count, extra-passes and timing-repeats must be positive")
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
    )
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
        args.count, 512, args.seed, target_family="high128",
    ))
    setup_seconds = time.perf_counter() - setup_started
    setup_cuda_allocated = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    )

    def forward(fixed: torch.Tensor, moving: torch.Tensor, *,
                record_extra: bool = False):
        seed, levels = encoder(fixed, moving)
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
                    gain = torch.exp(log_gains[min(current_level, 1)])
                    updated = relax[current_side](
                        updated, gain * hint, area_floor=floor,
                    )
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
            "accepted": bool(accepted.item()),
            "minimum_jacobian": minimum_jacobian(mapped),
            "minimum_jacobian_recomputed_float64": minimum_jacobian(
                mapped.double()
            ),
            "image_mse": float((warped - fixed).square().mean()),
            "map_rmse": float(
                (query - true_map).square().sum(dim=-1).mean().sqrt()
            ),
            "inference_seconds": statistics.median(inference_times),
            "inference_times": inference_times,
            "device": str(device),
            "torch_version": torch.__version__,
            "geometry_dtype": args.geometry_dtype,
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
            ),
            "gain_gradient_finite": bool(
                torch.isfinite(log_gains.grad).all()
            ),
            "minimum_jacobian": minimum_jacobian(mapped),
        }
    if args.count > 1:
        accepted_count = 0
        minimum_double = float("inf")
        squared_map_error = 0.0
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
        report["cohort_validation"] = {
            "count": args.count,
            "accepted_count": accepted_count,
            "minimum_jacobian_recomputed_float64": minimum_double,
            "query_map_rmse": (squared_map_error / args.count) ** .5,
        }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
