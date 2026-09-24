"""Real image-only training through the fixed-grid forward P1 pyramid.

The known analytic target map is used only for post-training evaluation. The
training loss sees fixed/moving image intensities and the decoded warp.
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
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    ForwardP1ImageEncoder, ForwardP1Pyramid, ForwardPatchP1Pyramid,
    HybridPatchSeedVertexP1Pyramid, PatchPyramidImageEncoder,
    SafeColoredVertexRelaxation,
    local_photometric_logits, spectralize_bounded_logits,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def minimum_jacobian(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]
    def cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return float(torch.minimum(cross(b - a, c - a).amin(), cross(c - a, d - a).amin())
                 * (mapped.shape[1] - 1) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--decoder-kind", choices=("colored", "patch", "hybrid"), default="colored")
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--target-family", choices=("base", "high32", "high64", "high128"), default="high32")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--seed-passes", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--feature-side", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--training-objective", choices=("image", "map"), default="image")
    parser.add_argument("--feedback-passes", type=int, default=0)
    parser.add_argument("--feedback-side", type=int, default=None,
                        help="Apply safe image feedback on a pyramid level before finer refinement.")
    parser.add_argument("--hint-window", type=int, default=3)
    parser.add_argument("--hint-ridge", type=float, default=1.0)
    parser.add_argument("--hint-gain", type=float, default=1.0)
    parser.add_argument("--hint-sine-modes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--load-state", default=None)
    parser.add_argument("--allow-new-levels", action="store_true",
                        help="Load a smaller-side encoder and retain zero-initialized new level heads.")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if min(args.side, args.image_side, args.train_count, args.test_count,
           args.width, args.batch, args.steps) < 1 or (
               args.feature_side is not None and args.feature_side < 17
           ) or args.adam_eps <= 0:
        raise ValueError("dimensions, counts, batch and steps must be positive")
    if args.decoder_kind == "hybrid" and args.feedback_passes:
        raise ValueError("hybrid decoder has no intermediate feedback path")
    torch.manual_seed(20260924)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    train = tuple(t.to(device) for t in make_dataset(
        args.train_count, args.image_side, 55101, target_family=args.target_family,
    ))
    test = tuple(t.to(device) for t in make_dataset(
        args.test_count, args.image_side, args.test_seed,
        target_family=args.target_family,
    ))
    if args.decoder_kind == "colored":
        decoder = ForwardP1Pyramid(5, args.side, seed_passes=args.seed_passes,
                                   minimum_jacobian=0.05).to(device)
        encoder = ForwardP1ImageEncoder(
            5, decoder.level_sides, seed_passes=args.seed_passes,
            feature_side=args.feature_side or min(args.side, 257), width=args.width,
        ).to(device)
    elif args.decoder_kind == "patch":
        decoder = ForwardPatchP1Pyramid(
            17, args.side, patch_cells=args.patch_cells,
            minimum_jacobian=0.05,
        ).to(device)
        encoder = PatchPyramidImageEncoder(
            17, decoder.level_sides,
            feature_side=args.feature_side or min(args.side, 257), width=args.width,
        ).to(device)
    else:
        decoder = HybridPatchSeedVertexP1Pyramid(
            17, args.side, patch_cells=args.patch_cells,
            seed_cycles=args.seed_passes,
            minimum_jacobian=.05,
            compute_dtype=torch.float64,
        ).to(device)
        encoder = ForwardP1ImageEncoder(
            17, decoder.level_sides, seed_passes=1,
            feature_side=args.feature_side or min(args.side, 257),
            width=args.width,
        ).to(device)
    feedback_side = args.feedback_side or args.side
    valid_feedback_sides = (decoder.seed_side,) + decoder.level_sides
    if feedback_side not in valid_feedback_sides:
        raise ValueError(f"feedback-side must be one of {valid_feedback_sides}")
    feedback_index = (
        0 if feedback_side == decoder.seed_side
        else decoder.level_sides.index(feedback_side) + 1
    )
    if args.feedback_passes and feedback_side < args.side:
        if args.decoder_kind == "colored":
            coarse_decoder = ForwardP1Pyramid(
                5, feedback_side, seed_passes=args.seed_passes,
                minimum_jacobian=0.05,
            ).to(device)
        else:
            coarse_decoder = ForwardPatchP1Pyramid(
                17, feedback_side, patch_cells=args.patch_cells,
                minimum_jacobian=0.05,
            ).to(device)
    else:
        coarse_decoder = None
    feedback = SafeColoredVertexRelaxation(
        feedback_side, motion_mode="radial", raw_span=2.0,
    ).to(device) if args.feedback_passes else None
    if args.load_state is not None:
        saved = torch.load(args.load_state, map_location=device, weights_only=True)
        if args.allow_new_levels:
            missing, unexpected = encoder.load_state_dict(saved["encoder"], strict=False)
            if unexpected or any(not key.startswith("level_heads.") for key in missing):
                raise ValueError(f"incompatible encoder transfer: missing={missing}, unexpected={unexpected}")
            if not missing:
                raise ValueError("--allow-new-levels requires a checkpoint with fewer level heads")
        else:
            encoder.load_state_dict(saved["encoder"])
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(
        device=device,
        dtype=(torch.float64 if args.decoder_kind == "hybrid"
               else torch.float32),
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate,
                                 eps=args.adam_eps)
    generator = torch.Generator(device="cpu").manual_seed(6019)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward(fixed: torch.Tensor, moving: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seed, levels = encoder(fixed, moving)
        control = (
            decoder(seed[0], levels) if args.decoder_kind == "hybrid" else
            coarse_decoder(seed, levels[:feedback_index])
            if coarse_decoder is not None else decoder(seed, levels)
        )
        for _ in range(args.feedback_passes):
            hint = local_photometric_logits(
                fixed, moving, control,
                window=args.hint_window, ridge=args.hint_ridge, raw_span=2.0,
            )
            if args.hint_sine_modes:
                hint = spectralize_bounded_logits(
                    hint, side=feedback_side, raw_span=2.0, count=args.hint_sine_modes,
                )
            floor = control.new_full((control.shape[0],), 0.05 / (feedback_side - 1) ** 2)
            control = feedback(control, args.hint_gain * hint, area_floor=floor)
        if coarse_decoder is not None:
            control = decoder.forward_from(
                control, levels[feedback_index:], start_index=feedback_index,
            )
        query = table.interpolate(control.reshape(control.shape[0], -1, 2)).float()
        warped = F.grid_sample(moving, 2 * query - 1,
                               mode="bilinear", padding_mode="border", align_corners=True)
        image_loss = (warped - fixed).square().mean()
        return image_loss, control, query

    @torch.no_grad()
    def evaluate(dataset: tuple[torch.Tensor, ...]) -> dict[str, float]:
        image_errors, map_errors, margins = [], [], []
        identity_outputs = 0
        for start in range(0, dataset[0].shape[0], args.batch):
            fixed, moving, true_map = (t[start:start + args.batch] for t in dataset)
            loss, control, query = forward(fixed, moving)
            image_errors.append(float(loss) * fixed.shape[0])
            map_errors.append(float((query.reshape_as(true_map) - true_map).square().sum(dim=-1).mean()) * fixed.shape[0])
            margins.append(minimum_jacobian(control))
            if args.decoder_kind == "hybrid":
                identity_outputs += int(torch.all(
                    control == decoder.final_identity,
                    dim=(1, 2, 3),
                ).sum())
        count = dataset[0].shape[0]
        return {
            "image_mse": sum(image_errors) / count,
            "query_map_vector_rmse": math.sqrt(sum(map_errors) / count),
            "minimum_jacobian": min(margins),
            "identity_outputs": identity_outputs if args.decoder_kind == "hybrid" else None,
        }

    encoder.eval()
    initial = evaluate(test)
    if args.eval_only:
        print(json.dumps({
            "method": "phase7_forward_p1_image_encoder_eval_only",
            "decoder_kind": args.decoder_kind,
            "side": args.side,
            "control_vertices": args.side ** 2,
            "control_faces": 2 * (args.side - 1) ** 2,
            "image_side": args.image_side,
            "target_family": args.target_family,
            "test_count": args.test_count,
            "test_seed": args.test_seed,
            "batch": args.batch,
            "load_state": args.load_state,
            "metrics": initial,
        }, sort_keys=True))
        return
    encoder.train()
    history = []
    times = []
    first_step_fine_head_gradient_max = None
    sync()
    began = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        idx = torch.randint(args.train_count, (args.batch,), generator=generator).to(device)
        fixed, moving, true_map = (tensor[idx] for tensor in train)
        optimizer.zero_grad(set_to_none=True)
        sync()
        tick = time.perf_counter()
        image_loss, control, query = forward(fixed, moving)
        loss = (
            image_loss if args.training_objective == "image"
            else (query.reshape_as(true_map) - true_map).square().sum(dim=-1).mean()
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"nonfinite training loss at step {step}")
        loss.backward()
        if step == 1:
            first_step_fine_head_gradient_max = [
                max(
                    float(parameter.grad.detach().abs().amax())
                    for parameter in head.parameters()
                    if parameter.grad is not None
                )
                for head in encoder.level_heads[-2:]
            ]
        optimizer.step()
        sync()
        times.append(time.perf_counter() - tick)
        with torch.no_grad():
            margin = minimum_jacobian(control)
        if margin <= 0 or not math.isfinite(margin):
            raise RuntimeError(f"P1 face failure at training step {step}: Jmin={margin}")
        if step in {1, args.steps} or step % max(1, args.steps // 10) == 0:
            history.append({"step": step, "train_batch_objective": float(loss.detach()),
                            "train_batch_image_mse": float(image_loss.detach()),
                            "train_batch_minimum_jacobian": margin})
    sync()
    train_seconds = time.perf_counter() - began
    allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    encoder.eval()
    final_train = evaluate(train)
    final_test = evaluate(test)
    if args.save_state:
        torch.save({"encoder": encoder.state_dict(), "config": vars(args)}, args.save_state)
    print(json.dumps({
        "method": "phase7_forward_p1_image_encoder",
        "decoder_kind": args.decoder_kind,
        "patch_cells": args.patch_cells if args.decoder_kind in ("patch", "hybrid") else None,
        "training_objective": "image_only_pixel_MSE" if args.training_objective == "image" else "target_query_map_vector_MSE",
        "target_map_use": "evaluation_only" if args.training_objective == "image" else "training_supervision_and_evaluation",
        "target_family": args.target_family,
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "image_side": args.image_side,
        "image_queries": args.image_side ** 2,
        "batch": args.batch,
        "train_count": args.train_count,
        "test_count": args.test_count,
        "train_seed": 55101,
        "test_seed": args.test_seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "adam_eps": args.adam_eps,
        "feature_side": encoder.feature_side,
        "first_step_last_two_level_head_gradient_max":
            first_step_fine_head_gradient_max,
        "seed_passes": args.seed_passes if args.decoder_kind in ("colored", "hybrid") else None,
        "seed_mechanism": "patch" if args.decoder_kind == "hybrid" else args.decoder_kind,
        "refinement_mechanism": "colored_new_vertex" if args.decoder_kind == "hybrid" else args.decoder_kind,
        "patch_passes_per_level": 4 if args.decoder_kind == "patch" else None,
        "width": args.width,
        "feedback_passes": args.feedback_passes,
        "feedback_side": feedback_side,
        "hint_window": args.hint_window,
        "hint_ridge": args.hint_ridge,
        "hint_gain": args.hint_gain,
        "hint_sine_modes": args.hint_sine_modes,
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "device": str(device),
        "torch_version": torch.__version__,
        "initial_test": initial,
        "final_train": final_train,
        "final_test": final_test,
        "history": history,
        "train_seconds": train_seconds,
        "median_training_step_seconds": statistics.median(times),
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "checkpoint_path": args.save_state,
        "loaded_checkpoint_path": args.load_state,
        "allow_new_levels": args.allow_new_levels,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
