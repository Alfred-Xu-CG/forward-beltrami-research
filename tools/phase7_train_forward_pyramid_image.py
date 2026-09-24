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
from qcopt.neural_bijection.dense import (
    ForwardP1ImageEncoder, ForwardP1Pyramid, ForwardPatchP1Pyramid,
    HybridPatchSeedVertexP1Pyramid, PatchPyramidImageEncoder,
    SafeColoredVertexRelaxation, certify_p1_or_identity,
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


def replace_test_appearance(
    dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mode: str, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the exact test maps and change only unseen moving-image content."""
    if mode == "standard":
        return dataset
    fixed, moving, true_map = dataset
    count, _, side, _ = moving.shape
    axis = torch.linspace(0, 1, side, device=moving.device, dtype=moving.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    random = torch.Generator(device="cpu").manual_seed(seed + 701129)
    if mode == "spots":
        centers = (.10 + .80 * torch.rand(count, 6, 2, generator=random)).to(moving.device)
        strengths = (2 * torch.rand(count, 6, generator=random) - 1).to(moving.device)
        texture = .08 * (xx + .3 * yy)[None].expand(count, -1, -1).clone()
        for k in range(6):
            radius = (xx - centers[:, k, 0, None, None]).square()
            radius += (yy - centers[:, k, 1, None, None]).square()
            texture += .40 * strengths[:, k, None, None] * torch.exp(-radius / .012)
    elif mode == "crosswaves":
        phase = (2 * math.pi * torch.rand(count, 4, generator=random)).to(moving.device)
        texture = (
            .29 * torch.sin(5 * math.pi * xx + 17 * math.pi * yy + phase[:, 0, None, None])
            + .24 * torch.cos(19 * math.pi * xx - 7 * math.pi * yy + phase[:, 1, None, None])
            + .19 * torch.sin(27 * math.pi * xx + 11 * math.pi * yy + phase[:, 2, None, None])
            + .16 * torch.cos(9 * math.pi * xx - 29 * math.pi * yy + phase[:, 3, None, None])
        )
    else:
        raise ValueError(f"unknown test appearance: {mode}")
    shifted_moving = texture[:, None].contiguous()
    shifted_fixed = F.grid_sample(
        shifted_moving, 2 * true_map - 1,
        mode="bilinear", padding_mode="border", align_corners=True,
    ).detach()
    return shifted_fixed, shifted_moving, true_map


def sinusoidal_mode_coefficients(
    mapped: torch.Tensor, cycles: int,
) -> torch.Tensor:
    """Project each two-component query displacement onto a known sine mode."""
    if mapped.ndim != 4 or mapped.shape[-1] != 2 or cycles < 1:
        raise ValueError("mapped must be BHWC2 and cycles must be positive")
    height, width = mapped.shape[1:3]
    xx = torch.linspace(0, 1, width, device=mapped.device, dtype=mapped.dtype)
    yy = torch.linspace(0, 1, height, device=mapped.device, dtype=mapped.dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    phi = torch.sin(2 * math.pi * cycles * gx) * torch.sin(2 * math.pi * cycles * gy)
    identity = torch.stack((gx, gy), dim=-1)
    return ((mapped - identity) * phi[None, ..., None]).sum(dim=(1, 2)) / phi.square().sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--decoder-kind", choices=("colored", "patch", "hybrid"), default="colored")
    parser.add_argument("--checkpoint-levels", action="store_true",
                        help="Recompute hybrid F1 refinement passes during VJP to reduce saved activations.")
    parser.add_argument("--patch-cells", type=int, default=8)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--target-family", choices=("base", "high32", "high64", "high128"), default="high32")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--train-appearance-augmentation",
                        choices=("none", "crosswaves", "spots", "all"), default="none",
                        help="Reuse each training map with additional image appearances.")
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--test-appearance", choices=("standard", "spots", "crosswaves"),
                        default="standard", help="Change test texture only; preserve test maps and training data.")
    parser.add_argument("--seed-passes", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--flow-hint", action="store_true",
                        help="Append differentiable local ridge-flow image features to the CNN input.")
    parser.add_argument("--flow-feature-gain", type=float, default=1.0,
                        help="Inference ablation of the appended flow feature; zero masks its channels.")
    parser.add_argument("--flow-feature-dropout", type=float, default=0.0,
                        help="Training-only whole-batch probability of masking the flow channels.")
    parser.add_argument("--feature-side", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--training-objective", choices=("image", "map"), default="image")
    parser.add_argument("--high128-mode-weight", type=float, default=0.0,
                        help="Oracle diagnostic: add supervised high128 projection error to the image loss.")
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
    parser.add_argument("--train-new-levels-only", action="store_true",
                        help="In an expanding checkpoint transfer, freeze every existing encoder parameter.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--certify-output", action="store_true",
                        help="Apply the same final-coordinate P1 sign filter to non-hybrid decoders.")
    args = parser.parse_args()
    if min(args.side, args.image_side, args.train_count, args.test_count,
           args.width, args.batch, args.steps) < 1 or (
               args.feature_side is not None and args.feature_side < 17
           ) or args.adam_eps <= 0:
        raise ValueError("dimensions, counts, batch and steps must be positive")
    if args.decoder_kind == "hybrid" and args.feedback_passes:
        raise ValueError("hybrid decoder has no intermediate feedback path")
    if args.checkpoint_levels and args.decoder_kind != "hybrid":
        raise ValueError("--checkpoint-levels currently applies only to hybrid")
    if args.high128_mode_weight < 0 or not math.isfinite(args.high128_mode_weight) or (
        args.high128_mode_weight and (
            args.target_family != "high128" or args.training_objective != "image"
        )
    ):
        raise ValueError("high128 mode supervision requires a finite nonnegative weight, high128 data and image training")
    if not math.isfinite(args.flow_feature_gain):
        raise ValueError("--flow-feature-gain must be finite")
    if not 0 <= args.flow_feature_dropout < 1 or (args.flow_feature_dropout and not args.flow_hint):
        raise ValueError("flow-feature-dropout needs --flow-hint and a probability in [0, 1)")
    if args.train_new_levels_only and not (args.load_state and args.allow_new_levels):
        raise ValueError("--train-new-levels-only requires --load-state and --allow-new-levels")
    torch.manual_seed(20260924)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    train = tuple(t.to(device) for t in make_dataset(
        args.train_count, args.image_side, 55101, target_family=args.target_family,
    ))
    if args.train_appearance_augmentation != "none":
        modes = (
            ("crosswaves", "spots") if args.train_appearance_augmentation == "all"
            else (args.train_appearance_augmentation,)
        )
        variants = (train,) + tuple(
            replace_test_appearance(train, mode, 55101) for mode in modes
        )
        train = tuple(torch.cat([variant[k] for variant in variants], dim=0)
                      for k in range(3))
    effective_train_pairs = train[0].shape[0]
    test = tuple(t.to(device) for t in make_dataset(
        args.test_count, args.image_side, args.test_seed,
        target_family=args.target_family,
    ))
    test = replace_test_appearance(test, args.test_appearance, args.test_seed)
    if args.decoder_kind == "colored":
        decoder = ForwardP1Pyramid(5, args.side, seed_passes=args.seed_passes,
                                   minimum_jacobian=0.05).to(device)
        encoder = ForwardP1ImageEncoder(
            5, decoder.level_sides, seed_passes=args.seed_passes,
            feature_side=args.feature_side or min(args.side, 257), width=args.width,
            flow_hint=args.flow_hint,
        ).to(device)
    elif args.decoder_kind == "patch":
        decoder = ForwardPatchP1Pyramid(
            17, args.side, patch_cells=args.patch_cells,
            minimum_jacobian=0.05,
        ).to(device)
        encoder = PatchPyramidImageEncoder(
            17, decoder.level_sides,
            feature_side=args.feature_side or min(args.side, 257), width=args.width,
            flow_hint=args.flow_hint,
        ).to(device)
    else:
        decoder = HybridPatchSeedVertexP1Pyramid(
            17, args.side, patch_cells=args.patch_cells,
            seed_cycles=args.seed_passes,
            minimum_jacobian=.05,
            checkpoint_levels=args.checkpoint_levels,
            compute_dtype=torch.float64,
        ).to(device)
        encoder = ForwardP1ImageEncoder(
            17, decoder.level_sides, seed_passes=1,
            feature_side=args.feature_side or min(args.side, 257),
            width=args.width,
            flow_hint=args.flow_hint,
        ).to(device)
    encoder.flow_feature_gain = args.flow_feature_gain
    if args.certify_output and args.decoder_kind != "hybrid":
        identity_axis = torch.arange(
            args.side, device=device, dtype=torch.float32,
        ) / (args.side - 1)
        identity_y, identity_x = torch.meshgrid(
            identity_axis, identity_axis, indexing="ij",
        )
        output_identity = torch.stack(
            (identity_x, identity_y), dim=-1,
        )[None]
    else:
        output_identity = (
            decoder.final_identity if args.decoder_kind == "hybrid"
            else None
        )
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
    if args.train_new_levels_only:
        new_parameter_names = set(missing)
        for name, parameter in encoder.named_parameters():
            parameter.requires_grad_(name in new_parameter_names)
    table = StructuredDenseQueryTable.from_shape(
        args.side - 1, args.side - 1,
        height=args.image_side, width=args.image_side,
    )
    table.prepare(
        device=device,
        dtype=(torch.float64 if args.decoder_kind == "hybrid"
               else torch.float32),
    )
    optimizer = torch.optim.Adam((p for p in encoder.parameters() if p.requires_grad), lr=args.learning_rate,
                                 eps=args.adam_eps)
    generator = torch.Generator(device="cpu").manual_seed(6019)
    dropout_generator = torch.Generator(device="cpu").manual_seed(20260924)

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
        if args.certify_output and args.decoder_kind != "hybrid":
            control, _ = certify_p1_or_identity(
                control, output_identity,
            )
        query = table.interpolate(control.reshape(control.shape[0], -1, 2)).float()
        warped = F.grid_sample(moving, 2 * query - 1,
                               mode="bilinear", padding_mode="border", align_corners=True)
        image_loss = (warped - fixed).square().mean()
        return image_loss, control, query

    @torch.no_grad()
    def evaluate(dataset: tuple[torch.Tensor, ...]) -> dict[str, float]:
        image_errors, map_errors, margins = [], [], []
        mode_errors, mode_predictions, mode_truths = [], [], []
        identity_outputs = 0
        for start in range(0, dataset[0].shape[0], args.batch):
            fixed, moving, true_map = (t[start:start + args.batch] for t in dataset)
            loss, control, query = forward(fixed, moving)
            image_errors.append(float(loss) * fixed.shape[0])
            map_errors.append(float((query.reshape_as(true_map) - true_map).square().sum(dim=-1).mean()) * fixed.shape[0])
            if args.target_family == "high128":
                predicted_mode = sinusoidal_mode_coefficients(query, 128)
                true_mode = sinusoidal_mode_coefficients(true_map, 128)
                mode_errors.append(float((predicted_mode - true_mode).square().sum(dim=-1).sum()))
                mode_predictions.append(float(predicted_mode.square().sum(dim=-1).sum()))
                mode_truths.append(float(true_mode.square().sum(dim=-1).sum()))
            margins.append(minimum_jacobian(control))
            if output_identity is not None:
                identity_outputs += int(torch.all(
                    control == output_identity,
                    dim=(1, 2, 3),
                ).sum())
        count = dataset[0].shape[0]
        result = {
            "image_mse": sum(image_errors) / count,
            "query_map_vector_rmse": math.sqrt(sum(map_errors) / count),
            "minimum_jacobian": min(margins),
            "identity_outputs": identity_outputs if output_identity is not None else None,
        }
        if mode_errors:
            result.update({
                "high128_mode_vector_rmse": math.sqrt(sum(mode_errors) / count),
                "high128_predicted_mode_rms": math.sqrt(sum(mode_predictions) / count),
                "high128_true_mode_rms": math.sqrt(sum(mode_truths) / count),
            })
        return result

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
            "test_appearance": args.test_appearance,
            "batch": args.batch,
            "load_state": args.load_state,
            "flow_hint": args.flow_hint,
            "flow_feature_gain": args.flow_feature_gain,
            "certify_output": args.certify_output or args.decoder_kind == "hybrid",
            "checkpoint_levels": args.checkpoint_levels,
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
        idx = torch.randint(effective_train_pairs, (args.batch,), generator=generator).to(device)
        fixed, moving, true_map = (tensor[idx] for tensor in train)
        if args.flow_feature_dropout:
            encoder.flow_feature_gain = (
                0.0 if float(torch.rand((), generator=dropout_generator)) < args.flow_feature_dropout
                else args.flow_feature_gain
            )
        optimizer.zero_grad(set_to_none=True)
        sync()
        tick = time.perf_counter()
        image_loss, control, query = forward(fixed, moving)
        loss = (
            image_loss if args.training_objective == "image"
            else (query.reshape_as(true_map) - true_map).square().sum(dim=-1).mean()
        )
        if args.high128_mode_weight:
            predicted_mode = sinusoidal_mode_coefficients(query, 128)
            true_mode = sinusoidal_mode_coefficients(true_map, 128)
            loss = loss + args.high128_mode_weight * (
                predicted_mode - true_mode
            ).square().sum(dim=-1).mean()
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
    encoder.flow_feature_gain = args.flow_feature_gain
    encoder.eval()
    final_train = evaluate(train)
    final_test = evaluate(test)
    if args.save_state:
        torch.save({"encoder": encoder.state_dict(), "config": vars(args)}, args.save_state)
    print(json.dumps({
        "method": "phase7_forward_p1_image_encoder",
        "decoder_kind": args.decoder_kind,
        "patch_cells": args.patch_cells if args.decoder_kind in ("patch", "hybrid") else None,
        "training_objective": (
            "image_pixel_MSE_plus_oracle_high128_mode" if args.high128_mode_weight else
            "image_only_pixel_MSE" if args.training_objective == "image" else
            "target_query_map_vector_MSE"
        ),
        "high128_mode_weight": args.high128_mode_weight,
        "flow_hint": args.flow_hint,
        "flow_feature_gain": args.flow_feature_gain,
        "flow_feature_dropout": args.flow_feature_dropout,
        "certify_output": args.certify_output or args.decoder_kind == "hybrid",
        "checkpoint_levels": args.checkpoint_levels,
        "target_map_use": (
            "mode_projection_supervision_and_evaluation" if args.high128_mode_weight else
            "evaluation_only" if args.training_objective == "image" else
            "training_supervision_and_evaluation"
        ),
        "target_family": args.target_family,
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "image_side": args.image_side,
        "image_queries": args.image_side ** 2,
        "batch": args.batch,
        "train_count": args.train_count,
        "train_appearance_augmentation": args.train_appearance_augmentation,
        "effective_train_pairs": effective_train_pairs,
        "test_count": args.test_count,
        "train_seed": 55101,
        "test_seed": args.test_seed,
        "test_appearance": args.test_appearance,
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
        "train_new_levels_only": args.train_new_levels_only,
        "trainable_encoder_parameters": sum(p.numel() for p in encoder.parameters() if p.requires_grad),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
