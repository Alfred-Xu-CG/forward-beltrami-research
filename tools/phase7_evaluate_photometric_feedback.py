"""Diagnostic frozen image feedback after a forward P1 pyramid decoder."""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    ForwardP1ImageEncoder, ForwardP1Pyramid, ForwardPatchP1Pyramid,
    PatchPyramidImageEncoder, SafeColoredVertexRelaxation,
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder-kind", choices=("colored", "patch"), default="colored")
    parser.add_argument("--feedback-side", type=int, default=None)
    parser.add_argument("--allow-new-levels", action="store_true")
    parser.add_argument("--zero-fine-latents", action="store_true")
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--feature-side", type=int, default=None)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--target-family", default="high32")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99317)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1)
    parser.add_argument("--gain", type=float, default=1)
    parser.add_argument("--sine-modes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(20260924)
    device = torch.device(args.device)
    dataset = tuple(x.to(device) for x in make_dataset(
        args.count, args.image_side, args.seed, target_family=args.target_family,
    ))
    if args.decoder_kind == "colored":
        decoder = ForwardP1Pyramid(5, args.side, seed_passes=2,
                                   minimum_jacobian=0.05).to(device)
        encoder = ForwardP1ImageEncoder(
            5, decoder.level_sides, seed_passes=2,
            feature_side=args.feature_side or min(args.side, 257), width=16,
        ).to(device)
    else:
        decoder = ForwardPatchP1Pyramid(17, args.side, patch_cells=8,
                                        minimum_jacobian=0.05).to(device)
        encoder = PatchPyramidImageEncoder(
            17, decoder.level_sides,
            feature_side=args.feature_side or min(args.side, 257), width=16,
        ).to(device)
    feedback_side = args.feedback_side or args.side
    valid_sides = (decoder.seed_side,) + decoder.level_sides
    if feedback_side not in valid_sides:
        raise ValueError(f"feedback-side must be one of {valid_sides}")
    feedback_index = 0 if feedback_side == decoder.seed_side else decoder.level_sides.index(feedback_side) + 1
    if feedback_side < args.side:
        if args.decoder_kind == "colored":
            coarse_decoder = ForwardP1Pyramid(
                5, feedback_side, seed_passes=2, minimum_jacobian=0.05,
            ).to(device)
        else:
            coarse_decoder = ForwardPatchP1Pyramid(
                17, feedback_side, patch_cells=8, minimum_jacobian=0.05,
            ).to(device)
    else:
        coarse_decoder = None
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if args.allow_new_levels:
        missing, unexpected = encoder.load_state_dict(saved["encoder"], strict=False)
        if unexpected or any(not key.startswith("level_heads.") for key in missing):
            raise ValueError(f"incompatible transfer: missing={missing}, unexpected={unexpected}")
    else:
        encoder.load_state_dict(saved["encoder"])
    encoder.eval()
    safe = SafeColoredVertexRelaxation(feedback_side, motion_mode="radial", raw_span=2).to(device)
    floor = torch.full((args.batch,), 0.05 / (feedback_side - 1) ** 2, device=device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float32)
    sums = [
        dict(image=0.0, map=0.0, minimum=float("inf"),
             mode_error=0.0, mode_target=0.0, mode_predicted=0.0)
        for _ in range(args.passes + 1)
    ]
    if args.target_family == "high128":
        axis = torch.linspace(0, 1, args.image_side, device=device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        mode = torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)
        identity = torch.stack((xx, yy), dim=-1)
        mode_norm = mode.square().mean()
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            fixed, moving, target = (x[start:start + args.batch] for x in dataset)
            seed, levels = encoder(fixed, moving)
            if coarse_decoder is None:
                mapped = decoder(seed, levels)
                remaining = ()
            else:
                mapped = coarse_decoder(seed, levels[:feedback_index])
                remaining = levels[feedback_index:]
                if args.zero_fine_latents:
                    remaining = tuple(
                        tuple(torch.zeros_like(x) for x in group)
                        if args.decoder_kind == "patch" else torch.zeros_like(group)
                        for group in remaining
                    )
            for step in range(args.passes + 1):
                output = (
                    decoder.forward_from(mapped, remaining, start_index=feedback_index)
                    if coarse_decoder is not None else mapped
                )
                query = table.interpolate(output.reshape(output.shape[0], -1, 2))
                warped = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                                       padding_mode="border", align_corners=True)
                sums[step]["image"] += float((warped - fixed).square().mean()) * fixed.shape[0]
                sums[step]["map"] += float((query.reshape_as(target) - target).square()
                                           .sum(dim=-1).mean()) * fixed.shape[0]
                sums[step]["minimum"] = min(sums[step]["minimum"], minimum_jacobian(output))
                if args.target_family == "high128":
                    predicted = ((query.reshape_as(target) - identity) * mode[..., None]
                                 ).mean(dim=(1, 2)) / mode_norm
                    desired = ((target - identity) * mode[..., None]
                               ).mean(dim=(1, 2)) / mode_norm
                    sums[step]["mode_error"] += float(
                        (predicted - desired).square().sum(dim=-1).sum()
                    )
                    sums[step]["mode_target"] += float(desired.square().sum(dim=-1).sum())
                    sums[step]["mode_predicted"] += float(predicted.square().sum(dim=-1).sum())
                if step == args.passes:
                    break
                proposal = local_photometric_logits(
                    fixed, moving, mapped, window=args.window,
                    ridge=args.ridge, raw_span=2,
                )
                if args.sine_modes:
                    proposal = spectralize_bounded_logits(
                        proposal, side=feedback_side, raw_span=2, count=args.sine_modes,
                    )
                mapped = safe(mapped, args.gain * proposal,
                              area_floor=floor[:mapped.shape[0]])
    print(json.dumps({
        "method": "phase7_frozen_forward_pyramid_plus_photometric_feedback",
        "decoder_kind": args.decoder_kind,
        "side": args.side,
        "feature_side": encoder.feature_side,
        "feedback_side": feedback_side,
        "zero_fine_latents": args.zero_fine_latents,
        "image_side": args.image_side,
        "count": args.count,
        "seed": args.seed,
        "target_family": args.target_family,
        "batch": args.batch,
        "feedback_passes": args.passes,
        "window": args.window,
        "ridge": args.ridge,
        "gain": args.gain,
        "sine_modes": args.sine_modes,
        "device": args.device,
        "stages": [
            {
                "passes": i,
                "image_mse": item["image"] / args.count,
                "query_map_vector_rmse": math.sqrt(item["map"] / args.count),
                "minimum_jacobian": item["minimum"],
                **({
                    "high128_mode_coefficient_vector_rmse":
                        math.sqrt(item["mode_error"] / args.count),
                    "high128_target_coefficient_vector_rms":
                        math.sqrt(item["mode_target"] / args.count),
                    "high128_predicted_coefficient_vector_rms":
                        math.sqrt(item["mode_predicted"] / args.count),
                } if args.target_family == "high128" else {}),
            }
            for i, item in enumerate(sums)
        ],
        "note": "Frozen encoder, no additional training. Parameter choices are exploratory.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
