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
    parser.add_argument("--side", type=int, default=257)
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
            feature_side=min(args.side, 257), width=16,
        ).to(device)
    else:
        decoder = ForwardPatchP1Pyramid(17, args.side, patch_cells=8,
                                        minimum_jacobian=0.05).to(device)
        encoder = PatchPyramidImageEncoder(
            17, decoder.level_sides,
            feature_side=min(args.side, 257), width=16,
        ).to(device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(saved["encoder"])
    encoder.eval()
    safe = SafeColoredVertexRelaxation(args.side, motion_mode="radial", raw_span=2).to(device)
    floor = torch.full((args.batch,), 0.05 / (args.side - 1) ** 2, device=device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float32)
    sums = [dict(image=0.0, map=0.0, minimum=float("inf")) for _ in range(args.passes + 1)]
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            fixed, moving, target = (x[start:start + args.batch] for x in dataset)
            seed, levels = encoder(fixed, moving)
            mapped = decoder(seed, levels)
            for step in range(args.passes + 1):
                query = table.interpolate(mapped.reshape(mapped.shape[0], -1, 2))
                warped = F.grid_sample(moving, 2 * query - 1, mode="bilinear",
                                       padding_mode="border", align_corners=True)
                sums[step]["image"] += float((warped - fixed).square().mean()) * fixed.shape[0]
                sums[step]["map"] += float((query.reshape_as(target) - target).square()
                                           .sum(dim=-1).mean()) * fixed.shape[0]
                sums[step]["minimum"] = min(sums[step]["minimum"], minimum_jacobian(mapped))
                if step == args.passes:
                    break
                proposal = local_photometric_logits(
                    fixed, moving, mapped, window=args.window,
                    ridge=args.ridge, raw_span=2,
                )
                if args.sine_modes:
                    proposal = spectralize_bounded_logits(
                        proposal, side=args.side, raw_span=2, count=args.sine_modes,
                    )
                mapped = safe(mapped, args.gain * proposal,
                              area_floor=floor[:mapped.shape[0]])
    print(json.dumps({
        "method": "phase7_frozen_forward_pyramid_plus_photometric_feedback",
        "decoder_kind": args.decoder_kind,
        "side": args.side,
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
            }
            for i, item in enumerate(sums)
        ],
        "note": "Frozen encoder, no additional training. Parameter choices are exploratory.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
