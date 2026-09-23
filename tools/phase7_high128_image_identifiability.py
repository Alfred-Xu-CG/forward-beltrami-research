"""Can images determine a high-frequency latent when the low map is supplied?"""

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
from qcopt.neural_bijection.dense.photometric_hint import physical_image_gradient
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=939031)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--decoder-kind", choices=("colored", "patch"), default="colored")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    fixed, moving, _, coefficients = make_dataset(
        args.count, args.image_side, args.seed,
        target_family="high128", return_coefficients=True,
    )
    axis = torch.linspace(0, 1, args.image_side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    high = torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)
    if args.checkpoint is not None:
        if args.decoder_kind == "colored":
            decoder = ForwardP1Pyramid(5, 257, seed_passes=2).to(device)
            encoder = ForwardP1ImageEncoder(
                5, decoder.level_sides, seed_passes=2, feature_side=257, width=16,
            ).to(device)
        else:
            decoder = ForwardPatchP1Pyramid(17, 257, patch_cells=8).to(device)
            encoder = PatchPyramidImageEncoder(
                17, decoder.level_sides, feature_side=257, width=16,
            ).to(device)
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(saved["encoder"])
        encoder.eval()
        safe = SafeColoredVertexRelaxation(257, motion_mode="radial", raw_span=2).to(device)
        table = StructuredDenseQueryTable.from_mesh(
            structured_rectangle(256, 256),
            height=args.image_side, width=args.image_side,
        )
        table.prepare(device=device, dtype=torch.float32)
    estimates, truths = [], []
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            f = fixed[start:start + args.batch].to(device)
            m = moving[start:start + args.batch].to(device)
            c = coefficients[start:start + args.batch].to(device)
            if args.checkpoint is None:
                low = torch.stack((
                    xx[None] + c[:, 0, None, None] * bump,
                    yy[None] + c[:, 1, None, None] * bump,
                ), dim=-1)
            else:
                seed, levels = encoder(f, m)
                control = decoder(seed, levels)
                for _ in range(2):
                    hint = local_photometric_logits(
                        f, m, control, window=3, ridge=1.0, raw_span=2.0,
                    )
                    hint = spectralize_bounded_logits(
                        hint, side=257, raw_span=2.0, count=16,
                    )
                    floor = control.new_full((control.shape[0],), 0.05 / 256**2)
                    control = safe(control, hint, area_floor=floor)
                low = table.interpolate(control.reshape(control.shape[0], -1, 2))
                low = low.reshape(-1, args.image_side, args.image_side, 2)
            grid = 2 * low - 1
            warped = F.grid_sample(m, grid, mode="bilinear",
                                   padding_mode="border", align_corners=True)
            gradient = F.grid_sample(
                physical_image_gradient(m), grid,
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            design = high * (gradient[:, 0] + gradient[:, 1])
            residual = (f - warped)[:, 0]
            estimate = (design * residual).mean(dim=(1, 2)) / (
                design.square().mean(dim=(1, 2)) + args.ridge
            )
            estimates.append(estimate.cpu())
            truths.append(c[:, 2].cpu())
    estimate = torch.cat(estimates)
    truth = torch.cat(truths)
    difference = estimate - truth
    print(json.dumps({
        "method": "phase7_high128_photometric_scalar_identifiability",
        "note": (
            "Uses exact low-frequency target coefficients as an oracle; fine coefficient is image-estimated."
            if args.checkpoint is None else
            "Uses a frozen image-trained 257-grid map plus safe feedback; fine coefficient is image-estimated."
        ),
        "count": args.count,
        "seed": args.seed,
        "image_side": args.image_side,
        "ridge": args.ridge,
        "device": str(device),
        "checkpoint": args.checkpoint,
        "decoder_kind": args.decoder_kind if args.checkpoint else None,
        "fine_coefficient_rms": float(truth.square().mean().sqrt()),
        "estimated_coefficient_rms": float(estimate.square().mean().sqrt()),
        "coefficient_rmse": float(difference.square().mean().sqrt()),
        "coefficient_bias": float(difference.mean()),
        "correlation": float(torch.corrcoef(torch.stack((estimate, truth)))[0, 1]),
        "maximum_absolute_error": float(difference.abs().amax()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
