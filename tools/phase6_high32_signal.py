"""Quantify the image-only loss left by omitting the known fine target term.

This diagnostic uses true coefficients only for constructing an evaluation
baseline. It is not a trainable model or a source of supervision.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=8)
    args = parser.parse_args()
    fixed, moving, true_map, coefficients = make_dataset(
        args.test_count, args.image_side, 99317, return_coefficients=True, target_family="high32"
    )
    line = torch.linspace(0.0, 1.0, args.image_side)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    low_map = torch.stack((
        xx[None] + coefficients[:, 0, None, None] * bump,
        yy[None] + coefficients[:, 1, None, None] * bump,
    ), dim=-1)
    identity = torch.stack((xx, yy), dim=-1)[None].expand(args.test_count, -1, -1, -1)
    warped_low = F.grid_sample(moving, 2 * low_map - 1, mode="bilinear", padding_mode="border", align_corners=True)
    warped_identity = F.grid_sample(moving, 2 * identity - 1, mode="bilinear", padding_mode="border", align_corners=True)
    per_sample_low_image_mse = (warped_low - fixed).square().mean(dim=(1, 2, 3))
    print(json.dumps({
        "question": "image-only signal of the omitted 32-cycle target component",
        "target_family": "high32",
        "test_seed": 99317,
        "test_count": args.test_count,
        "image_side": args.image_side,
        "image_queries": args.image_side**2,
        "identity_image_mse": (warped_identity - fixed).square().mean().item(),
        "known_low_only_image_mse": per_sample_low_image_mse.mean().item(),
        "known_low_only_query_map_rmse": (low_map - true_map).square().mean().sqrt().item(),
        "per_sample": [
            {"true_fine_amplitude": coefficients[index, 2].item(), "low_only_image_mse": per_sample_low_image_mse[index].item()}
            for index in range(args.test_count)
        ],
        "warning": "The low-only baseline uses target coefficients for diagnosis and is not an image-inferred method.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
