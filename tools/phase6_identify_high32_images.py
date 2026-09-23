"""Image-only identifiability diagnostic for the declared synthetic high32 family.

This uses the known three-dimensional synthetic basis, so it is deliberately
not a generic neural registration layer. Target coefficients enter evaluation
only, never the estimation objective.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset


def map_from_coefficients(coefficients: torch.Tensor, side: int) -> torch.Tensor:
    line = torch.linspace(0.0, 1.0, side, dtype=coefficients.dtype, device=coefficients.device)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    fine = torch.sin(64 * math.pi * xx) * torch.sin(64 * math.pi * yy)
    x = xx[None] + coefficients[:, 0, None, None] * bump + coefficients[:, 2, None, None] * fine
    y = yy[None] + coefficients[:, 1, None, None] * bump + coefficients[:, 2, None, None] * fine
    return torch.stack((x, y), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--refine-steps", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    fixed, moving, true_map, true_coefficients = make_dataset(
        args.test_count, args.image_side, args.test_seed, return_coefficients=True, target_family="high32"
    )
    scale = (args.image_side - 1) / 2
    image = moving[:, 0]
    gx = scale * (image[:, 1:-1, 2:] - image[:, 1:-1, :-2])
    gy = scale * (image[:, 2:, 1:-1] - image[:, :-2, 1:-1])
    line = torch.linspace(0.0, 1.0, args.image_side)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    bump = torch.sin(2 * math.pi * xx[1:-1, 1:-1]) * torch.sin(2 * math.pi * yy[1:-1, 1:-1])
    fine = torch.sin(64 * math.pi * xx[1:-1, 1:-1]) * torch.sin(64 * math.pi * yy[1:-1, 1:-1])
    design = torch.stack((gx * bump, gy * bump, (gx + gy) * fine), dim=-1).reshape(args.test_count, -1, 3)
    residual = (fixed[:, 0, 1:-1, 1:-1] - image[:, 1:-1, 1:-1]).reshape(args.test_count, -1, 1)
    gram = design.transpose(1, 2) @ design
    rhs = design.transpose(1, 2) @ residual
    linear = torch.linalg.solve(gram, rhs)[..., 0]
    estimated = linear
    if args.refine_steps:
        estimated = torch.nn.Parameter(linear.clone())
        optimizer = torch.optim.Adam([estimated], lr=0.0005)
        for _ in range(args.refine_steps):
            optimizer.zero_grad(set_to_none=True)
            predicted = map_from_coefficients(estimated, args.image_side)
            warped = F.grid_sample(moving, 2 * predicted - 1, mode="bilinear", padding_mode="border", align_corners=True)
            (warped - fixed).square().mean().backward()
            optimizer.step()
        estimated = estimated.detach()
    predicted_map = map_from_coefficients(estimated, args.image_side)
    warped = F.grid_sample(moving, 2 * predicted_map - 1, mode="bilinear", padding_mode="border", align_corners=True)
    print(json.dumps({
        "question": "synthetic_high32_image_identifiability_not_generic_registration",
        "image_side": args.image_side,
        "test_count": args.test_count,
        "test_seed": args.test_seed,
        "refine_steps": args.refine_steps,
        "linear_estimated_coefficients": linear.tolist(),
        "final_estimated_coefficients": estimated.tolist(),
        "true_coefficients_evaluation_only": true_coefficients.tolist(),
        "gram_condition_numbers": torch.linalg.cond(gram).tolist(),
        "fine_coefficient_rmse": (estimated[:, 2] - true_coefficients[:, 2]).square().mean().sqrt().item(),
        "all_coefficients_rmse": (estimated - true_coefficients).square().mean().sqrt().item(),
        "image_mse": (warped - fixed).square().mean().item(),
        "map_coordinate_rmse": (predicted_map - true_map).square().mean().sqrt().item(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
