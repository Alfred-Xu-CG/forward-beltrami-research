"""Evaluation-only first-order visibility of a specified fine deformation mode."""
from __future__ import annotations

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset
from phase7_train_forward_pyramid_image import replace_test_appearance
from qcopt.neural_bijection.dense.photometric_hint import physical_image_gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=939031)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.count, args.batch) < 1:
        raise ValueError("count and batch must be positive")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    original = tuple(t.to(device) for t in make_dataset(
        args.count, 512, args.seed, target_family="high128",
    ))
    axis = torch.arange(512, device=device, dtype=torch.float32) / 511
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    high = torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)
    identity = torch.stack((xx, yy), dim=-1)[None]
    report = {}
    for appearance in ("standard", "crosswaves", "spots"):
        data = replace_test_appearance(original, appearance, args.seed)
        sensitivity = []
        identity_image_mse = []
        for start in range(0, args.count, args.batch):
            fixed, moving, true_map = (
                tensor[start:start + args.batch] for tensor in data
            )
            gradient = physical_image_gradient(moving)
            sampled = F.grid_sample(
                gradient, 2 * true_map - 1,
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            derivative = high[None] * (sampled[:, 0] + sampled[:, 1])
            sensitivity.extend(
                derivative.square().mean(dim=(1, 2)).sqrt().tolist()
            )
            identity_image_mse.extend(
                (moving - fixed).square().mean(dim=(1, 2, 3)).tolist()
            )
        report[appearance] = {
            "mode_image_derivative_rms_mean": statistics.mean(sensitivity),
            "mode_image_derivative_rms_median": statistics.median(sensitivity),
            "identity_image_mse_mean": statistics.mean(identity_image_mse),
            "identity_map_rmse": float(
                (original[2] - identity).square().sum(dim=-1).mean().sqrt()
            ),
        }
    print(json.dumps({
        "method": "phase7_high128_texture_sensitivity",
        "count": args.count, "seed": args.seed, "image_side": 512,
        "derivative_definition": "RMS of H(q)*(grad_x+grad_y) I_moving(F(q))",
        "finite_difference_gradient": "central differences with replicated ends",
        "device": str(device), "results": report,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
