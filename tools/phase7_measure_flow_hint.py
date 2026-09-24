"""Decisive pretraining check: does a local ridge-flow image cue track motion?"""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset
from qcopt.neural_bijection.dense.forward_p1_encoder import (
    ridge_local_flow_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-family", choices=("high32", "high64", "base"),
                        default="high32")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=939031)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    fixed, moving, truth = (
        item.to(device) for item in make_dataset(
            args.count, 512, args.seed,
            target_family=args.target_family,
        )
    )
    side = 257
    axis = torch.arange(side, device=device, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=0)[None]
    square_flow = square_zero = cross = truth_square = flow_square = 0.
    count = 0
    for start in range(0, args.count, args.batch):
        stop = min(start + args.batch, args.count)
        pair = F.interpolate(
            torch.cat((fixed[start:stop], moving[start:stop]), dim=1),
            size=(side, side), mode="bilinear", align_corners=True,
        )
        flow_feature = ridge_local_flow_features(
            pair[:, :1], pair[:, 1:],
        )
        flow = torch.atanh(flow_feature.clamp(-.999, .999)) * .02
        true_map = F.interpolate(
            truth[start:stop].permute(0, 3, 1, 2),
            size=(side, side), mode="bilinear", align_corners=True,
        )
        true_motion = true_map - identity
        inner = (slice(None), slice(None), slice(8, -8), slice(8, -8))
        estimated = flow[inner]
        true_motion = true_motion[inner]
        square_flow += float((estimated - true_motion).square().sum(dim=1).sum())
        square_zero += float(true_motion.square().sum(dim=1).sum())
        cross += float((estimated * true_motion).sum())
        flow_square += float(estimated.square().sum())
        truth_square += float(true_motion.square().sum())
        count += int(true_motion.shape[0] * true_motion.shape[2] * true_motion.shape[3])
    print(json.dumps(dict(
        target_family=args.target_family,
        count=args.count,
        image_side=512,
        feature_side=side,
        device=str(device),
        zero_vector_rmse=math.sqrt(square_zero / count),
        flow_vector_rmse=math.sqrt(square_flow / count),
        cosine=cross / math.sqrt(flow_square * truth_square),
        torch_version=torch.__version__,
    ), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
