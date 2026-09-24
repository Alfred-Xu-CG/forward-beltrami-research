"""Evaluate analytic high128 truth sampled as one fixed-grid P1 map."""
from __future__ import annotations

import argparse
import json
import math

import torch

from phase6_train_multisample_image import make_dataset
from phase7_train_forward_pyramid_image import minimum_jacobian
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, choices=(1025, 2049, 4097), required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20270215)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    _, _, true_map, coefficients = make_dataset(
        args.count, 512, args.seed,
        return_coefficients=True, target_family="high128",
    )
    true_map = true_map.to(device)
    coefficients = coefficients.to(device=device, dtype=torch.float64)
    axis = torch.arange(args.side, device=device, dtype=torch.float64) / (args.side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    fine = torch.sin(256 * math.pi * xx) * torch.sin(256 * math.pi * yy)
    table = StructuredDenseQueryTable.from_shape(
        args.side - 1, args.side - 1, height=512, width=512,
    )
    table.prepare(device=device, dtype=torch.float64)
    sum_squared = 0.0
    minimum_j = float("inf")
    for index in range(args.count):
        ax, ay, af = coefficients[index]
        controls = torch.stack((
            xx + ax * bump + af * fine,
            yy + ay * bump + af * fine,
        ), dim=-1)[None]
        query = table.interpolate(controls.flatten(1, 2)).float()
        sum_squared += float(
            (query - true_map[index:index + 1]).square().sum(dim=-1).mean()
        )
        minimum_j = min(minimum_j, minimum_jacobian(controls))
    print(json.dumps({
        "method": "analytic_high128_nodal_P1_teacher",
        "side": args.side,
        "count": args.count,
        "seed": args.seed,
        "image_queries": 512 ** 2,
        "query_map_rmse": (sum_squared / args.count) ** .5,
        "minimum_jacobian": minimum_j,
        "dtype": "float64",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
