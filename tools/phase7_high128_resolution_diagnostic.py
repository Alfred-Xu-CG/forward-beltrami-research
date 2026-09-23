"""Check that high128 tests actual control-grid detail, not image resolution."""

from __future__ import annotations

import argparse
import json
import math

import torch

from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def target_vertices(side: int, coefficients: torch.Tensor,
                    target_family: str = "high128") -> torch.Tensor:
    axis = torch.arange(side, dtype=torch.float64,
                        device=coefficients.device) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xx, yy = xx[None], yy[None]
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    modes = ((128, 128),) if target_family == "high128" else (
        (128, 128), (128, 64), (64, 128)
    )
    fine_disp = torch.zeros_like(xx).expand(coefficients.shape[0], -1, -1).clone()
    for mode_index, (kx, ky) in enumerate(modes):
        wave = torch.sin(2 * math.pi * kx * xx) * torch.sin(2 * math.pi * ky * yy)
        wave = torch.where(wave.abs() < 1e-12, 0, wave)
        fine_disp += coefficients[:, mode_index + 2, None, None] * wave
    ax, ay = (coefficients[:, i, None, None] for i in range(2))
    return torch.stack((xx + ax * bump + fine_disp,
                        yy + ay * bump + fine_disp), dim=-1).to(coefficients.dtype)


def minimum_jacobian(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]

    def cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    return float(torch.minimum(
        cross(b - a, c - a).amin(),
        cross(c - a, d - a).amin(),
    ) * (mapped.shape[1] - 1) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=99317)
    parser.add_argument("--target-family", choices=("high128", "high128_tri"),
                        default="high128")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    fixed, moving, target, coefficients = make_dataset(
        args.count, args.image_side, args.seed,
        target_family=args.target_family, return_coefficients=True,
    )
    target, coefficients = target.to(device), coefficients.to(device)
    results = []
    for side in (257, 1025):
        mapped = target_vertices(side, coefficients, args.target_family)
        table = StructuredDenseQueryTable.from_mesh(
            structured_rectangle(side - 1, side - 1),
            height=args.image_side, width=args.image_side,
        )
        table.prepare(device=device, dtype=torch.float32)
        query = table.interpolate(mapped.reshape(args.count, -1, 2)).reshape_as(target)
        results.append({
            "control_side": side,
            "control_vertices": side ** 2,
            "control_faces": 2 * (side - 1) ** 2,
            "minimum_jacobian": minimum_jacobian(mapped),
            "query_map_vector_rmse": float((query - target).square()
                                           .sum(dim=-1).mean().sqrt()),
            "maximum_coordinate_error": float((query - target).abs().amax()),
        })
    print(json.dumps({
        "method": "phase7_high128_control_resolution_diagnostic",
        "count": args.count,
        "seed": args.seed,
        "image_side": args.image_side,
        "image_queries": args.image_side ** 2,
        "device": str(device),
        "target_family": args.target_family,
        "fine_coefficient_range": [
            float(coefficients[:, 2:].amin()),
            float(coefficients[:, 2:].amax()),
        ],
        "all_coefficient_box_p1_jacobian_lower_bound":
            (1 - 4 * math.pi * 0.025
             - 4 * math.pi * 0.0001 * (128 + 128 + 64)) ** 2
            if args.target_family == "high128_tri" else
            1 - 2 * math.pi * (0.015 + 0.025)
            - 0.000375 * 2 * 256 * math.pi
            - 0.000375 * 0.02 * 2 * (2 * math.pi) * (256 * math.pi),
        "identity_image_mse": float((moving - fixed).square().mean()),
        "results": results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
