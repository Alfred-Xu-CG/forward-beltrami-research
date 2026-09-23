"""Check that high128 tests actual control-grid detail, not image resolution."""

from __future__ import annotations

import argparse
import json
import math

import torch

from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import SineModeP1Refiner, exact_dyadic_p1_refine
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def target_vertices(side: int, coefficients: torch.Tensor,
                    target_family: str = "high128") -> torch.Tensor:
    axis = torch.arange(side, dtype=torch.float64,
                        device=coefficients.device) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xx, yy = xx[None], yy[None]
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    if target_family == "high128":
        modes = ((128, 128),)
        windows = (None,)
    elif target_family == "high128_tri":
        modes = ((128, 128), (128, 64), (64, 128))
        windows = (None,) * 3
    else:
        modes = ((128, 128),) * 16
        windows = tuple(
            (column / 4, (column + 1) / 4, row / 4, (row + 1) / 4)
            for row in range(4) for column in range(4)
        )
    fine_disp = torch.zeros_like(xx).expand(coefficients.shape[0], -1, -1).clone()
    for mode_index, (kx, ky) in enumerate(modes):
        wave = torch.sin(2 * math.pi * kx * xx) * torch.sin(2 * math.pi * ky * yy)
        wave = torch.where(wave.abs() < 1e-12, 0, wave)
        if windows[mode_index] is not None:
            xlo, xhi, ylo, yhi = windows[mode_index]
            tx = (xx - xlo) / (xhi - xlo)
            ty = (yy - ylo) / (yhi - ylo)
            wave = wave * torch.where((tx >= 0) & (tx <= 1),
                                      torch.sin(math.pi * tx).square(), 0.0)
            wave = wave * torch.where((ty >= 0) & (ty <= 1),
                                      torch.sin(math.pi * ty).square(), 0.0)
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
    parser.add_argument("--target-family", choices=("high128", "high128_tri", "high128_tiles"),
                        default="high128")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--decoder-oracle", choices=("colored", "patch"))
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
    oracle = None
    if args.decoder_oracle:
        if args.target_family == "high128":
            modes, windows = ((128, 128),), None
        elif args.target_family == "high128_tri":
            modes, windows = ((128, 128), (128, 64), (64, 128)), None
        else:
            modes = ((128, 128),) * 16
            windows = tuple(
                (column / 4, (column + 1) / 4, row / 4, (row + 1) / 4)
                for row in range(4) for column in range(4)
            )
        decoder = SineModeP1Refiner(
            257, 1025, cycles=modes, windows=windows,
            mechanism=args.decoder_oracle,
        ).to(device)
        squared, maximum, minimum = 0.0, 0.0, float("inf")
        representable_squared, representable_maximum = 0.0, 0.0
        with torch.no_grad():
            for start in range(0, args.count, 2):
                coeff = coefficients[start:start + 2]
                coarse = target_vertices(257, coeff, args.target_family)
                actual = decoder(coarse, coeff[:, 2:])
                expected = target_vertices(1025, coeff, args.target_family)
                low_only_coeff = coeff.clone()
                low_only_coeff[:, 2:] = 0
                low_only = target_vertices(1025, low_only_coeff,
                                           args.target_family)
                high_only = expected - low_only
                representable = exact_dyadic_p1_refine(
                    exact_dyadic_p1_refine(coarse)
                ) + high_only
                squared += float((actual - expected).square().sum())
                maximum = max(maximum, float((actual - expected).abs().amax()))
                representable_squared += float((actual - representable).square().sum())
                representable_maximum = max(
                    representable_maximum,
                    float((actual - representable).abs().amax()),
                )
                minimum = min(minimum, minimum_jacobian(actual))
        oracle = {
            "mechanism": args.decoder_oracle,
            "vertex_vector_rmse": math.sqrt(squared / (args.count * 1025**2)),
            "maximum_coordinate_error": maximum,
            "representable_vertex_vector_rmse": math.sqrt(
                representable_squared / (args.count * 1025**2)
            ),
            "representable_maximum_coordinate_error": representable_maximum,
            "minimum_jacobian": minimum,
        }
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
            (1 - 4 * math.pi * 0.025
             - 2 * 0.0002 * (256 * math.pi + 4 * math.pi)) ** 2
            if args.target_family == "high128_tiles" else
            1 - 2 * math.pi * (0.015 + 0.025)
            - 0.000375 * 2 * 256 * math.pi
            - 0.000375 * 0.02 * 2 * (2 * math.pi) * (256 * math.pi),
        "identity_image_mse": float((moving - fixed).square().mean()),
        "results": results,
        "decoder_oracle": oracle,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
