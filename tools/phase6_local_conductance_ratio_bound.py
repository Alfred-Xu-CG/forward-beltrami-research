"""A necessary local weight-ratio bound for positive Tutte equilibrium."""

from __future__ import annotations

import argparse
import json
import math

import torch

from phase6_train_multisample_image import make_dataset
from qcopt.neural_bijection.dense.sine_pcg_tutte import _minimum_signed_area_ratio


def target_vertices(side: int, item: int, device: torch.device) -> tuple[torch.Tensor, tuple[float, float, float]]:
    _, _, _, coefficients = make_dataset(32, 512, 55101, target_family="high32", return_coefficients=True)
    ax, ay, af = (float(value) for value in coefficients[item])
    line = torch.linspace(0, 1, side, device=device, dtype=torch.float64)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    low = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    high = torch.sin(64 * math.pi * xx) * torch.sin(64 * math.pi * yy)
    return torch.stack((xx + ax * low + af * high, yy + ay * low + af * high), dim=-1), (ax, ay, af)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--item", type=int, required=True)
    parser.add_argument("--directions", type=int, default=360)
    parser.add_argument("--ratio", type=float, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.item < 32 or args.directions < 4 or args.ratio <= 1:
        raise ValueError("invalid item, direction count, or ratio")
    device = torch.device(args.device)
    mapped, coefficients = target_vertices(args.side, args.item, device)
    center = mapped[1:-1, 1:-1]
    neighbor_vectors = torch.stack((
        mapped[1:-1, :-2] - center,
        mapped[1:-1, 2:] - center,
        mapped[:-2, 1:-1] - center,
        mapped[2:, 1:-1] - center,
        mapped[:-2, :-2] - center,
        mapped[2:, 2:] - center,
    ), dim=-2)
    maximum = 0.0
    witness = None
    violating = torch.zeros((args.side - 2, args.side - 2), device=device, dtype=torch.bool)
    for index in range(args.directions):
        angle = math.pi * index / args.directions
        direction = neighbor_vectors.new_tensor((math.cos(angle), math.sin(angle)))
        projection = (neighbor_vectors * direction).sum(dim=-1)
        positive = projection.clamp_min(0).sum(dim=-1)
        negative = (-projection).clamp_min(0).sum(dim=-1)
        required = torch.maximum(
            positive / negative.clamp_min(torch.finfo(projection.dtype).tiny),
            negative / positive.clamp_min(torch.finfo(projection.dtype).tiny),
        )
        violating |= required > args.ratio + 1e-10
        value, flat_index = required.reshape(-1).max(dim=0)
        if float(value) > maximum:
            maximum = float(value)
            row, col = divmod(int(flat_index), args.side - 2)
            witness = {"interior_row": row + 1, "interior_column": col + 1,
                       "direction_index": index, "angle_radians": angle,
                       "positive_projection_sum": float(positive[row, col]),
                       "negative_projection_magnitude_sum": float(negative[row, col]),
                       "required_ratio": maximum}
    print(json.dumps({
        "control_side": args.side,
        "control_vertices": args.side**2,
        "control_faces": 2 * (args.side - 1)**2,
        "target_set_size": 32,
        "target_seed": 55101,
        "target_item": args.item,
        "target_coefficients": {"ax": coefficients[0], "ay": coefficients[1], "af": coefficients[2]},
        "target_sampled_p1_minimum_area_ratio": _minimum_signed_area_ratio(mapped[None]),
        "declared_conductance_maximum_ratio": args.ratio,
        "directions_checked": args.directions,
        "maximum_sampled_direction_local_required_ratio": maximum,
        "vertices_with_at_least_one_sampled_direction_exceeding_ratio": int(violating.sum()),
        "witness": witness,
        "device": str(device),
        "dtype": "float64",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
