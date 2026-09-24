"""Compute the fixed-grid P1 attenuation of a specified sinusoidal test mode.

This is an evaluation-only representation calculation; it does not read model
outputs or any true map during image-to-latent training.
"""
from __future__ import annotations

import json
import math

import torch

from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def mode(side: int, cycles: int) -> torch.Tensor:
    axis = torch.arange(side, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.sin(2 * math.pi * cycles * xx) * torch.sin(2 * math.pi * cycles * yy)


def main() -> None:
    query_mode = mode(512, 128)
    denominator = query_mode.square().sum()
    rows = []
    for side in (257, 513, 1025, 2049):
        table = StructuredDenseQueryTable.from_shape(
            side - 1, side - 1, height=512, width=512,
        )
        table.prepare(device=torch.device("cpu"), dtype=torch.float64)
        high = mode(side, 128)
        low = mode(side, 1)
        high_query = table.interpolate(
            torch.stack((high, high), dim=-1).reshape(1, side * side, 2)
        )[0, ..., 0]
        low_query = table.interpolate(
            torch.stack((low, low), dim=-1).reshape(1, side * side, 2)
        )[0, ..., 0]
        rows.append({
            "control_side": side,
            "image_query_side": 512,
            "high128_mode_transfer": float((high_query * query_mode).sum() / denominator),
            "low1_leakage_into_high128": float((low_query * query_mode).sum() / denominator),
        })
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
