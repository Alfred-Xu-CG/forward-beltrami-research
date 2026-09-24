"""High-frequency P1 interpolation is measured on the actual fixed mesh."""
from __future__ import annotations

import torch

from phase7_high128_p1_mode_transfer import mode
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def test_high128_fixed_p1_transfer_is_not_raw_image_resolution() -> None:
    query_mode = mode(512, 128)
    denominator = query_mode.square().sum()
    ratios = []
    for side in (257, 513, 1025):
        table = StructuredDenseQueryTable.from_shape(
            side - 1, side - 1, height=512, width=512,
        )
        table.prepare(device=torch.device("cpu"), dtype=torch.float64)
        high = mode(side, 128)
        sampled = table.interpolate(
            torch.stack((high, high), dim=-1).reshape(1, side * side, 2)
        )[0, ..., 0]
        ratios.append(float((sampled * query_mode).sum() / denominator))
    assert abs(ratios[0]) < 1e-10
    assert abs(ratios[1] - 0.66329706757) < 1e-8
    assert abs(ratios[2] - 0.90230822377) < 1e-8
