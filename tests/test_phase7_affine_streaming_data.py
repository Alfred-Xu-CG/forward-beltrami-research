"""Streamed synthetic affine data must match the resident reference."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from phase7_train_local_affine_residual import (  # noqa: E402
    affine_dataset, affine_dataset_cpu_stream,
)


def test_affine_streaming_data_matches_reference() -> None:
    device = torch.device("cpu")
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024),
        height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    reference = affine_dataset(2, 314159, device, 1, table)
    streamed = affine_dataset_cpu_stream(
        2, 314159, device, 1, table, with_masks=True)
    for left, right in zip(reference, streamed):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
