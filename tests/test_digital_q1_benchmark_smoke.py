"""The measurement entry point must report an actual output and VJP."""

from __future__ import annotations

import pytest
import torch
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.digital_q1_benchmark import run_benchmark


def test_small_cpu_benchmark_runs_both_q1_layers() -> None:
    for mode in ("f1", "f2"):
        result = run_benchmark(side=9, mode=mode, device="cpu", dtype="float64", repeats=1)
        assert result["side"] == 9
        assert result["control_vertices"] == 81
        assert result["cells"] == 64
        assert result["corner_min"] > 0
        assert result["nonpositive_corners"] == 0
        assert result["boundary_max_error"] == 0
        assert result["grad_finite"]
        assert result["grad_max_abs"] > 0
        assert result["forward_seconds_median"] > 0
        assert result["vjp_seconds_median"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_small_cuda_benchmark_places_layer_buffers_on_gpu() -> None:
    for mode in ("f1", "f2"):
        result = run_benchmark(side=9, mode=mode, device="cuda:0", repeats=1)
        assert result["nonpositive_corners"] == 0
        assert result["grad_finite"]


def test_saved_q1_map_is_reloaded_and_validated_on_d_drive() -> None:
    destination = Path(__file__).resolve().parents[1] / "docs" / "digital_topology_wsi"
    with TemporaryDirectory(dir=destination) as temporary:
        output = Path(temporary) / "map.npz"
        result = run_benchmark(
            side=9, mode="f1", device="cpu", dtype="float64",
            repeats=1, save_path=output,
        )
        assert output.is_file()
        with np.load(output) as arrays:
            assert arrays["vertices"].shape == (1, 9, 9, 2)
            assert arrays["boundary_reference"].shape == (1, 9, 9, 2)
        assert result["saved_map_valid"]
        assert result["saved_map_corner_min"] > 0
