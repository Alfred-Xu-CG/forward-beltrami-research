"""The measurement entry point must report an actual output and VJP."""

from __future__ import annotations

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
