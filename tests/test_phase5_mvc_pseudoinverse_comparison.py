"""Contract tests for the explicit three-way Route-II latent-lift comparison."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

# Initialize the NumPy/SciPy-compatible OpenMP runtime before Torch on Windows.
np.linalg.solve(np.eye(1), np.ones(1))


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/phase5/route2_mvc_pseudoinverse_comparison.py"
)


def _module():
    assert SCRIPT.is_file(), "Route-II pseudoinverse comparison is missing"
    spec = importlib.util.spec_from_file_location(
        "phase5_route2_mvc_pseudoinverse_comparison", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_complete_fixed_boundary_common_gauge_comparison() -> None:
    receipt = _module().run_comparison()

    assert receipt["status"] == "ok", receipt
    assert receipt["schema"] == "phase5_route2_mvc_pseudoinverse_comparison_v1"
    mesh = receipt["mesh"]
    jacobian = receipt["jacobian"]
    assert jacobian["shape"] == [2 * mesh["vertices"], mesh["gauge_dimension"]]
    assert (
        jacobian["rank"]
        == jacobian["expected_fixed_boundary_rank"]
        == 2 * mesh["interior"]
    )
    assert jacobian["rank_tolerance"] > 0.0
    assert jacobian["smallest_retained_singular_value"] > jacobian["rank_tolerance"]
    assert jacobian["largest_discarded_singular_value"] < jacobian["rank_tolerance"]
    assert jacobian["boundary_row_maximum_absolute"] == 0.0
    assert jacobian["finite_difference_relative_frobenius_error"] < 2.0e-8

    gauge = receipt["gauge"]
    assert gauge["name"] == "supported arithmetic-zero-row-mean logits"
    assert gauge["basis_orthogonality_maximum_absolute_error"] < 3.0e-16
    assert gauge["basis_row_sum_maximum_absolute"] < 3.0e-16
    assert gauge["base_logit_row_mean_maximum_absolute"] < 3.0e-16
    assert receipt["direction"]["boundary_maximum_absolute"] == 0.0

    for name in ("moore_penrose", "mvc_encoder_derivative", "covariance_common_gauge"):
        lift = receipt["lifts"][name]
        assert lift["unsupported_maximum_absolute"] == 0.0
        assert lift["supported_row_sum_maximum_absolute"] < 1.0e-14
        assert lift["gauge_projection_l2_residual"] < 2.0e-14
        assert lift["decoded_tangent_l2_residual"] < 2.0e-12
        assert lift["decoded_tangent_relative_l2_residual"] < 2.0e-11
        assert lift["finite_difference_tangent_relative_l2_residual"] < 2.0e-8

    lifts = receipt["lifts"]
    assert lifts["moore_penrose"]["euclidean_norm"] <= (
        lifts["mvc_encoder_derivative"]["euclidean_norm"] + 1.0e-13
    )
    assert lifts["moore_penrose"]["euclidean_norm"] <= (
        lifts["covariance_common_gauge"]["euclidean_norm"] + 1.0e-13
    )

    for pair in (
        "mvc_encoder_derivative_minus_moore_penrose",
        "covariance_common_gauge_minus_moore_penrose",
        "mvc_encoder_derivative_minus_covariance_common_gauge",
    ):
        difference = receipt["pairwise_differences"][pair]
        assert difference["euclidean_norm"] > 1.0e-5
        assert difference["kernel_l2_residual"] < 3.0e-12
        assert difference["kernel_relative_l2_residual"] < 3.0e-11

    json.dumps(receipt, allow_nan=False)


def test_native_covariance_metric_gauge_is_reported_separately() -> None:
    receipt = _module().run_comparison()
    lifts = receipt["lifts"]
    native = lifts["covariance_native_probability_gauge"]
    common = lifts["covariance_common_gauge"]

    assert native["probability_weighted_row_mean_maximum_absolute"] < 2.0e-16
    assert native["supported_row_sum_maximum_absolute"] > 1.0e-5
    assert native["weighted_norm"] < common["weighted_norm"]
    assert native["weighted_norm"] <= lifts["moore_penrose"]["weighted_norm"] + 1.0e-13
    assert native["weighted_norm"] <= (
        lifts["mvc_encoder_derivative"]["weighted_norm"] + 1.0e-13
    )
    assert native["explicit_weighted_pseudoinverse_l2_difference"] < 3.0e-12
    assert native["explicit_weighted_pseudoinverse_weighted_norm"] == pytest.approx(
        native["weighted_norm"], abs=3.0e-13, rel=3.0e-12
    )
    assert (
        native["explicit_weighted_pseudoinverse_rank"]
        == 2 * receipt["mesh"]["interior"]
    )
    assert native["explicit_weighted_pseudoinverse_decoded_l2_residual"] < 3.0e-12
    assert receipt["optimality_checks"][
        "moore_penrose_is_euclidean_minimum_in_common_gauge"
    ]
    assert receipt["optimality_checks"][
        "native_covariance_is_weighted_minimum_over_representatives"
    ]


def test_cli_writes_one_finite_compact_receipt(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"
    clean_environment = os.environ.copy()
    for name in ("MKL_THREADING_LAYER", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        clean_environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
        env=clean_environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(output)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "ok"
    json.dumps(receipt, allow_nan=False)
