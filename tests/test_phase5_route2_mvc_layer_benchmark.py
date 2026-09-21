"""Receipt-contract tests for the compact Route-II M1/M2 layer benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


pytestmark = pytest.mark.filterwarnings("error")


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/phase5/route2_mvc_layer_benchmark.py"
)


def _module():
    assert SCRIPT.is_file(), "Route-II MVC layer benchmark is missing"
    spec = importlib.util.spec_from_file_location("phase5_route2_mvc_layer_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(**kwargs):
    module = _module()
    defaults = dict(
        control_side=3,
        batch=1,
        dtype="float64",
        device="cpu",
        backend="direct",
        warmup=0,
        repeats=1,
        threads=1,
        strength=0.08,
        step_size=0.04,
    )
    defaults.update(kwargs)
    return module.run_case(module.BenchmarkConfig(**defaults))


def test_tiny_cpu_receipt_contains_m1_m2_timing_accuracy_and_topology_contracts() -> None:
    receipt = _run()

    assert receipt["status"] == "ok", receipt
    assert receipt["schema"] == "phase5_route2_mvc_layer_case_v1"
    assert receipt["resolved_backend"] == "direct"
    assert receipt["shapes"]["control"] == [1, 9, 2]
    assert receipt["shapes"]["logits"][0] == 1
    for stage in ("encoder_only", "full_m1", "canonical_m2"):
        measured = receipt["timing"][stage]
        assert len(measured["samples"]) == 1
        for field in (
            "mean_forward_seconds",
            "mean_loss_seconds",
            "mean_backward_seconds",
            "mean_end_to_end_seconds",
        ):
            assert measured[field] >= 0.0
        assert measured["gradients_finite"]
        assert measured["gpu_memory"] is None

    audit = receipt["audit"]
    assert audit["canonical_redecode"]["maximum_vertex_error"] < 1.0e-12
    assert audit["m2_zero_step"]["maximum_vertex_error_vs_base"] < 1.0e-12
    assert audit["m2_finite_step"]["independent_redecode_maximum_vertex_error"] < 1.0e-12
    assert audit["first_variation"]["finite_difference_relative_l2_error"] < 1.0e-6
    assert audit["first_variation"]["projected_jvp_relative_error"] < 1.0e-10
    assert audit["mvc_diagnostics"]["maximum_barycentric_residual"] < 1.0e-12
    assert audit["mvc_diagnostics"]["maximum_covariance_condition"] >= 1.0
    assert audit["m2_lift_diagnostics"]["maximum_linearized_residual"] < 1.0e-12
    assert audit["solver_residuals"]["canonical_redecode"]["maximum_relative_residual"] < 1.0e-12
    for name in ("base", "canonical_redecode", "m2_zero", "m2_finite"):
        topology = audit["topology"][name]
        assert topology["all_certified"]
        assert topology["maximum_flip_count"] == 0
        assert topology["minimum_area_ratio"] > 0.0
    json.dumps(receipt, allow_nan=False)


def test_batch_repeats_float32_and_warmups_are_recorded_without_entering_samples() -> None:
    # The Windows CI environment loads incompatible Torch/SciPy OpenMP runtimes
    # when a nontrivial SuperLU solve is first exercised.  This test concerns
    # batching/timing rather than the CPU reference factorization, so use the
    # production-shaped iterative path instead of enabling the unsafe
    # KMP_DUPLICATE_LIB_OK workaround.
    receipt = _run(
        control_side=4,
        batch=2,
        dtype="float32",
        backend="matrix_free_directed",
        warmup=1,
        repeats=2,
    )

    assert receipt["status"] == "ok", receipt
    assert receipt["config"]["warmup"] == 1
    assert receipt["config"]["repeats"] == 2
    assert receipt["shapes"]["control"] == [2, 16, 2]
    for measured in receipt["timing"].values():
        assert len(measured["samples"]) == 2
        assert measured["warmup_iterations"] == 1
    assert receipt["audit"]["topology"]["m2_finite"]["sample_count"] == 2
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_tiny_cpu_matrix_free_exercises_the_cuda_solver_receipt_path(dtype: str) -> None:
    receipt = _run(backend="matrix_free_directed", dtype=dtype)

    assert receipt["status"] == "ok", receipt
    assert receipt["resolved_backend"] == "matrix_free_directed"
    assert receipt["config"]["solver_rtol"] is None
    assert receipt["solver_settings"]["relative_tolerance"] == (
        1.0e-5 if dtype == "float32" else 1.0e-11
    )
    assert receipt["solver_settings"]["relative_tolerance_source"] == "dtype_default"
    diagnostics = receipt["audit"]["solver_diagnostics"]
    assert diagnostics["canonical_redecode"]["forward"]["converged"] == [[True, True]]
    assert diagnostics["last_first_variation_backward"]["adjoint"]["converged"] == [
        [True, True]
    ]
    assert receipt["audit"]["topology"]["m2_finite"]["all_certified"]
    json.dumps(receipt, allow_nan=False)


def test_explicit_solver_rtol_is_applied_and_identified_in_the_receipt() -> None:
    receipt = _run(
        backend="matrix_free_directed",
        dtype="float32",
        solver_rtol=3.0e-6,
    )

    assert receipt["status"] == "ok", receipt
    assert receipt["config"]["solver_rtol"] == 3.0e-6
    assert receipt["solver_settings"]["relative_tolerance"] == 3.0e-6
    assert receipt["solver_settings"]["relative_tolerance_source"] == "explicit"
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("solver_rtol", [0.0, -1.0e-6, float("nan"), float("inf")])
def test_invalid_explicit_solver_rtol_is_an_explicit_validation_failure(
    solver_rtol: float,
) -> None:
    receipt = _run(backend="matrix_free_directed", solver_rtol=solver_rtol)

    assert receipt["status"] == "failure"
    assert receipt["stage"] == "validation"
    assert "solver_rtol" in receipt["error"]["message"]
    json.dumps(receipt, allow_nan=False)


def test_explicit_solver_rtol_cannot_be_silently_ignored_by_direct_backend() -> None:
    receipt = _run(backend="direct", solver_rtol=1.0e-6)

    assert receipt["status"] == "failure"
    assert receipt["stage"] == "validation"
    assert "solver_rtol" in receipt["error"]["message"]
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"control_side": 2},
        {"repeats": 0},
        {"dtype": "float16"},
        {"backend": "direct", "device": "cuda"},
    ],
)
def test_invalid_configuration_is_an_explicit_failure_row(kwargs) -> None:
    receipt = _run(**kwargs)

    assert receipt["status"] == "failure"
    assert receipt["stage"] == "validation"
    assert receipt["config"]
    assert receipt["error"]["type"]
    assert receipt["error"]["message"]
    json.dumps(receipt, allow_nan=False)


def test_runtime_failure_preserves_stage_and_partial_receipt() -> None:
    receipt = _run(strength=10000.0)

    assert receipt["status"] == "failure"
    assert receipt["stage"] != "validation"
    assert receipt["config"]["strength"] == 10000.0
    assert receipt["error"]["type"]
    json.dumps(receipt, allow_nan=False)


def test_nonfinite_audit_measurement_cannot_be_reported_as_success(monkeypatch) -> None:
    module = _module()
    original = module._map_error

    def nonfinite_once(actual, expected):
        result = original(actual, expected)
        result["maximum_vertex_error"] = float("nan")
        return result

    monkeypatch.setattr(module, "_map_error", nonfinite_once)
    config = module.BenchmarkConfig(
        control_side=3,
        batch=1,
        dtype="float64",
        device="cpu",
        backend="direct",
        warmup=0,
        repeats=1,
        threads=1,
    )

    receipt = module.run_case(config)

    assert receipt["status"] == "failure"
    assert receipt["stage"] == "audit_topology_residual"
    assert "nonfinite" in receipt["error"]["message"]
    json.dumps(receipt, allow_nan=False)


def test_suite_preserves_success_and_failure_rows_and_default_profiles() -> None:
    module = _module()
    assert module.parse_profiles("11:8,25:4,49:1") == [(11, 8), (25, 4), (49, 1)]
    good = module.BenchmarkConfig(
        control_side=3,
        batch=1,
        dtype="float64",
        device="cpu",
        backend="direct",
        warmup=0,
        repeats=1,
    )
    bad = module.BenchmarkConfig(
        control_side=2,
        batch=1,
        dtype="float64",
        device="cpu",
        backend="direct",
        warmup=0,
        repeats=1,
    )

    receipt = module.run_suite([good, bad])

    assert receipt["schema"] == "phase5_route2_mvc_layer_suite_v1"
    assert receipt["row_count"] == 2
    assert receipt["success_count"] == receipt["failure_count"] == 1
    assert [row["status"] for row in receipt["rows"]] == ["ok", "failure"]
    json.dumps(receipt, allow_nan=False)


def test_tiny_cli_emits_exactly_one_json_document() -> None:
    _module()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profiles",
            "3:1",
            "--dtypes",
            "float64",
            "--devices",
            "cpu",
            "--backend",
            "direct",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    receipt = json.loads(result.stdout)
    assert receipt["success_count"] == 1
    assert receipt["failure_count"] == 0


def test_tiny_cli_propagates_explicit_solver_rtol() -> None:
    _module()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profiles",
            "3:1",
            "--dtypes",
            "float32",
            "--devices",
            "cpu",
            "--backend",
            "matrix_free_directed",
            "--solver-rtol",
            "3e-6",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    receipt = json.loads(result.stdout)
    row = receipt["rows"][0]
    assert row["config"]["solver_rtol"] == 3.0e-6
    assert row["solver_settings"]["relative_tolerance"] == 3.0e-6
    assert row["solver_settings"]["relative_tolerance_source"] == "explicit"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tiny_cuda_uses_only_matrix_free_directed_and_reports_peak_memory() -> None:
    receipt = _run(device="cuda", backend="auto", dtype="float64")

    assert receipt["status"] == "ok", receipt
    assert receipt["resolved_backend"] == "matrix_free_directed"
    for measured in receipt["timing"].values():
        memory = measured["gpu_memory"]
        assert memory["peak_allocated_bytes"] >= memory["baseline_allocated_bytes"]
        assert memory["peak_reserved_bytes"] >= memory["baseline_reserved_bytes"]
    json.dumps(receipt, allow_nan=False)
