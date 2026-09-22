"""Receipt tests for the Route-II fixed-boundary O1--O4 benchmark CLI."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.filterwarnings("error")


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/phase5/route2_mvc_instance_benchmark.py"
)


def _module():
    assert SCRIPT.is_file(), "Route-II MVC instance benchmark is missing"
    spec = importlib.util.spec_from_file_location(
        "phase5_route2_mvc_instance_benchmark", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(**kwargs):
    module = _module()
    defaults = dict(
        task="map",
        methods=module.METHOD_NAMES,
        control_side=5,
        resolution=17,
        steps=3,
        learning_rate=0.01,
        objective_threshold=1.0e-4,
        seed=521,
        target_strength=0.20,
        image_name="smooth_blobs",
        backend="direct",
        dtype="float64",
        device="cpu",
        slice_global_solves=5,
        slice_outer_step=None,
    )
    defaults.update(kwargs)
    return module.run_benchmark(module.BenchmarkConfig(**defaults))


def test_tiny_map_receipt_keeps_full_trace_counts_metrics_and_audits() -> None:
    module = _module()
    receipt = _run()

    assert receipt["status"] == "ok", receipt
    assert receipt["schema"] == "phase5_route2_mvc_instance_benchmark_v1"
    assert receipt["configuration"]["methods"] == list(module.METHOD_NAMES)
    assert receipt["shared_target"]["generated_once"]
    assert receipt["shared_target"]["setup_primal_solves"] == 1
    assert receipt["shared_target"]["comparison_target_setup_primal_solves"] == 0
    assert (
        receipt["target_reuse_audit"][
            "maximum_control_cast_error_vs_cpu_float64_authority"
        ]
        == 0.0
    )
    assert (
        receipt["target_reuse_audit"][
            "maximum_dense_cast_error_vs_cpu_float64_authority"
        ]
        == 0.0
    )
    assert receipt["target_reuse_audit"]["authority_was_reused_without_regeneration"]
    assert receipt["method_failure_count"] == 0
    assert receipt["independent_audit_failure_count"] == 0
    assert receipt["nonfinite_guard"] == {"count": 0, "paths": []}
    assert receipt["problem_size"] == {
        "control_side": 5,
        "control_vertex_count": 25,
        "boundary_vertex_count": 16,
        "interior_row_count": 9,
        "face_count": 32,
        "dense_query_count": 17 * 17,
    }
    assert receipt["solver_settings"]["system_condition_dense_limit"] == 256
    assert receipt["solver_settings"]["max_covariance_condition"] == 1.0e8
    assert receipt["solver_settings"]["torch_threads"] == 1
    assert set(receipt["methods"]) == set(module.METHOD_NAMES)

    expected = {
        "O1_sigmoid_positive": (4, 3, 0),
        "O2_row_softmax": (4, 3, 0),
        "O3_mvc_adam": (8, 3, 0),
        "O4_covariance_retraction": (5, 0, 3),
    }
    for name, row in receipt["methods"].items():
        assert row["status"] == "success"
        assert len(row["trace"]) == 4
        assert (
            row["primal_solves"],
            row["adjoint_solves"],
            row["local_backwards"],
        ) == expected[name]
        assert row["primal_attempts"] >= row["primal_solves"]
        assert row["adjoint_attempts"] >= row["adjoint_solves"]
        assert row["final_metrics"]["global_injectivity_certificate"]
        assert row["trace"][-1]["topology_certified"]
        assert "maximum_mvc_canonical_covariance_condition" in row
        assert "final_control" not in row and "final_logits" not in row
        audit = receipt["independent_final_redecode"][name]
        assert audit["status"] == "ok"
        assert audit["verified"]
        assert audit["audit_primal_solves"] == 1
        assert audit["maximum_vertex_error"] < 1.0e-12
        assert audit["maximum_boundary_error"] == 0.0

    assert not receipt["protocol_slices"]["secondary_common_outer_step"]["complete"]
    assert not receipt["protocol_slices"]["primary_exact_global_solves"]["complete"]
    assert (
        receipt["memory"]["cpu"]["before"]["rss_bytes"] is None
        or receipt["memory"]["cpu"]["before"]["rss_bytes"] > 0
    )
    json.dumps(receipt, allow_nan=False)


def test_exact_83_and_secondary_40_slices_are_serialized_without_mixing() -> None:
    receipt = _run(
        control_side=3,
        resolution=7,
        steps=81,
        learning_rate=0.001,
        target_strength=0.08,
        objective_threshold=None,
        slice_global_solves=83,
        slice_outer_step=40,
    )

    primary = receipt["protocol_slices"]["primary_exact_global_solves"]
    secondary = receipt["protocol_slices"]["secondary_common_outer_step"]
    assert primary["complete"] and not primary["missing"]
    assert secondary["complete"] and not secondary["missing"]
    assert [
        primary["rows"][name]["iteration"]
        for name in receipt["configuration"]["methods"]
    ] == [
        41,
        41,
        27,
        81,
    ]
    assert {row["global_solves"] for row in primary["rows"].values()} == {83}
    assert [
        secondary["rows"][name]["global_solves"]
        for name in receipt["configuration"]["methods"]
    ] == [
        81,
        81,
        122,
        42,
    ]
    assert primary["interpretation"] != secondary["interpretation"]
    json.dumps(receipt, allow_nan=False)


def test_requested_exact_23_solve_pilot_slice_is_not_selected_post_hoc() -> None:
    receipt = _run(
        control_side=3,
        resolution=7,
        steps=21,
        learning_rate=0.001,
        target_strength=0.08,
        objective_threshold=None,
        slice_global_solves=23,
        slice_outer_step=None,
    )

    requested = receipt["protocol_slices"]["requested_exact_global_solves"]
    assert receipt["protocol"]["requested_global_solve_budget"] == 23
    assert receipt["protocol"]["requested_expected_accepted_steps"] == {
        "O1_sigmoid_positive": 11,
        "O2_row_softmax": 11,
        "O3_mvc_adam": 7,
        "O4_covariance_retraction": 21,
    }
    assert requested["complete"] and not requested["missing"]
    assert {row["global_solves"] for row in requested["rows"].values()} == {23}
    assert not receipt["protocol_slices"]["requested_outer_step"]["enabled"]
    assert not receipt["protocol_slices"]["primary_exact_global_solves"]["complete"]


def test_image_receipt_uses_backward_warp_and_compact_numeric_target_identity() -> None:
    receipt = _run(
        task="image",
        methods=("O2_row_softmax", "O4_covariance_retraction"),
        control_side=3,
        resolution=24,
        steps=2,
        learning_rate=0.005,
        target_strength=0.10,
        image_name="medical_phantom",
        slice_global_solves=3,
    )

    assert receipt["status"] == "ok", receipt
    assert receipt["configuration"]["warp_convention"] == "backward_map_fixed_to_moving"
    target = receipt["shared_target"]["numeric_identity"]
    for tensor_name in (
        "interior_logits",
        "boundary_logits",
        "raw_modulus",
        "control",
        "dense",
    ):
        assert set(target[tensor_name]) == {"shape", "sum", "l2", "minimum", "maximum"}
    assert target["dense"]["shape"] == [24, 24, 2]
    assert "values" not in target["dense"]
    assert len(json.dumps(receipt, allow_nan=False)) < 1_000_000


def test_invalid_configuration_is_a_strict_failure_receipt() -> None:
    receipt = _run(control_side=2)

    assert receipt["status"] == "failure"
    assert receipt["stage"] == "validation"
    assert receipt["error"]["type"] == "ValueError"
    assert "control_side" in receipt["error"]["message"]
    json.dumps(receipt, allow_nan=False)


def test_optimizer_and_independent_audit_failures_change_overall_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer_failure = _run(
        methods=("O1_sigmoid_positive",),
        control_side=3,
        resolution=7,
        steps=1,
        learning_rate=1.0e308,
        slice_global_solves=1,
    )
    assert optimizer_failure["status"] == "complete_with_failures"
    assert optimizer_failure["method_failure_count"] == 1
    assert optimizer_failure["methods"]["O1_sigmoid_positive"]["status"] == (
        "optimizer_failure"
    )

    module = _module()
    monkeypatch.setattr(
        module,
        "_independent_redecode",
        lambda *args, **kwargs: {
            "status": "failure",
            "audit_primal_attempts": 0,
            "audit_primal_solves": 0,
            "error": {"type": "InjectedFailure", "message": "audit failed"},
        },
    )
    receipt = module.run_benchmark(
        module.BenchmarkConfig(
            task="map",
            methods=("O2_row_softmax",),
            control_side=3,
            resolution=7,
            steps=1,
            learning_rate=0.005,
            seed=31,
            target_strength=0.1,
            backend="direct",
            dtype="float64",
            device="cpu",
            slice_global_solves=3,
            slice_outer_step=None,
        )
    )
    assert receipt["status"] == "complete_with_failures"
    assert receipt["method_failure_count"] == 0
    assert receipt["independent_audit_failure_count"] == 1


def test_independent_redecode_requires_map_agreement_and_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    real_layer = module.DirectTutteLayer

    class ShiftedDirectTutteLayer(real_layer):
        def forward(self, logits, boundary):
            return super().forward(logits, boundary) + 0.1

    monkeypatch.setattr(module, "DirectTutteLayer", ShiftedDirectTutteLayer)
    receipt = module.run_benchmark(
        module.BenchmarkConfig(
            task="map",
            methods=("O2_row_softmax",),
            control_side=3,
            resolution=7,
            steps=1,
            learning_rate=0.005,
            seed=32,
            target_strength=0.1,
            backend="direct",
            dtype="float64",
            device="cpu",
            slice_global_solves=3,
            slice_outer_step=None,
        )
    )

    audit = receipt["independent_final_redecode"]["O2_row_softmax"]
    assert receipt["status"] == "complete_with_failures"
    assert receipt["audit_failure_count"] == 1
    assert audit["status"] == "verification_failure"
    assert not audit["verified"]
    assert audit["audit_primal_attempts"] == audit["audit_primal_solves"] == 1
    assert audit["maximum_vertex_error"] > audit["agreement_atol"]
    assert audit["verification_failures"]


def test_nonfinite_guard_is_explicit_and_never_silent_success() -> None:
    module = _module()
    receipt = module._finalize_receipt(
        {"schema": "test", "status": "ok", "nested": {"value": float("inf")}}
    )

    assert receipt["status"] == "complete_with_failures"
    assert receipt["nested"]["value"] is None
    assert receipt["nonfinite_guard"]["count"] == 1
    assert receipt["nonfinite_guard"]["paths"] == ["$.nested.value"]
    json.dumps(receipt, allow_nan=False)


def test_cuda_snapshot_labels_successful_queries_as_partial_after_benchmark_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(module.torch.cuda, "max_memory_allocated", lambda device: 30)
    monkeypatch.setattr(module.torch.cuda, "max_memory_reserved", lambda device: 40)

    snapshot = module._cuda_memory_snapshot(
        module.torch.device("cuda"),
        baseline_allocated=10,
        baseline_reserved=20,
        benchmark_complete=False,
        failure_stage="comparison",
    )

    assert snapshot == {
        "baseline_allocated_bytes": 10,
        "baseline_reserved_bytes": 20,
        "peak_allocated_bytes": 30,
        "peak_reserved_bytes": 40,
        "measurement_status": "partial",
        "scope": "partial allocator receipt through benchmark failure stage comparison",
    }


def test_requested_slice_and_cuda_measurement_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unreachable = _run(steps=2, slice_global_solves=5)
    assert unreachable["status"] == "failure"
    assert unreachable["stage"] == "validation"
    assert "O4_covariance_retraction" in unreachable["error"]["message"]

    unreachable_outer = _run(steps=3, slice_outer_step=4)
    assert unreachable_outer["status"] == "failure"
    assert unreachable_outer["stage"] == "validation"
    assert "slice_outer_step" in unreachable_outer["error"]["message"]

    runtime_missing = _run(
        steps=3,
        slice_global_solves=5,
        learning_rate=1.0e308,
    )
    assert runtime_missing["status"] == "complete_with_failures"
    assert runtime_missing["protocol_failure_count"] == 1
    assert runtime_missing["protocol_slices"]["requested_exact_global_solves"][
        "missing"
    ]

    outer_missing = _run(
        methods=("O1_sigmoid_positive",),
        steps=2,
        learning_rate=1.0e308,
        slice_global_solves=1,
        slice_outer_step=2,
    )
    assert outer_missing["status"] == "complete_with_failures"
    assert outer_missing["protocol_failure_count"] == 1
    assert outer_missing["protocol_failures"]["requested_outer_step"]

    module = _module()
    monkeypatch.setattr(
        module,
        "_cuda_memory_snapshot",
        lambda *args, **kwargs: {
            "measurement_status": "failure",
            "measurement_error": {
                "type": "InjectedAllocatorFailure",
                "message": "allocator query failed",
            },
        },
    )
    memory_failure = module.run_benchmark(
        module.BenchmarkConfig(
            task="map",
            methods=("O2_row_softmax",),
            control_side=3,
            resolution=7,
            steps=1,
            learning_rate=0.005,
            seed=33,
            target_strength=0.1,
            backend="direct",
            dtype="float64",
            device="cpu",
            slice_global_solves=3,
            slice_outer_step=None,
        )
    )
    assert memory_failure["status"] == "complete_with_failures"
    assert memory_failure["memory_failure_count"] == 1


def test_fresh_process_single_method_cli_emits_one_strict_json_document() -> None:
    _module()
    environment = dict(os.environ)
    environment.pop("MKL_THREADING_LAYER", None)
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            str(SCRIPT),
            "--task",
            "map",
            "--methods",
            "O2",
            "--N",
            "3",
            "--R",
            "9",
            "--steps",
            "2",
            "--lr",
            "0.005",
            "--seed",
            "44",
            "--strength",
            "0.1",
            "--backend",
            "direct",
            "--dtype",
            "float64",
            "--device",
            "cpu",
            "--slice-global-solves",
            "5",
            "--slice-outer-step",
            "none",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "ok"
    assert receipt["configuration"]["fresh_process_single_method"]
    assert list(receipt["methods"]) == ["O2_row_softmax"]
    assert result.stderr == ""


def test_cli_rejects_unknown_method_before_running() -> None:
    _module()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "map",
            "--methods",
            "O9",
            "--N",
            "3",
            "--R",
            "9",
            "--steps",
            "1",
            "--lr",
            "0.005",
            "--backend",
            "direct",
            "--dtype",
            "float64",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "argument_failure"
    assert receipt["stage"] == "argument_parsing"
    assert "unknown method" in receipt["error"]["message"]
    assert result.stderr == ""


def test_cli_validation_failure_is_strict_json_and_nonzero() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "map",
            "--methods",
            "O1,,O2",
            "--N",
            "2",
            "--R",
            "9",
            "--steps",
            "1",
            "--lr",
            "0.005",
            "--backend",
            "direct",
            "--dtype",
            "float64",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "argument_failure"
    assert "empty entries" in receipt["error"]["message"]
    assert result.stderr == ""

    runtime = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "map",
            "--methods",
            "O1",
            "--N",
            "2",
            "--R",
            "9",
            "--steps",
            "1",
            "--lr",
            "0.005",
            "--backend",
            "direct",
            "--dtype",
            "float64",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    runtime_receipt = json.loads(runtime.stdout)
    assert runtime.returncode == 2
    assert runtime_receipt["status"] == "failure"
    assert runtime_receipt["stage"] == "validation"
    assert runtime.stderr == ""


@pytest.mark.skipif(
    not _module().torch.cuda.is_available(), reason="CUDA is unavailable"
)
def test_cuda_receipt_has_allocator_baseline_and_peak() -> None:
    receipt = _run(
        methods=("O2_row_softmax",),
        control_side=3,
        resolution=9,
        steps=1,
        backend="directed_iterative",
        dtype="float32",
        device="cuda",
        slice_global_solves=3,
    )

    assert receipt["status"] == "ok", receipt
    gpu = receipt["memory"]["cuda"]
    assert gpu["peak_allocated_bytes"] >= gpu["baseline_allocated_bytes"]
    assert gpu["peak_reserved_bytes"] >= gpu["baseline_reserved_bytes"]
    json.dumps(receipt, allow_nan=False)
