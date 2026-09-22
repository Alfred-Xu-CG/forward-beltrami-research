"""Tiny tests for the common-input full-Hodge P-ref receipt."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

# The existing CPU direct target solve and Whitney reference share MKL on
# Windows; configure the fresh test process before importing NumPy/Torch.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import pytest

from qcopt.forward.mmatrix import beltrami_conductivity
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.instance_optimization import build_directed_target


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/phase5/route3_pref_common_benchmark.py"


def module():
    spec = importlib.util.spec_from_file_location("route3_pref_common_test", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_vectorized_beltrami_tensor_matches_independent_validated_scalar_helper() -> None:
    implementation = module()
    mu = np.array([0.0, 0.2 + 0.1j, -0.31 + 0.27j], dtype=np.complex128)
    actual = implementation.beltrami_to_tensor(mu)
    expected = np.stack([beltrami_conductivity(complex(value)) for value in mu])
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(np.linalg.det(actual), 1.0, rtol=2e-14, atol=2e-14)


def test_shared_target_identity_matches_route2_contract() -> None:
    implementation = module()
    mesh = structured_rectangle(4, 4)
    target = build_directed_target(
        mesh,
        image_height=13,
        image_width=13,
        seed=20260922,
        strength=0.25,
        height=0.9,
    )
    identity = implementation.target_identity(target)
    assert set(identity) == {
        "interior_logits",
        "boundary_logits",
        "raw_modulus",
        "control",
        "dense",
    }
    for name, summary in identity.items():
        value = getattr(target, name).detach().to(dtype=implementation.torch.float64)
        assert summary["shape"] == list(value.shape)
        assert summary["sum"] == float(value.sum())
        assert summary["l2"] == float(value.reshape(-1).norm())


@pytest.mark.parametrize("task", ["map", "image"])
def test_formal_identity_validator_checks_every_route2_summary_field(task: str) -> None:
    implementation = module()
    path = (
        ROOT
        / "docs/research_phase5/raw_results"
        / f"route2_mvc_instance_formal_{task}_gpu_O1_ai_74b452c_clean.json"
    )
    authority = json.loads(path.read_text(encoding="utf-8"))
    identity = authority["shared_target"]["numeric_identity"]
    config = implementation.BenchmarkConfig(task=task)
    audit = implementation.validate_formal_target_identity(identity, config)
    assert audit["required"] is True
    assert audit["matched"] is True
    assert audit["rtol"] == 1.0e-12
    assert audit["atol"] == 1.0e-12
    assert audit["compared_fields"] == ["shape", "sum", "l2", "minimum", "maximum"]

    changed = json.loads(json.dumps(identity))
    changed["control"]["maximum"] += 1.0e-5
    with pytest.raises(ValueError, match="target numeric identity"):
        implementation.validate_formal_target_identity(changed, config)


@pytest.mark.parametrize("task,expected_height", [("map", 0.9), ("image", 1.0)])
def test_tiny_reference_recovers_exact_p1_target_and_checks_full_vjp(
    task: str, expected_height: float
) -> None:
    implementation = module()
    config = implementation.BenchmarkConfig(
        task=task,
        control_side=5,
        resolution=17,
        seed=20260922,
        target_strength=0.25,
        threads=1,
        fd_control_side=4,
        fd_resolution=9,
        fd_epsilon=1.0e-6,
    )
    receipt = implementation.run_benchmark(config)

    assert receipt["status"] == "ok"
    assert receipt["protocol_kind"] == "one_shot_fixed_boundary_structure_preserving_reference"
    assert receipt["topology_scope"] == (
        "observed_P1_topology_for_this_target_only_no_general_guarantee"
    )
    assert receipt["configuration"]["target_height"] == expected_height
    assert receipt["configuration"]["image_name"] == "medical_phantom"
    assert receipt["configuration"]["warp_convention"] == "backward_map_fixed_to_moving"
    assert receipt["optimization"] == {
        "performed": False,
        "iterations": None,
        "stopping_threshold": None,
    }

    recovery = receipt["exact_p1_recovery"]
    assert recovery["whitney_vs_target_max_abs"] < 2.0e-11
    assert recovery["independent_p1_vs_target_max_abs"] < 2.0e-11
    assert recovery["whitney_vs_independent_p1_max_abs"] < 2.0e-11
    assert recovery["independent_operator_residual_linf"] < 2.0e-11
    assert recovery["whitney_solver_residual_linf"] < 2.0e-11

    metrics = receipt["metrics"]
    assert metrics["flip_count"] == 0
    assert metrics["global_injectivity_certificate"] is True
    assert metrics["map_rmse"] < 2.0e-11
    assert metrics["mu_rmse"] < 2.0e-10
    assert metrics["image_mse"] < 2.0e-20
    assert "image_similarity" not in metrics
    assert "image_similarity_kind" not in metrics
    assert receipt["primary_objective"]["kind"] == (
        "dense_map_coordinate_mse" if task == "map" else "backward_warp_image_mse"
    )
    assert receipt["primary_objective"]["optimized"] is False
    assert receipt["solve_counts"]["primary_forward_scalar_rhs"] == 2
    assert receipt["solve_counts"]["primary_backward_scalar_rhs"] == 2
    assert receipt["solve_counts"]["finite_difference_diagnostic_excluded_from_primary"] is True

    vjp = receipt["vjp_check"]
    assert vjp["path"] == (
        "tensor_to_whitney_solve_to_dense_map"
        if task == "map"
        else "tensor_to_whitney_solve_to_dense_map_to_backward_image_warp"
    )
    assert vjp["relative_error"] < 2.0e-6
    assert receipt["backward"]["diagnostic_only"] is True
    assert receipt["backward"]["excluded_from_common_primary_objective"] is True
    assert receipt["shared_target"]["route2_authority_match"]["required"] is False
    assert receipt["replayable_state"]["target_builder"]["height"] == expected_height

    # The receipt writer must never silently emit JavaScript NaN/Infinity.
    encoded = implementation.strict_json_text(receipt)
    decoded = json.loads(encoded, parse_constant=lambda value: pytest.fail(value))
    assert decoded["status"] == "ok"
    with pytest.raises(ValueError):
        implementation.strict_json_text({"bad": float("nan")})
