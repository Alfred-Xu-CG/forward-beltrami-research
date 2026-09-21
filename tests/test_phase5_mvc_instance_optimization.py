"""Tiny decisive tests for the fair fixed-boundary Route-II comparison."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer
from qcopt.neural_bijection.tutte.instance_optimization import build_directed_target
from qcopt.neural_bijection.tutte.mvc_instance_optimization import (
    METHOD_NAMES,
    extract_exact_global_solve_slice,
    extract_outer_step_slice,
    run_fixed_boundary_comparison,
)


def _supervised(*, steps: int = 2, learning_rate: float = 0.01):
    return run_fixed_boundary_comparison(
        task="supervised_map",
        control_vertices=5,
        image_resolution=17,
        steps=steps,
        backend="direct",
        dtype=torch.float64,
        seed=521,
        target_strength=0.20,
        learning_rate=learning_rate,
    )


def test_all_methods_share_the_uniform_initial_map_and_fixed_target_boundary() -> None:
    comparison = _supervised(steps=1)

    assert tuple(comparison.results) == METHOD_NAMES
    for result in comparison.results.values():
        assert result.failure is None
        torch.testing.assert_close(
            result.initial_control,
            comparison.uniform_initial_control,
            atol=3.0e-13,
            rtol=3.0e-13,
        )
        torch.testing.assert_close(
            result.fixed_boundary,
            comparison.fixed_target_boundary,
            atol=0.0,
            rtol=0.0,
        )
        assert result.trace[0].topology_certified


def test_actual_global_solve_accounting_includes_mvc_redecodes() -> None:
    comparison = _supervised(steps=2)
    o1, o2, o3, o4 = (comparison.results[name] for name in METHOD_NAMES)

    assert (o1.primal_solves, o1.adjoint_solves, o1.local_backwards) == (3, 2, 0)
    assert (o2.primal_solves, o2.adjoint_solves, o2.local_backwards) == (3, 2, 0)
    assert (o3.primal_solves, o3.adjoint_solves, o3.local_backwards) == (6, 2, 0)
    assert (o4.primal_solves, o4.adjoint_solves, o4.local_backwards) == (4, 0, 2)
    for result in comparison.results.values():
        assert result.global_solves == result.primal_solves + result.adjoint_solves
        assert result.trace[-1].global_solves == result.global_solves
        assert result.primal_attempts == result.primal_solves
        assert result.adjoint_attempts == result.adjoint_solves


def test_predeclared_outer_step_and_exact_global_solve_slices_are_distinct() -> None:
    comparison = _supervised(steps=3)

    step_slice = extract_outer_step_slice(comparison, 2)
    assert step_slice.complete
    assert [row.global_solves for row in step_slice.rows.values()] == [5, 5, 8, 4]
    solve_slice = extract_exact_global_solve_slice(comparison, 5)
    assert solve_slice.complete
    assert {row.global_solves for row in solve_slice.rows.values()} == {5}
    assert [row.iteration for row in solve_slice.rows.values()] == [2, 2, 1, 3]
    assert comparison.primary_global_solve_budget == 83
    assert list(comparison.primary_expected_accepted_steps.values()) == [41, 41, 27, 81]
    assert comparison.secondary_outer_step == 40
    assert list(comparison.secondary_expected_global_solves.values()) == [
        81,
        81,
        122,
        42,
    ]
    assert "different units" in comparison.shared_learning_rate_semantics


def test_tiny_supervised_run_reduces_loss_and_keeps_mvc_states_canonical() -> None:
    comparison = _supervised(steps=3)

    for result in comparison.results.values():
        assert result.failure is None
        assert result.best_objective < result.initial_objective
        assert result.final_metrics.global_injectivity_certificate
        assert result.final_metrics.boundary_order_min_gap > 0.0
        assert result.all_iterates_certified
    for name in ("O3_mvc_adam", "O4_covariance_retraction"):
        result = comparison.results[name]
        assert result.final_canonicality_error < 2.0e-12
        assert result.maximum_mvc_canonical_covariance_condition >= 1.0
        assert result.final_logit_spread >= 0.0
    assert "identity transport" in comparison.results["O3_mvc_adam"].moment_policy
    for result in comparison.results.values():
        assert result.algorithm_wall_seconds > 0.0
        assert result.audit_wall_seconds > 0.0
        assert all(row.dense_loss_seconds > 0.0 for row in result.trace)


def test_o4_accepts_only_decoder_outputs_not_raw_vertex_euler_candidates() -> None:
    result = _supervised(steps=2).results["O4_covariance_retraction"]

    assert result.failure is None
    assert result.adjoint_solves == 0
    assert result.local_backwards == 2
    assert all(row.raw_euler_gap_rmse is None for row in result.trace[:1])
    assert all(
        row.raw_euler_gap_rmse is not None and row.raw_euler_gap_rmse > 0.0
        for row in result.trace[1:]
    )
    mesh = structured_rectangle(4, 4)
    recovered = DirectTutteLayer(mesh)(result.final_logits, result.fixed_boundary)
    torch.testing.assert_close(
        recovered,
        result.final_control,
        atol=3.0e-13,
        rtol=3.0e-13,
    )


def test_image_task_uses_route_i_backward_warp_convention() -> None:
    comparison = run_fixed_boundary_comparison(
        task="image_registration",
        control_vertices=5,
        image_resolution=24,
        steps=2,
        backend="direct",
        dtype=torch.float64,
        seed=811,
        target_strength=0.12,
        learning_rate=0.005,
        image_name="smooth_blobs",
        methods=("O2_row_softmax", "O4_covariance_retraction"),
    )

    assert comparison.warp_convention == "backward_map_fixed_to_moving"
    assert tuple(comparison.results) == ("O2_row_softmax", "O4_covariance_retraction")
    for result in comparison.results.values():
        assert result.failure is None
        assert result.best_objective < result.initial_objective
        assert result.final_metrics.flip_count == 0


def test_matrix_free_directed_backend_obeys_the_same_counting_contract() -> None:
    comparison = run_fixed_boundary_comparison(
        task="supervised_map",
        control_vertices=4,
        image_resolution=11,
        steps=1,
        backend="directed_iterative",
        dtype=torch.float64,
        seed=912,
        target_strength=0.10,
        learning_rate=0.005,
        methods=("O2_row_softmax", "O4_covariance_retraction"),
    )

    o2 = comparison.results["O2_row_softmax"]
    o4 = comparison.results["O4_covariance_retraction"]
    assert o2.failure is None
    assert o4.failure is None
    assert (o2.primal_solves, o2.adjoint_solves) == (2, 1)
    assert (o4.primal_solves, o4.adjoint_solves, o4.local_backwards) == (3, 0, 1)
    assert o2.all_iterates_certified and o4.all_iterates_certified
    assert o2.primal_krylov_rhs_iterations > 0
    assert o2.adjoint_krylov_rhs_iterations > 0
    assert o4.primal_krylov_rhs_iterations > 0
    assert o4.adjoint_krylov_rhs_iterations == 0
    assert o2.maximum_primal_krylov_relative_residual >= 0.0


def test_failure_is_preserved_instead_of_silently_retuning_the_shared_rate() -> None:
    comparison = _supervised(steps=1, learning_rate=1.0e308)

    assert comparison.learning_rate == 1.0e308
    assert any(result.failure is not None for result in comparison.results.values())
    for result in comparison.results.values():
        assert result.learning_rate == comparison.learning_rate
        if result.failure is not None:
            assert result.completed_steps == 0
            assert result.trace
            assert bool(torch.isfinite(result.final_logits).all())
            mesh = structured_rectangle(4, 4)
            recovered = DirectTutteLayer(mesh)(
                result.final_logits, result.fixed_boundary
            )
            torch.testing.assert_close(
                recovered,
                result.final_control,
                atol=3.0e-13,
                rtol=3.0e-13,
            )
    o1 = comparison.results["O1_sigmoid_positive"]
    o2 = comparison.results["O2_row_softmax"]
    o3 = comparison.results["O3_mvc_adam"]
    o4 = comparison.results["O4_covariance_retraction"]
    assert (o1.primal_solves, o1.primal_attempts, o1.adjoint_solves) == (1, 2, 1)
    assert (o2.primal_solves, o2.primal_attempts, o2.adjoint_solves) == (1, 2, 1)
    assert (o3.primal_solves, o3.primal_attempts, o3.adjoint_solves) == (2, 3, 1)
    assert (o4.primal_solves, o4.primal_attempts, o4.adjoint_solves) == (2, 2, 0)
    assert o4.local_backwards == o4.local_backward_attempts == 1
    assert all(result.failure.phase != "latent_update" for result in (o1, o2, o3))
    assert o4.failure.phase == "covariance_lift"


def test_prebuilt_target_must_match_its_dense_p1_interpolation() -> None:
    mesh = structured_rectangle(4, 4)
    target = build_directed_target(
        mesh,
        image_height=17,
        image_width=17,
        seed=17,
        strength=0.1,
        height=0.9,
    )
    inconsistent = replace(target, dense=torch.zeros_like(target.dense))

    with pytest.raises(ValueError, match="inconsistent"):
        run_fixed_boundary_comparison(
            task="supervised_map",
            control_vertices=5,
            image_resolution=17,
            steps=1,
            backend="direct",
            dtype=torch.float64,
            seed=17,
            target_strength=0.1,
            learning_rate=0.01,
            target=inconsistent,
        )


@pytest.mark.parametrize("case", ("domain", "dtype", "topology"))
def test_prebuilt_target_reuses_full_route_i_validation(case: str) -> None:
    mesh = structured_rectangle(4, 4)
    target = build_directed_target(
        mesh,
        image_height=17,
        image_width=17,
        seed=19,
        strength=0.1,
        height=0.9,
    )
    if case == "domain":
        dense = target.dense.clone()
        dense[0, 0, 0] = 2.0
        invalid = replace(target, dense=dense)
        message = "unit image domain"
    elif case == "dtype":
        invalid = replace(target, control=target.control.to(torch.int64))
        message = "float32 or float64"
    else:
        control = target.control.clone()
        control[12] = control[6]
        dense = StructuredDenseQueryTable.from_mesh(
            mesh, height=17, width=17
        ).interpolate(control)
        invalid = replace(target, control=control, dense=dense)
        message = "topology certificate"

    with pytest.raises(ValueError, match=message):
        run_fixed_boundary_comparison(
            task="supervised_map",
            control_vertices=5,
            image_resolution=17,
            steps=1,
            backend="direct",
            dtype=torch.float64,
            seed=19,
            target_strength=0.1,
            learning_rate=0.01,
            target=invalid,
        )


def test_upstream_backward_failure_is_an_attempt_not_a_completed_adjoint(
    monkeypatch,
) -> None:
    import qcopt.neural_bijection.tutte.mvc_instance_optimization as module

    class RaiseBeforeDecoderAdjoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            raise RuntimeError("injected before decoder adjoint")

    original = module._timed_dense_loss

    def wrapped(*args, **kwargs):
        loss, elapsed = original(*args, **kwargs)
        return RaiseBeforeDecoderAdjoint.apply(loss), elapsed

    monkeypatch.setattr(module, "_timed_dense_loss", wrapped)
    comparison = run_fixed_boundary_comparison(
        task="supervised_map",
        control_vertices=5,
        image_resolution=17,
        steps=1,
        backend="direct",
        dtype=torch.float64,
        seed=20,
        target_strength=0.1,
        learning_rate=0.01,
        methods=("O2_row_softmax",),
    )
    result = comparison.results["O2_row_softmax"]

    assert result.failure.phase == "loss_and_decoder_adjoint_backward"
    assert result.adjoint_attempts == 1
    assert result.adjoint_solves == 0
    assert result.primal_attempts == result.primal_solves == 1
    assert result.completed_steps == 0
