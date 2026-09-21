from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte import direct as direct_module
from qcopt.neural_bijection.tutte.instance_optimization import (
    build_directed_target,
    run_image_instance,
    run_supervised_instance,
)


def test_positive_tutte_target_is_deterministic_nontrivial_and_certified() -> None:
    mesh = structured_rectangle(4, 4)
    first = build_directed_target(
        mesh,
        image_height=32,
        image_width=32,
        seed=1701,
        strength=0.35,
        height=0.9,
    )
    second = build_directed_target(
        mesh,
        image_height=32,
        image_width=32,
        seed=1701,
        strength=0.35,
        height=0.9,
    )

    torch.testing.assert_close(first.control, second.control, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.dense, second.dense, atol=0.0, rtol=0.0)
    assert first.metrics.global_injectivity_certificate
    assert first.metrics.flip_count == 0
    assert first.metrics.minimum_area_ratio > 0.0
    assert not np.allclose(first.control.numpy(), mesh.vertices)


def test_short_supervised_adam_instance_reduces_actual_dense_decoder_error() -> None:
    result = run_supervised_instance(
        control_vertices=5,
        image_resolution=32,
        steps=12,
        backend="direct",
        optimizer_name="adam",
        dtype=torch.float64,
        seed=2702,
        target_strength=0.25,
        learning_rate=0.12,
        objective_threshold=1.0e-3,
    )

    assert result.final_objective < 0.75 * result.initial_objective
    assert result.best_objective <= result.final_objective
    assert result.primal_solves == 13
    assert result.adjoint_solves == 12
    assert result.global_solves == 25
    assert result.global_solves_to_threshold is not None
    assert result.global_solves_to_threshold < result.global_solves
    assert result.wall_seconds_to_threshold is not None
    assert result.wall_seconds_to_threshold <= result.wall_seconds
    assert result.control_vertices_per_side == 5
    assert result.control_vertex_count == 25
    assert all(row.audit_seconds >= 0.0 for row in result.trace)
    assert all(
        left.observation_wall_seconds <= right.observation_wall_seconds
        for left, right in zip(result.trace, result.trace[1:])
    )
    assert result.all_iterates_certified
    assert all(row.flip_count == 0 for row in result.trace)
    assert all(row.minimum_area_ratio > 0.0 for row in result.trace)


def test_short_image_adam_instance_reduces_actual_warp_loss() -> None:
    result = run_image_instance(
        control_vertices=5,
        image_resolution=32,
        steps=12,
        backend="direct",
        optimizer_name="adam",
        dtype=torch.float64,
        seed=3703,
        target_strength=0.22,
        learning_rate=0.08,
        image_name="textured",
    )

    assert result.final_objective < 0.9 * result.initial_objective
    assert result.best_objective <= result.final_objective
    assert result.primal_solves == 13
    assert result.adjoint_solves == 12
    assert result.global_solves == 25
    assert result.all_iterates_certified


def test_short_supervised_lbfgs_instance_reduces_same_decoder_error() -> None:
    result = run_supervised_instance(
        control_vertices=5,
        image_resolution=32,
        steps=3,
        backend="direct",
        optimizer_name="lbfgs",
        dtype=torch.float64,
        seed=4704,
        target_strength=0.25,
        learning_rate=0.8,
    )

    assert result.final_objective < 0.7 * result.initial_objective
    assert result.primal_solves == result.adjoint_solves + 1
    assert result.global_solves == result.primal_solves + result.adjoint_solves
    assert result.global_solves > 3
    assert result.all_iterates_certified
    assert result.optimizer_name == "lbfgs"


def test_prebuilt_target_is_reused_without_setup_solve_or_mutation() -> None:
    mesh = structured_rectangle(4, 4)
    target = build_directed_target(
        mesh,
        image_height=24,
        image_width=24,
        seed=5705,
        strength=0.2,
        height=0.9,
    )
    control_before = target.control.clone()
    dense_before = target.dense.clone()

    first = run_supervised_instance(
        control_vertices=5,
        image_resolution=24,
        steps=2,
        backend="direct",
        optimizer_name="adam",
        dtype=torch.float64,
        seed=999,
        target_strength=9.0,
        learning_rate=0.05,
        target=target,
    )
    second = run_supervised_instance(
        control_vertices=5,
        image_resolution=24,
        steps=2,
        backend="directed_iterative",
        optimizer_name="adam",
        dtype=torch.float64,
        seed=1,
        target_strength=0.01,
        learning_rate=0.05,
        target=target,
    )

    assert first.target_setup_solves == second.target_setup_solves == 0
    assert first.target_setup_seconds == second.target_setup_seconds == 0.0
    torch.testing.assert_close(target.control, control_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(target.dense, dense_before, atol=0.0, rtol=0.0)


def test_reported_direct_solve_counts_match_superlu_calls(monkeypatch) -> None:
    mesh = structured_rectangle(4, 4)
    target = build_directed_target(
        mesh,
        image_height=24,
        image_width=24,
        seed=6706,
        strength=0.2,
        height=0.9,
    )
    original_splu = direct_module.sparse_linalg.splu
    calls = {"primal": 0, "adjoint": 0}

    class CountingFactor:
        def __init__(self, factor) -> None:
            self.factor = factor

        def solve(self, rhs, trans="N"):
            key = "adjoint" if trans == "T" else "primal"
            calls[key] += 1
            return self.factor.solve(rhs, trans=trans)

    def counting_splu(*args, **kwargs):
        return CountingFactor(original_splu(*args, **kwargs))

    monkeypatch.setattr(direct_module.sparse_linalg, "splu", counting_splu)
    result = run_supervised_instance(
        control_vertices=5,
        image_resolution=24,
        steps=2,
        backend="direct",
        optimizer_name="adam",
        dtype=torch.float64,
        seed=0,
        target_strength=1.0,
        learning_rate=0.05,
        target=target,
    )

    assert calls == {"primal": 3, "adjoint": 2}
    assert result.primal_solves == calls["primal"]
    assert result.adjoint_solves == calls["adjoint"]
    assert result.global_solves == sum(calls.values())


def test_prebuilt_image_target_is_detached_and_dense_control_mismatch_is_rejected() -> None:
    mesh = structured_rectangle(4, 4)
    target = build_directed_target(
        mesh,
        image_height=24,
        image_width=24,
        seed=7707,
        strength=0.2,
        height=1.0,
    )
    differentiable_target = replace(target, dense=target.dense.clone().requires_grad_())

    result = run_image_instance(
        control_vertices=5,
        image_resolution=24,
        steps=2,
        backend="direct",
        optimizer_name="adam",
        dtype=torch.float64,
        seed=0,
        target_strength=1.0,
        learning_rate=0.05,
        image_name="smooth_blobs",
        target=differentiable_target,
    )
    assert result.global_solves == 5

    inconsistent = replace(target, dense=torch.zeros_like(target.dense))
    with np.testing.assert_raises_regex(ValueError, "dense.*control|interpolation"):
        run_supervised_instance(
            control_vertices=5,
            image_resolution=24,
            steps=1,
            backend="direct",
            optimizer_name="adam",
            dtype=torch.float64,
            seed=0,
            target_strength=1.0,
            learning_rate=0.05,
            target=inconsistent,
        )
