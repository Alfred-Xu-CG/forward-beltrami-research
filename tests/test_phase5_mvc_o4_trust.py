"""Decisive tests for the separately named O4 latent-trust variant."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import qcopt.neural_bijection.tutte.mvc_instance_optimization as optimization


def _run(*, method: str, steps: int = 1, **kwargs):
    return optimization.run_fixed_boundary_comparison(
        task="supervised_map",
        control_vertices=5,
        image_resolution=17,
        steps=steps,
        backend="direct",
        dtype=torch.float64,
        seed=521,
        target_strength=0.20,
        learning_rate=0.01,
        methods=(method,),
        **kwargs,
    ).results[method]


def test_baseline_o4_default_and_accounting_are_unchanged() -> None:
    result = _run(method="O4_covariance_retraction")

    assert result.failure is None
    assert (result.primal_attempts, result.primal_solves) == (3, 3)
    assert result.retraction_trial_attempts == 0
    assert result.retraction_trial_solves == 0
    assert result.retraction_extra_solves == 0
    assert result.minimum_accepted_retraction_scale is None
    assert result.trace[-1].accepted_retraction_scale is None


def test_trust_backtracks_a_failed_decode_and_counts_its_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = optimization._decode
    calls = 0

    def reject_first_trial(solver, logits, boundary, counters):
        nonlocal calls
        calls += 1
        if calls == 3:  # two initialization solves precede the first trial
            counters.primal_attempts += 1
            raise ValueError("synthetic full-step decoder rejection")
        return original(solver, logits, boundary, counters)

    monkeypatch.setattr(optimization, "_decode", reject_first_trial)
    result = _run(
        method=optimization.O4_TRUST_METHOD_NAME,
        o4_trust_backtrack_factor=0.5,
        o4_trust_max_trials=3,
        o4_trust_min_area_fraction=0.1,
    )

    assert result.failure is None
    assert result.completed_steps == 1
    assert (result.primal_attempts, result.primal_solves) == (4, 3)
    assert result.retraction_trial_attempts == 2
    assert result.retraction_trial_solves == 1
    assert result.retraction_extra_solves == 0
    assert result.minimum_accepted_retraction_scale == pytest.approx(0.5)
    assert result.trace[-1].accepted_retraction_scale == pytest.approx(0.5)
    assert result.trace[-1].retraction_trial_attempts == 2
    assert result.trace[-1].retraction_trial_solves == 1


def test_trust_counts_a_completed_geometry_rejection_as_an_extra_global_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = optimization._trust_candidate_metrics
    calls = 0

    def reject_first_geometry(mesh, control, target):
        nonlocal calls
        calls += 1
        metrics = original(mesh, control, target)
        if calls == 1:
            return replace(
                metrics,
                flip_count=1,
                global_injectivity_certificate=False,
            )
        return metrics

    monkeypatch.setattr(optimization, "_trust_candidate_metrics", reject_first_geometry)
    result = _run(
        method=optimization.O4_TRUST_METHOD_NAME,
        o4_trust_backtrack_factor=0.5,
        o4_trust_max_trials=3,
        o4_trust_min_area_fraction=0.1,
    )

    assert result.failure is None
    assert (result.primal_attempts, result.primal_solves) == (4, 4)
    assert result.global_solves == 4
    assert result.trace[-1].global_solves == 4
    assert result.retraction_trial_attempts == 2
    assert result.retraction_trial_solves == 2
    assert result.retraction_extra_solves == 1
    assert result.trace[-1].retraction_extra_solves == 1
    assert result.trace[-1].accepted_retraction_scale == pytest.approx(0.5)
    assert result.trace[-1].topology_certified


def test_trust_exhaustion_fails_closed_and_preserves_last_accepted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = optimization._trust_candidate_metrics

    def reject_all_geometry(mesh, control, target):
        metrics = original(mesh, control, target)
        return replace(
            metrics,
            minimum_area_ratio=0.0,
            global_injectivity_certificate=False,
        )

    monkeypatch.setattr(optimization, "_trust_candidate_metrics", reject_all_geometry)
    result = _run(
        method=optimization.O4_TRUST_METHOD_NAME,
        o4_trust_backtrack_factor=0.5,
        o4_trust_max_trials=2,
        o4_trust_min_area_fraction=0.5,
    )

    assert result.failure is not None
    assert result.failure.phase == "retraction_trust_search"
    assert result.completed_steps == 0
    assert (result.primal_attempts, result.primal_solves) == (4, 4)
    assert result.retraction_trial_attempts == 2
    assert result.retraction_trial_solves == 2
    assert result.retraction_extra_solves == 2
    assert result.final_objective == result.initial_objective
    torch.testing.assert_close(
        result.final_control, result.initial_control, atol=1.0e-14, rtol=1.0e-14
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("o4_trust_backtrack_factor", 1.0, "backtrack_factor"),
        ("o4_trust_backtrack_factor", 0.0, "backtrack_factor"),
        ("o4_trust_max_trials", 0, "max_trials"),
        ("o4_trust_min_area_fraction", 0.0, "min_area_fraction"),
        ("o4_trust_min_area_fraction", 1.1, "min_area_fraction"),
    ],
)
def test_trust_configuration_fails_closed(field: str, value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _run(method=optimization.O4_TRUST_METHOD_NAME, **{field: value})
