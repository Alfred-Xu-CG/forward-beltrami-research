"""Tiny external-target fitting for both safe schedule operators."""

from __future__ import annotations

import pytest
import torch

from tools.digital_q1_schedule_fit import fit_schedule


@pytest.mark.parametrize("schedule", ["1", "2"])
def test_schedule_fit_improves_external_smooth_target(schedule):
    axis = torch.linspace(0, 1, 17)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    bubble = torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
    teacher = torch.stack((xx + .01 * bubble, yy - .005 * bubble), dim=-1)[None]
    mapped, report = fit_schedule(
        teacher, schedule=schedule, steps=4, learning_rate=.01, device="cpu",
    )
    assert mapped.shape == teacher.shape
    assert report["best_interior_vector_rmse"] < report["initial_interior_vector_rmse"]
    assert report["finite_gradient_steps"] == 4
    assert report["nonpositive_corners"] == 0
    assert report["boundary_max_error"] == 0


@pytest.mark.parametrize("schedule,expected_seed_fields", [("1", 3), ("2", 8)])
def test_schedule_fit_accepts_repeated_seed_fields(schedule, expected_seed_fields):
    axis = torch.linspace(0, 1, 17)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    bubble = torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
    teacher = torch.stack((xx + .08 * bubble, yy), dim=-1)[None]
    _, report = fit_schedule(
        teacher, schedule=schedule, steps=4, learning_rate=.01,
        device="cpu", seed_repeats=(3 if schedule == "1" else 2),
    )
    assert report["passes_per_stage"][0] == expected_seed_fields
    assert report["seed_repeats"] == (3 if schedule == "1" else 2)
    assert report["finite_gradient_steps"] == 4
    assert report["best_interior_vector_rmse"] < report["initial_interior_vector_rmse"]
