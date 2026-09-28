"""All-F1, all-F2 and mixed Q1 schedules share exact refinement semantics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qcopt.neural_bijection.dense.digital_q1 import q1_dyadic_refine, validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.scheduled_q1_pyramid import ScheduledQ1Pyramid


def _fields(model: ScheduledQ1Pyramid, *, seed: int, fine_zero: bool = False):
    torch.manual_seed(seed)
    return tuple(tuple(
        (torch.zeros if fine_zero and stage > 0 else torch.randn)(
            1, side - 2, side - 2, 2, dtype=torch.float32,
        ).requires_grad_()
        for _ in range(model.passes_per_stage[stage]))
        for stage, side in enumerate(model.sides))


@pytest.mark.parametrize("schedule", ["11", "22", "21", "12"])
def test_scheduled_33_outputs_safe_map_with_finite_vjp(schedule, tmp_path):
    model = ScheduledQ1Pyramid(17, 33, schedule)
    fields = _fields(model, seed=74)
    mapped = model(fields)
    axis = torch.arange(33) / 32
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    assert validate_q1_map(mapped, identity)["nonpositive_corners"] == 0
    (mapped.square().mean()).backward()
    assert all(field.grad is not None and bool(torch.isfinite(field.grad).all())
               for stage in fields for field in stage)
    path = tmp_path / f"{schedule}.npz"
    np.savez_compressed(path, vertices=mapped.detach().numpy(),
                        boundary_reference=identity.numpy())
    assert certify_q1_binary_map(path)["valid"]


@pytest.mark.parametrize("mode", ["1", "2"])
def test_zero_fine_latents_exactly_refine_previous_q1_map(mode):
    coarse = ScheduledQ1Pyramid(17, 17, mode)
    full = ScheduledQ1Pyramid(17, 33, mode + "1")
    seed_fields = _fields(coarse, seed=12)
    fine_fields = (torch.zeros(1, 31, 31, 2),)
    expected = q1_dyadic_refine(coarse(seed_fields))
    actual = full((seed_fields[0], fine_fields))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_invalid_schedule_is_rejected():
    with pytest.raises(ValueError, match="schedule"):
        ScheduledQ1Pyramid(17, 33, "1")


@pytest.mark.parametrize("schedule,active", [
    ("11111", 130050), ("21111", 130482),
    ("22211", 141970), ("22222", 341426),
])
def test_active_count_excludes_unselected_and_old_vertex_logits(schedule, active):
    model = ScheduledQ1Pyramid(17, 257, schedule)
    assert model.active_latent_scalar_count_per_sample() == active


@pytest.mark.parametrize("mode,repeats", [("1", 3), ("2", 2)])
def test_repeated_seed_safe_passes_have_independent_vjps(mode, repeats, tmp_path):
    model = ScheduledQ1Pyramid(17, 33, mode + "1", seed_repeats=repeats)
    assert model.passes_per_stage[0] == repeats * (1 if mode == "1" else 4)
    fields = _fields(model, seed=61, fine_zero=True)
    output = model(fields)
    axis = torch.arange(33) / 32
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    assert validate_q1_map(output, identity)["nonpositive_corners"] == 0
    output.square().mean().backward()
    assert all(field.grad is not None and bool(torch.isfinite(field.grad).all())
               for field in fields[0])
    archive = tmp_path / "repeated.npz"
    np.savez_compressed(archive, vertices=output.detach().numpy(),
                        boundary_reference=identity.numpy())
    assert certify_q1_binary_map(archive)["valid"]


def test_one_seed_repeat_is_exactly_default():
    default = ScheduledQ1Pyramid(17, 33, "21")
    explicit = ScheduledQ1Pyramid(17, 33, "21", seed_repeats=1)
    fields = _fields(default, seed=117)
    torch.testing.assert_close(default(fields), explicit(fields), rtol=0, atol=0)


def test_nonpositive_seed_repeat_is_rejected():
    with pytest.raises(ValueError, match="seed_repeats"):
        ScheduledQ1Pyramid(17, 33, "11", seed_repeats=0)
