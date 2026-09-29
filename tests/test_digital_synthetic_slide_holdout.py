"""Small protocol and coordinate checks for the slide-texture capability probe."""

from __future__ import annotations

import pytest
import torch

from tools.digital_q1_synthetic_slide_holdout import (
    analytic_map, point_grid, run, synthesize,
)
from tools.digital_q1_synthetic_slide_baseline import optimize_baseline


def test_identity_sampled_at_pixel_centers() -> None:
    device = torch.device("cpu")
    centers = point_grid(8, centers=True, device=device)
    nodes = point_grid(17, centers=False, device=device)
    assert torch.equal(centers[0, 0, 0], torch.tensor((1 / 16, 1 / 16)))
    assert torch.equal(nodes[0, 0, 0], torch.tensor((0., 0.)))
    image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
    fixed = synthesize(image, torch.zeros((1, 6)), centers)
    assert torch.allclose(fixed, image, atol=2e-7, rtol=0)


def test_analytic_map_has_exact_fixed_boundary() -> None:
    grid = point_grid(33, centers=False, device=torch.device("cpu"))
    coefficients = torch.tensor(((.018, .008, -.008, -.018, -.008, .008),))
    mapped = analytic_map(grid, coefficients)
    assert torch.equal(mapped[:, 0, :, :], grid[:, 0, :, :])
    assert torch.equal(mapped[:, -1, :, :], grid[:, -1, :, :])
    assert torch.equal(mapped[:, :, 0, :], grid[:, :, 0, :])
    assert torch.equal(mapped[:, :, -1, :], grid[:, :, -1, :])


def test_negative_texture_alias_cannot_bypass_holdout() -> None:
    textures = torch.rand((3, 1, 32, 32))
    with pytest.raises(ValueError, match="canonical nonnegative"):
        run(textures=textures, train_texture_indices=[2],
            test_texture_indices=[-1], side=33, steps=1, batch=1,
            train_count=1, test_count=1, device="cpu")
    with pytest.raises(ValueError, match="canonical nonnegative"):
        optimize_baseline(textures=textures, test_texture_indices=[-1],
                          side=33, steps=1, batch=1, test_count=1,
                          device="cpu", learning_rate=.01)
