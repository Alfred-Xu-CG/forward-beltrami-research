"""Local feedback can use the actual bilinear interpolation derivative."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.photometric_hint import (
    bilinear_image_value_and_gradient, local_photometric_logits,
)


def test_exact_bilinear_value_and_query_gradient_match_grid_sample() -> None:
    torch.manual_seed(417)
    image = torch.randn(2, 1, 11, 13, dtype=torch.float64)
    coordinates = (
        .11 + .78 * torch.rand(2, 5, 7, 2, dtype=torch.float64)
    ).requires_grad_()
    value, gradient = bilinear_image_value_and_gradient(image, coordinates)
    reference = F.grid_sample(
        image, 2 * coordinates - 1, mode="bilinear",
        padding_mode="border", align_corners=True,
    )
    assert torch.allclose(value, reference, atol=2e-15, rtol=2e-15)
    cotangent = torch.randn_like(reference)
    actual = torch.autograd.grad((reference * cotangent).sum(), coordinates)[0]
    expected = gradient.permute(0, 2, 3, 1) * cotangent.permute(0, 2, 3, 1)
    assert torch.allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_exact_bilinear_local_hint_is_finite_and_differentiable() -> None:
    torch.manual_seed(418)
    fixed = torch.rand(1, 1, 19, 19, dtype=torch.float64)
    moving = torch.rand(1, 1, 19, 19, dtype=torch.float64).requires_grad_()
    axis = torch.arange(9, dtype=torch.float64) / 8
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None]
    logits = local_photometric_logits(
        fixed, moving, base, window=3, ridge=1., raw_span=2.,
        gradient_mode="bilinear_exact",
    )
    assert bool(torch.isfinite(logits).all())
    derivative = torch.autograd.grad(logits.square().mean(), moving)[0]
    assert bool(torch.isfinite(derivative).all())
    assert float(derivative.abs().amax()) > 0
