"""Small, decisive checks for the image-pair Q1 predictor."""

from __future__ import annotations

import torch
import pytest

from qcopt.neural_bijection.dense.q1_image_network import (
    PositiveAffineHead,
    Q1ImageRegistrationNetwork,
)


def test_affine_head_is_identity_at_zero_and_positive_at_extreme_finite_inputs():
    head = PositiveAffineHead(width=8)
    features = torch.randn(2, 8, 9, 9, dtype=torch.float64)
    head = head.to(torch.float64)
    matrix, offset = head(features)
    eye = torch.eye(2, dtype=torch.float64).expand(2, -1, -1)
    assert torch.equal(matrix, eye)
    assert torch.equal(offset, torch.zeros_like(offset))
    with torch.no_grad():
        head.output.bias.copy_(torch.tensor([1000., -1000., 1000., -1000., 1000., -1000.]))
    matrix, offset = head(features)
    determinant = torch.linalg.det(matrix)
    assert bool(torch.isfinite(matrix).all())
    assert bool(torch.isfinite(offset).all())
    assert bool((determinant > 0.5).all())


@pytest.mark.parametrize("kwargs", [
    {"max_log_scale": 1000.},
    {"max_shear": 1e30},
    {"max_translation": float("inf")},
])
def test_affine_head_rejects_unsafe_configured_bounds(kwargs):
    with pytest.raises(ValueError, match="bounds"):
        PositiveAffineHead(width=8, **kwargs)


def test_image_network_identity_q1_and_gradients():
    torch.manual_seed(71)
    model = Q1ImageRegistrationNetwork(seed_side=5, final_side=9, width=4,
                                       feature_side=16, flow_hint=False)
    fixed = torch.rand(1, 1, 16, 16)
    moving = torch.rand(1, 1, 16, 16)
    residual, matrix, offset = model(fixed, moving)
    axis = torch.arange(9) / 8
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    assert torch.allclose(residual, identity, atol=1e-7)
    assert torch.equal(matrix, torch.eye(2)[None])
    assert torch.equal(offset, torch.zeros_like(offset))
    target = identity + torch.tensor([0.03, -0.02])
    mapped = model.apply_affine(residual, matrix, offset)
    loss = (mapped - target).square().mean()
    loss.backward()
    assert model.affine_head.output.weight.grad is not None
    assert bool(torch.isfinite(model.affine_head.output.weight.grad).all())
    assert float(model.affine_head.output.weight.grad.abs().sum()) > 0
    assert model.encoder.seed_heads[0].weight.grad is not None
    assert bool(torch.isfinite(model.encoder.seed_heads[0].weight.grad).all())
