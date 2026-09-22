"""Topology and VJP tests for nested convex-cell fine-grid P1 maps."""

from __future__ import annotations

import pytest
import torch

from qcopt.neural_bijection.dense import (
    HierarchicalConvexQuadFreeCenterLayer,
    HierarchicalConvexQuadLayer,
    certify_convex_quad_output,
)


def _latent(side: int, batch: int, dtype: torch.dtype, scale: float) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    generator = torch.Generator().manual_seed(20260923)
    result = []
    for current in HierarchicalConvexQuadLayer(side).latent_sides:
        horizontal = (scale * torch.randn(batch, current, current - 1, generator=generator, dtype=dtype)).requires_grad_()
        vertical = (scale * torch.randn(batch, current - 1, current, generator=generator, dtype=dtype)).requires_grad_()
        result.append((horizontal, vertical))
    return tuple(result)


@pytest.mark.parametrize("side", [5, 17, 257])
def test_random_hierarchical_map_is_fine_grid_p1_homeomorphism(side: int) -> None:
    layer = HierarchicalConvexQuadLayer(side)
    latent = _latent(side, batch=2, dtype=torch.float32, scale=0.6)
    output = layer(latent)
    assert output.shape == (2, side, side, 2)
    assert certify_convex_quad_output(output) > 0
    line = torch.arange(side) / (side - 1)
    assert torch.equal(output[:, 0, :, 0], line.expand(2, -1))
    assert torch.equal(output[:, :, 0, 1], line.expand(2, -1))


def test_extreme_logits_retain_represented_face_orientation() -> None:
    layer = HierarchicalConvexQuadLayer(17)
    latent = _latent(17, batch=1, dtype=torch.float64, scale=25.0)
    output = layer(latent)
    assert certify_convex_quad_output(output) > 0


def test_directional_vjp_matches_central_difference() -> None:
    layer = HierarchicalConvexQuadLayer(5)
    horizontal = torch.tensor([[[0.0, 0.0], [0.3, -0.2], [0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    vertical = torch.tensor([[[0.0, -0.4, 0.0], [0.0, 0.2, 0.0]]], dtype=torch.float64, requires_grad=True)
    weight = torch.linspace(-0.5, 0.7, 50, dtype=torch.float64).reshape(1, 5, 5, 2)
    loss = (layer(((horizontal, vertical),)) * weight).sum()
    grads = torch.autograd.grad(loss, (horizontal, vertical))
    direction_h = torch.randn_like(horizontal)
    direction_v = torch.randn_like(vertical)
    predicted = (grads[0] * direction_h).sum() + (grads[1] * direction_v).sum()
    step = 1e-6
    plus = (layer(((horizontal + step * direction_h, vertical + step * direction_v),)) * weight).sum()
    minus = (layer(((horizontal - step * direction_h, vertical - step * direction_v),)) * weight).sum()
    observed = (plus - minus) / (2 * step)
    assert torch.allclose(predicted, observed, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("side", [5, 17, 257])
def test_free_center_hierarchy_is_represented_homeomorphism(side: int) -> None:
    generator = torch.Generator().manual_seed(80877)
    layer = HierarchicalConvexQuadFreeCenterLayer(side)
    root = torch.randn(2, 1, 1, 2, generator=generator)
    latents = tuple(
        (
            0.7 * torch.randn(2, current, current - 1, generator=generator),
            0.7 * torch.randn(2, current - 1, current, generator=generator),
            0.7 * torch.randn(2, current - 1, current - 1, 2, generator=generator),
        )
        for current in layer.latent_sides
    )
    output = layer(root, latents)
    assert output.shape == (2, side, side, 2)
    assert certify_convex_quad_output(output) > 0


def test_free_root_center_can_move_while_boundary_stays_fixed() -> None:
    layer = HierarchicalConvexQuadFreeCenterLayer(5)
    root = torch.tensor([[[[1.0, -0.7]]]], dtype=torch.float64, requires_grad=True)
    latents = ((
        torch.zeros(1, 3, 2, dtype=torch.float64, requires_grad=True),
        torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True),
        torch.zeros(1, 2, 2, 2, dtype=torch.float64, requires_grad=True),
    ),)
    output = layer(root, latents)
    assert torch.linalg.vector_norm(output[0, 2, 2] - torch.tensor([0.5, 0.5], dtype=torch.float64)) > 0.1
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(root.grad).all()
    assert root.grad.abs().sum() > 0


def test_free_center_directional_vjp_matches_finite_difference() -> None:
    layer = HierarchicalConvexQuadFreeCenterLayer(5)
    root = torch.tensor([[[[0.3, -0.2]]]], dtype=torch.float64, requires_grad=True)
    horizontal = torch.tensor([[[0.0, 0.0], [0.3, -0.2], [0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    vertical = torch.tensor([[[0.0, -0.4, 0.0], [0.0, 0.2, 0.0]]], dtype=torch.float64, requires_grad=True)
    center = torch.tensor([[[[0.2, 0.1], [-0.2, 0.4]], [[0.1, -0.3], [0.2, 0.2]]]], dtype=torch.float64, requires_grad=True)
    weight = torch.linspace(-0.5, 0.7, 50, dtype=torch.float64).reshape(1, 5, 5, 2)
    loss = (layer(root, ((horizontal, vertical, center),)) * weight).sum()
    variables = (root, horizontal, vertical, center)
    grads = torch.autograd.grad(loss, variables)
    directions = tuple(torch.randn_like(variable) for variable in variables)
    predicted = sum((gradient * direction).sum() for gradient, direction in zip(grads, directions))
    step = 1e-6
    plus_inputs = tuple(variable + step * direction for variable, direction in zip(variables, directions))
    minus_inputs = tuple(variable - step * direction for variable, direction in zip(variables, directions))
    plus = (layer(plus_inputs[0], ((plus_inputs[1], plus_inputs[2], plus_inputs[3]),)) * weight).sum()
    minus = (layer(minus_inputs[0], ((minus_inputs[1], minus_inputs[2], minus_inputs[3]),)) * weight).sum()
    observed = (plus - minus) / (2 * step)
    assert torch.allclose(predicted, observed, rtol=1e-7, atol=1e-9)


def test_standalone_certificate_rejects_coordinates_outside_unit_square() -> None:
    layer = HierarchicalConvexQuadFreeCenterLayer(5)
    root = torch.zeros(1, 1, 1, 2)
    latents = ((torch.zeros(1, 3, 2), torch.zeros(1, 2, 3), torch.zeros(1, 2, 2, 2)),)
    control = layer(root, latents).clone()
    control[0, 2, 2, 0] = 2.0
    with pytest.raises(ValueError, match="outside the unit square"):
        certify_convex_quad_output(control)
