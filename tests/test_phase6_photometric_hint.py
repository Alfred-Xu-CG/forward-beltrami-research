"""Photometric hint is dimensionally consistent and differentiates through A4."""

from __future__ import annotations

import torch

from qcopt.neural_bijection.dense import (
    SafeColoredVertexRelaxation,
    certify_convex_quad_output,
    local_photometric_logits,
    physical_image_gradient,
)


def test_physical_gradient_on_linear_image() -> None:
    line = torch.linspace(0.0, 1.0, 33, dtype=torch.float64)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    image = (2 * xx - 3 * yy)[None, None]
    gradient = physical_image_gradient(image)
    assert torch.allclose(gradient[:, 0, 1:-1, 1:-1], torch.full((1, 31, 31), 2.0, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(gradient[:, 1, 1:-1, 1:-1], torch.full((1, 31, 31), -3.0, dtype=torch.float64), atol=1e-12)


def test_hint_to_radial_map_vjp_and_orientation() -> None:
    torch.manual_seed(20260923)
    line = torch.linspace(0.0, 1.0, 33, dtype=torch.float64)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    moving = (torch.sin(6 * torch.pi * xx + 2 * torch.pi * yy) + 0.3 * torch.cos(4 * torch.pi * yy))[None, None]
    fixed = moving.roll(shifts=1, dims=-1)
    control_line = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    cy, cx = torch.meshgrid(control_line, control_line, indexing="ij")
    identity = torch.stack((cx, cy), dim=-1)[None]
    bump = (torch.sin(torch.pi * cx) * torch.sin(torch.pi * cy))[None, :, :, None]
    base = (identity + 5e-4 * bump * torch.randn_like(identity)).requires_grad_(True)
    layer = SafeColoredVertexRelaxation(17, safety_fraction=0.85, motion_mode="radial")
    cotangent = torch.randn_like(base)

    def objective(control: torch.Tensor) -> torch.Tensor:
        hint = local_photometric_logits(fixed, moving, control, window=3, ridge=10.0, raw_span=2.0)
        mapped = layer(control, hint)
        return (mapped * cotangent).sum()

    value = objective(base)
    value.backward()
    direction = bump * torch.randn_like(base)
    direction /= direction.norm()
    epsilon = 1e-5
    finite_difference = (
        objective(base.detach() + epsilon * direction).item()
        - objective(base.detach() - epsilon * direction).item()
    ) / (2 * epsilon)
    analytic = (base.grad * direction).sum().item()
    assert abs(finite_difference - analytic) < 3e-4
    with torch.no_grad():
        mapped = layer(base, local_photometric_logits(fixed, moving, base, window=3, ridge=10.0, raw_span=2.0))
        assert certify_convex_quad_output(mapped) > 0
