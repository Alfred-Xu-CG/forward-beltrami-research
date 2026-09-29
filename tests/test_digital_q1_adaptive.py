"""Independent corner signs and VJP for current-geometry F1 proposals."""

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveEllipsoidQ1Relaxation, AdaptiveSoftRadialQ1Relaxation,
)


def _identity(side: int, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.linspace(0, 1, side, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None]


def _independent_four_corners(vertices: np.ndarray) -> np.ndarray:
    out = []
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
    for row in range(vertices.shape[0] - 1):
        for col in range(vertices.shape[1] - 1):
            a, b = vertices[row, col], vertices[row, col + 1]
            c, d = vertices[row + 1, col + 1], vertices[row + 1, col]
            out.extend((cross(b - a, d - a), cross(b - a, c - b),
                        cross(c - d, c - b), cross(c - d, d - a)))
    return np.asarray(out)


def test_repeated_adaptive_updates_preserve_independent_q1_corner_floor():
    side = 17
    layer = AdaptiveEllipsoidQ1Relaxation(side)
    torch.manual_seed(1729)
    logits = torch.randn(1, side - 2, side - 2, 2, dtype=torch.float64) * 5
    mapped = _identity(side, torch.float64)
    for _ in range(4):
        mapped = layer(mapped, logits)
        corners = _independent_four_corners(mapped[0].detach().numpy())
        assert corners.min() > .05 / (side - 1) ** 2
        assert torch.equal(mapped[:, 0], _identity(side, torch.float64)[:, 0])
        assert torch.equal(mapped[:, -1], _identity(side, torch.float64)[:, -1])


def test_identity_equal_eigenvalues_have_finite_nonzero_vjp():
    side = 7
    layer = AdaptiveEllipsoidQ1Relaxation(side, minimum_jacobian=None)
    logits = torch.zeros(1, side - 2, side - 2, 2, dtype=torch.float64,
                         requires_grad=True)
    mapped = layer(_identity(side, torch.float64), logits)
    weight = torch.randn(mapped.shape, dtype=mapped.dtype)
    gradient, = torch.autograd.grad((mapped * weight).sum(), logits)
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().max()) > 0
    index = (0, 2, 2, 0)
    with torch.no_grad():
        plus = logits.detach().clone()
        minus = logits.detach().clone()
        plus[index] = 1e-5
        minus[index] = -1e-5
        directional = ((layer(_identity(side, torch.float64), plus) * weight).sum()
                       - (layer(_identity(side, torch.float64), minus) * weight).sum()
                       ) / 2e-5
    assert torch.allclose(gradient[index], directional, atol=1e-8, rtol=1e-5)


def test_float32_near_floor_257_has_finite_forward_and_both_vjps():
    side = 257
    base = _identity(side, torch.float32).requires_grad_(True)
    logits = torch.full((1, side - 2, side - 2, 2), .7,
                        dtype=torch.float32, requires_grad=True)
    floor = torch.nextafter(torch.tensor([1 / (side - 1) ** 2],
                                           dtype=torch.float32), torch.zeros(1))
    mapped = AdaptiveEllipsoidQ1Relaxation(side, minimum_jacobian=None)(
        base, logits, area_floor=floor,
    )
    grad_base, grad_logits = torch.autograd.grad(mapped.square().mean(),
                                                  (base, logits))
    assert bool(torch.isfinite(mapped).all())
    assert bool(torch.isfinite(grad_base).all())
    assert bool(torch.isfinite(grad_logits).all())


def test_float32_tiny_valid_rectangle_avoids_gram_overflow():
    base = (1e-10 * _identity(3, torch.float32)).requires_grad_(True)
    logits = torch.full((1, 1, 1, 2), .7, requires_grad=True)
    mapped = AdaptiveEllipsoidQ1Relaxation(3, minimum_jacobian=None)(base, logits)
    grad_base, grad_logits = torch.autograd.grad(mapped.square().sum(),
                                                  (base, logits))
    assert bool(torch.isfinite(mapped).all())
    assert bool(torch.isfinite(grad_base).all())
    assert bool(torch.isfinite(grad_logits).all())


def test_nonfinite_or_extreme_metric_parameters_rejected():
    for bad in (float("nan"), float("inf"), float("-inf"), 1e308, 1e-20):
        try:
            AdaptiveEllipsoidQ1Relaxation(3, ridge_fraction=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite ridge parameter accepted")


def test_current_edge_soft_radial_repeated_four_corners_and_vjp():
    side = 17
    torch.manual_seed(3389)
    layer = AdaptiveSoftRadialQ1Relaxation(side)
    base = _identity(side, torch.float64)
    logits = (4 * torch.randn(4, 1, side - 2, side - 2, 2,
                              dtype=torch.float64)).requires_grad_(True)
    mapped = base
    for round_index in range(4):
        mapped = layer(mapped, logits[round_index])
        corners = _independent_four_corners(mapped[0].detach().numpy())
        assert corners.min() > 0
        assert corners.min() >= .05 / (side - 1) ** 2 - 1e-12
    gradient, = torch.autograd.grad(mapped.square().mean(), logits)
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().max()) > 0


def test_current_edge_soft_radial_257_near_floor_vjps_finite():
    side = 257
    base = _identity(side, torch.float32).requires_grad_(True)
    logits = torch.full((1, side - 2, side - 2, 2), .7,
                        dtype=torch.float32, requires_grad=True)
    floor = torch.nextafter(torch.tensor([1 / (side - 1) ** 2],
                                           dtype=torch.float32), torch.zeros(1))
    mapped = AdaptiveSoftRadialQ1Relaxation(side, minimum_jacobian=None)(
        base, logits, area_floor=floor,
    )
    gradients = torch.autograd.grad(mapped.square().mean(), (base, logits))
    assert bool(torch.isfinite(mapped).all())
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
