"""Independent NumPy geometry and gradient attacks for the Q1 update paths.

This checker deliberately does not use the production cross-product or area
helpers.  Its corner formula evaluates the two partial derivatives of the
bilinear map directly at each corner.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    SafeColoredQ1Relaxation,
    SafePatchQ1Pass,
    q1_corner_determinants,
)


def independent_q1_corners(vertices: np.ndarray) -> np.ndarray:
    """Evaluate det(partial_s Phi, partial_t Phi) at four cell corners."""
    if vertices.ndim != 4 or vertices.shape[-1] != 2:
        raise ValueError("expected (batch, rows, columns, xy)")
    batch, rows, columns, _ = vertices.shape
    result = np.empty((batch, rows - 1, columns - 1, 4), dtype=np.float64)
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for k in range(batch):
        for row in range(rows - 1):
            for column in range(columns - 1):
                southwest = vertices[k, row, column]
                southeast = vertices[k, row, column + 1]
                northeast = vertices[k, row + 1, column + 1]
                northwest = vertices[k, row + 1, column]
                mixed = southwest - southeast + northeast - northwest
                for corner, (s, t) in enumerate(corners):
                    ds = southeast - southwest + t * mixed
                    dt = northwest - southwest + s * mixed
                    result[k, row, column, corner] = (
                        ds[0] * dt[1] - ds[1] * dt[0]
                    )
    return result


def regular_grid(side: int) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None]


def check_module_corners(vertices: torch.Tensor) -> np.ndarray:
    independent = independent_q1_corners(vertices.detach().numpy())
    actual = q1_corner_determinants(vertices).detach().numpy()
    np.testing.assert_allclose(actual, independent, rtol=0.0, atol=2e-15)
    return independent


def directional_vjp_check(
    layer: torch.nn.Module, base: torch.Tensor, logits: torch.Tensor,
    direction: torch.Tensor, cotangent: torch.Tensor,
) -> tuple[float, float]:
    variable = logits.detach().clone().requires_grad_()
    output = layer(base, variable)
    analytical = torch.autograd.grad((output * cotangent).sum(), variable)[0]
    vjp = float((analytical * direction).sum())
    for step in (1e-4, 1e-5, 1e-6):
        with torch.no_grad():
            plus = float((layer(base, logits + step * direction) * cotangent).sum())
            minus = float((layer(base, logits - step * direction) * cotangent).sum())
        finite_difference = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(vjp, finite_difference, rtol=1e-5, atol=1e-8)
    return vjp, finite_difference


def test_old_p1_and_center_difference_false_positive() -> None:
    # Rational SW, SE, NE, NW coordinates.  The old SW--NE triangles and
    # centered Jacobian are positive, but the SW Q1 derivative is negative.
    a = (Fraction(0), Fraction(0))
    b = (Fraction(1), Fraction(0))
    c = (Fraction(1), Fraction(1))
    d = (Fraction(-1), Fraction(-1, 2))

    def det(u: tuple[Fraction, Fraction], v: tuple[Fraction, Fraction]) -> Fraction:
        return u[0] * v[1] - u[1] * v[0]

    def sub(u: tuple[Fraction, Fraction], v: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
        return u[0] - v[0], u[1] - v[1]

    assert det(sub(b, a), sub(c, a)) == 1
    assert det(sub(c, a), sub(d, a)) == Fraction(1, 2)
    center_ds = tuple((u + v) / 2 for u, v in zip(sub(b, a), sub(c, d)))
    center_dt = tuple((u + v) / 2 for u, v in zip(sub(d, a), sub(c, b)))
    assert det(center_ds, center_dt) == Fraction(3, 4)
    vertices = torch.tensor([[[a, b], [d, c]]], dtype=torch.float64)
    corners = check_module_corners(vertices)[0, 0, 0]
    np.testing.assert_array_equal(corners, [-0.5, 1.0, 2.0, 0.5])


def test_independent_all_cell_corners_and_f1_vjp() -> None:
    base = regular_grid(5)
    logits = torch.zeros((1, 3, 3, 2), dtype=torch.float64)
    logits[0, 0, 0] = torch.tensor((1.3, -0.25), dtype=torch.float64)
    logits[0, 1, 1] = torch.tensor((-0.8, 0.65), dtype=torch.float64)
    logits[0, 2, 2] = torch.tensor((0.4, 0.2), dtype=torch.float64)
    layer = SafeColoredQ1Relaxation(5, raw_span=2.0)
    output = layer(base, logits)
    assert np.min(check_module_corners(output)) > 0.0
    raw_first = 2.0 / 4.0 * torch.tanh(logits[0, 0, 0])
    assert torch.linalg.vector_norm(output[0, 1, 1] - base[0, 1, 1]) < torch.linalg.vector_norm(raw_first)

    generator = torch.Generator().manual_seed(29091)
    direction = torch.randn(logits.shape, generator=generator, dtype=torch.float64)
    cotangent = torch.randn(output.shape, generator=generator, dtype=torch.float64)
    directional_vjp_check(layer, base, logits, direction, cotangent)


def test_independent_all_cell_corners_and_f2_vjp() -> None:
    base = regular_grid(5)
    logits = torch.zeros((1, 3, 3, 2), dtype=torch.float64)
    logits[0, :, :, 0] = torch.tensor(
        ((1.1, 1.25, 1.42), (1.32, 0.86, 1.05), (1.54, 1.16, 0.93)),
        dtype=torch.float64,
    )
    logits[0, :, :, 1] = torch.tensor(
        ((0.04, -0.2, 0.11), (-0.07, 0.03, 0.09), (0.15, -0.06, 0.02)),
        dtype=torch.float64,
    )
    layer = SafePatchQ1Pass(5, 4, raw_span=0.5)
    output = layer(base, logits)
    assert np.min(check_module_corners(output)) > 0.0
    assert not torch.equal(output[0, 1, 1], base[0, 1, 1])

    generator = torch.Generator().manual_seed(29092)
    direction = torch.randn(logits.shape, generator=generator, dtype=torch.float64)
    cotangent = torch.randn(output.shape, generator=generator, dtype=torch.float64)
    directional_vjp_check(layer, base, logits, direction, cotangent)


def test_f2_detached_safety_scale_is_caught_by_vjp_attack() -> None:
    # Uniform x proposals saturate a patch-boundary Q1 constraint.  Along the
    # common-logit direction, scale and raw motion cancel.  A detached scale
    # reports a spurious nonzero gradient despite an unchanged output.
    base = regular_grid(5)
    logits = torch.zeros((1, 3, 3, 2), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits[..., 0] = 1.2
    layer = SafePatchQ1Pass(5, 4, raw_span=0.5)
    output = layer(base, logits)
    check_module_corners(output)
    raw = 0.5 * torch.tanh(logits[..., 0])
    observed_scale = (output[0, 1, 1, 0] - base[0, 1, 1, 0]) / raw[0, 0, 0]
    assert 0.0 < float(observed_scale) < 0.99

    true_objective = output[:, 1:4, 1:4, 0].sum()
    true_gradient = torch.autograd.grad(true_objective, logits, retain_graph=True)[0]
    fake_objective = (base[:, 1:4, 1:4, 0] + observed_scale.detach() * raw).sum()
    fake_gradient = torch.autograd.grad(fake_objective, logits)[0]
    common_x_direction = torch.zeros_like(logits)
    common_x_direction[..., 0] = 1.0
    true_directional = float((true_gradient * common_x_direction).sum())
    fake_directional = float((fake_gradient * common_x_direction).sum())
    _, finite_difference = directional_vjp_check(
        layer, base, logits.detach(), common_x_direction,
        torch.stack((torch.ones_like(base[..., 0]), torch.zeros_like(base[..., 0])), dim=-1),
    )
    np.testing.assert_allclose(true_directional, finite_difference, atol=1e-8)
    assert abs(fake_directional - finite_difference) > 1e-2
