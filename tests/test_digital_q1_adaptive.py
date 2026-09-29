"""Independent corner signs and VJP for current-geometry F1 proposals."""

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import (
    AdaptiveEllipsoidQ1Relaxation, AdaptiveSoftRadialQ1Relaxation,
    AdaptivePatchQ1Pass, FixedSpanSoftRadialQ1Relaxation,
    SafePatchQ1Pass, StaggeredPatchQ1Layer,
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


def test_current_edge_f2_equals_fixed_h_at_identity_and_stays_safe_repeated() -> None:
    side, cells = 17, 4
    torch.manual_seed(2917)
    base = _identity(side, torch.float64)
    latent = torch.randn(1, side - 2, side - 2, 2, dtype=torch.float64) * 2
    fixed = SafePatchQ1Pass(side, cells, raw_span=.5)
    current = AdaptivePatchQ1Pass(side, cells, raw_span=.5)
    torch.testing.assert_close(fixed(base, latent), current(base, latent),
                               atol=1e-15, rtol=1e-12)

    axis = torch.linspace(0, 1, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    mapped = base.clone()
    mapped[..., 0] += .08 * torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
    assert _independent_four_corners(mapped[0].numpy()).min() > 0
    layer = StaggeredPatchQ1Layer(
        side, patch_cells=cells, proposal_mode="current_edge", raw_span=.5,
    )
    latents = [torch.randn_like(latent).requires_grad_() for _ in range(8)]
    for round_index in range(2):
        mapped = layer(mapped, tuple(latents[4 * round_index:4 * round_index + 4]))
        assert _independent_four_corners(mapped[0].detach().numpy()).min() > 0
        torch.testing.assert_close(mapped[:, 0], base[:, 0], atol=0, rtol=0)
        torch.testing.assert_close(mapped[:, -1], base[:, -1], atol=0, rtol=0)
    gradients = torch.autograd.grad(mapped.square().mean(), tuple(latents))
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert any(float(gradient.abs().max()) > 0 for gradient in gradients)


def test_current_edge_f2_rejects_extreme_raw_span() -> None:
    for value in (0, -1, 1e20, float("nan"), float("inf")):
        try:
            AdaptivePatchQ1Pass(5, 2, raw_span=value)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe or unsupported F2 raw span accepted")


def test_fixed_span_soft_radial_is_matched_safety_law_control() -> None:
    side = 17
    base = _identity(side, torch.float64)
    torch.manual_seed(2918)
    latent = torch.randn(1, side - 2, side - 2, 2, dtype=torch.float64)
    current_edge = AdaptiveSoftRadialQ1Relaxation(side)
    fixed_span = FixedSpanSoftRadialQ1Relaxation(side)
    flat = base.reshape(1, side * side, 2)
    source_floor = torch.tensor([.05 / (side - 1) ** 2], dtype=base.dtype)
    first_current = current_edge._update_color(flat, latent.reshape(1, -1, 2),
                                               source_floor, 0)
    first_fixed = fixed_span._update_color(flat, latent.reshape(1, -1, 2),
                                          source_floor, 0)
    torch.testing.assert_close(first_current, first_fixed, atol=1e-15, rtol=1e-12)
    mapped = base
    for _ in range(3):
        mapped = fixed_span(mapped, latent)
        assert _independent_four_corners(mapped[0].numpy()).min() > 0


def test_soft_radial_local_inverse_reaches_nearby_map_exactly():
    side = 9
    layer = AdaptiveSoftRadialQ1Relaxation(side)
    base = _identity(side, torch.float64)
    axis = torch.linspace(0, 1, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    target = base.clone()
    target[..., 0] += .001 * torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
    current = base.reshape(1, side * side, 2)
    floor = torch.tensor([.05 / (side - 1) ** 2], dtype=torch.float64)
    for color in range(4):
        vertices = getattr(layer, f"_vertices_{color}")
        opposite = vertices[:, None, None] + layer._opposite_offsets[None]
        local = ((vertices // side - 1) * (side - 2) + vertices % side - 1)
        point = current[:, vertices]
        start = current[:, opposite[..., 0]]
        end = current[:, opposite[..., 1]]
        edge = end - start
        relative = point[:, :, None] - start
        areas = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
        budget = torch.minimum(.75 * areas, areas - floor[:, None, None])
        assert bool((budget > 0).all())
        desired = target.reshape(1, side * side, 2)[:, vertices] - point
        adverse = -(edge[..., 0] * desired[:, :, None, 1]
                    - edge[..., 1] * desired[:, :, None, 0]) / budget
        maximum = adverse.clamp_min(0).amax(dim=-1)
        assert bool((maximum < 1).all())
        raw = desired / (1 - maximum)[..., None]
        horizontal = (current[:, vertices + 1] - current[:, vertices - 1]) / 2
        vertical = (current[:, vertices + side] - current[:, vertices - side]) / 2
        edge_matrix = torch.stack((horizontal, vertical), dim=-1)
        hyperbolic = torch.linalg.solve(edge_matrix, raw[..., None])[..., 0] / layer.raw_span
        assert bool((hyperbolic.abs() < 1).all())
        logits = torch.zeros(1, side - 2, side - 2, 2, dtype=torch.float64)
        logits.reshape(1, -1, 2)[:, local] = torch.atanh(hyperbolic)
        current = layer._update_color(current, logits.reshape(1, -1, 2), floor, color)
    torch.testing.assert_close(current.reshape_as(base), target, rtol=0, atol=5e-16)


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


def test_f2_post_safety_gain_controls_saturated_step_and_vjp():
    side = 17
    base = _identity(side, torch.float64)
    logits = torch.full((1, side - 2, side - 2, 2), 2.,
                        dtype=torch.float64, requires_grad=True)
    full = AdaptivePatchQ1Pass(side, 8, raw_span=.5, accepted_gain=1.)
    quarter = AdaptivePatchQ1Pass(side, 8, raw_span=.5, accepted_gain=.25)
    broad = full(base, logits)
    damped = quarter(base, logits)
    assert float((broad - base).abs().amax()) > 1e-4
    torch.testing.assert_close(damped - base, .25 * (broad - base),
                               rtol=1e-12, atol=1e-15)
    assert _independent_four_corners(damped[0].detach().numpy()).min() > 0
    gradient, = torch.autograd.grad(damped.square().mean(), logits)
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0


def test_current_edge_f1_commutes_with_positive_affine_without_absolute_floor():
    side = 17
    base = _identity(side, torch.float64)
    torch.manual_seed(941)
    logits = (.12 * torch.randn(1, side - 2, side - 2, 2,
                                   dtype=torch.float64)).requires_grad_()
    matrix = torch.tensor([[1.3, .2], [.1, .8]], dtype=torch.float64)
    offset = torch.tensor([.07, -.04], dtype=torch.float64)
    transform = lambda x: torch.einsum("ij,...j->...i", matrix, x) + offset
    current = AdaptiveSoftRadialQ1Relaxation(side, minimum_jacobian=None)
    reference = current(base, logits)
    transformed = current(transform(base), logits)
    torch.testing.assert_close(transformed, transform(reference), rtol=0, atol=2e-15)
    cotangent = torch.randn_like(reference)
    pullback_cotangent = torch.einsum("ij,...i->...j", matrix, cotangent)
    transformed_vjp, = torch.autograd.grad((transformed * cotangent).sum(), logits,
                                            retain_graph=True)
    reference_vjp, = torch.autograd.grad((reference * pullback_cotangent).sum(), logits)
    torch.testing.assert_close(transformed_vjp, reference_vjp, rtol=2e-12, atol=2e-14)
    fixed = FixedSpanSoftRadialQ1Relaxation(side, minimum_jacobian=None)
    assert float((fixed(transform(base), logits) - transform(fixed(base, logits))).abs().max()) > 1e-4


def test_current_edge_f2_commutes_with_positive_affine_when_guard_inactive():
    side = 17
    base = _identity(side, torch.float64)
    torch.manual_seed(942)
    logits = tuple((.12 * torch.randn(1, side - 2, side - 2, 2,
                                         dtype=torch.float64)).requires_grad_()
                   for _ in range(4))
    matrix = torch.tensor([[1.3, .2], [.1, .8]], dtype=torch.float64)
    offset = torch.tensor([.07, -.04], dtype=torch.float64)
    transform = lambda x: torch.einsum("ij,...j->...i", matrix, x) + offset
    current = StaggeredPatchQ1Layer(side, patch_cells=8, proposal_mode="current_edge",
                                    minimum_jacobian=None)
    reference = current(base, logits)
    transformed = current(transform(base), logits)
    torch.testing.assert_close(transformed, transform(reference), rtol=0, atol=3e-15)
    cotangent = torch.randn_like(reference)
    pullback_cotangent = torch.einsum("ij,...i->...j", matrix, cotangent)
    transformed_vjp = torch.autograd.grad((transformed * cotangent).sum(), logits,
                                           retain_graph=True)
    reference_vjp = torch.autograd.grad((reference * pullback_cotangent).sum(), logits)
    for left, right in zip(transformed_vjp, reference_vjp, strict=True):
        torch.testing.assert_close(left, right, rtol=3e-12, atol=2e-14)
    fixed = StaggeredPatchQ1Layer(side, patch_cells=8, proposal_mode="fixed_h",
                                  minimum_jacobian=None)
    assert float((fixed(transform(base), logits) - transform(fixed(base, logits))).abs().max()) > 1e-4
