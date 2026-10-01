"""Small decisive fixtures for shared-direction safety and the true VJP."""

import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_update import (
    CoordinatedQ1Update, interpolate_proposal, single_direction_corner_change,
)


def grid(rows=3, columns=4, dtype=torch.float64):
    y, x = torch.meshgrid(torch.linspace(0, 1, rows, dtype=dtype),
                          torch.linspace(0, 1, columns, dtype=dtype), indexing="ij")
    return torch.stack((x, y), dim=-1)[None]


def independent_corners(vertices):
    # Deliberately recompute oriented triangles by a separate explicit loop.
    result = []
    for row in range(vertices.shape[1] - 1):
        cells = []
        for col in range(vertices.shape[2] - 1):
            a, b = vertices[:, row, col], vertices[:, row, col + 1]
            d, c = vertices[:, row + 1, col], vertices[:, row + 1, col + 1]
            determinants = []
            for p, q, r in ((a, b, d), (a, b, c), (d, b, c), (a, c, d)):
                determinants.append(torch.linalg.det(torch.stack((q - p, r - p), dim=-1)))
            cells.append(torch.stack(determinants, dim=-1))
        result.append(torch.stack(cells, dim=1))
    return torch.stack(result, dim=1)


def test_arbitrary_geometry_affine_identity_all_corners_non_square():
    torch.manual_seed(141)
    base = grid() + 0.04 * torch.randn(1, 3, 4, 2, dtype=torch.float64)
    raw = torch.randn(1, 3, 4, dtype=torch.float64)
    e = torch.tensor((0.6, -0.8), dtype=torch.float64)
    delta = single_direction_corner_change(base, raw, e)
    expected = independent_corners(base + raw[..., None] * e) - independent_corners(base)
    torch.testing.assert_close(delta, expected, atol=1e-15, rtol=1e-13)


@pytest.mark.parametrize("mode", ["radial", "analytic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_preservation_and_usable_zero_gradient(mode, dtype):
    base = grid(dtype=dtype)
    proposal = torch.zeros(base.shape[:-1], dtype=dtype, requires_grad=True)
    result = CoordinatedQ1Update(mode=mode)(base, proposal)
    assert torch.equal(result.vertices, base)
    result.vertices[:, 1, 1, 0].backward()
    assert torch.isfinite(proposal.grad).all()
    assert proposal.grad[0, 1, 1] == 1
    assert torch.count_nonzero(proposal.grad[:, 0]) == 0


@pytest.mark.parametrize("mode", ["radial", "analytic"])
def test_large_proposal_protects_both_diagonals_fixed_boundary(mode):
    torch.manual_seed(43)
    base = grid(5, 7, torch.float32)
    proposal = torch.randn(base.shape[:-1], dtype=base.dtype) * 100
    result = CoordinatedQ1Update(direction=(0.6, 0.8), mode=mode,
                                 minimum_jacobian=0.07)(base, proposal)
    qref = independent_corners(base.double())
    assert (independent_corners(result.vertices.double()) / qref > 0.07).all()
    assert torch.equal(result.vertices[:, 0], base[:, 0])
    assert torch.equal(result.vertices[:, -1], base[:, -1])
    assert torch.equal(result.vertices[:, :, 0], base[:, :, 0])
    assert torch.equal(result.vertices[:, :, -1], base[:, :, -1])


@pytest.mark.parametrize("mode", ["radial", "analytic"])
@pytest.mark.parametrize("direction", [(1, 0), (0, -1)])
def test_sliding_preserves_tangency_corners_and_all_ordered_gaps(mode, direction):
    torch.manual_seed(98)
    base = grid(4, 7)
    proposal = torch.randn(base.shape[:-1], dtype=base.dtype) * 20
    result = CoordinatedQ1Update(direction=direction, mode=mode, boundary="sliding",
                                 minimum_boundary_gap=0.2)(base, proposal)
    out = result.vertices
    assert torch.equal(out[:, 0, :, 1], base[:, 0, :, 1])
    assert torch.equal(out[:, -1, :, 1], base[:, -1, :, 1])
    assert torch.equal(out[:, :, 0, 0], base[:, :, 0, 0])
    assert torch.equal(out[:, :, -1, 0], base[:, :, -1, 0])
    for row, col in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert torch.equal(out[:, row, col], base[:, row, col])
    for row in (0, -1):
        assert (torch.diff(out[:, row, :, 0], dim=1) > 0.2 / 6).all()
    for col in (0, -1):
        assert (torch.diff(out[:, :, col, 1], dim=1) > 0.2 / 3).all()


def test_radial_inverse_encoding_recovers_single_direction_feasible_target():
    base = grid(7, 9)
    x, y = base[..., 0], base[..., 1]
    amplitude = 0.09 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    layer = CoordinatedQ1Update(minimum_jacobian=0.01)
    amplitude = layer._mask(amplitude)
    probe = layer(base, amplitude)
    assert 0 < probe.gauge.item() < 1
    raw = amplitude / (1 - probe.gauge[:, None, None])
    encoded = layer(base, raw)
    torch.testing.assert_close(encoded.vertices, base + amplitude[..., None] *
                               torch.tensor((1., 0.)), atol=1e-15, rtol=1e-13)


@pytest.mark.parametrize("mode", ["radial", "analytic"])
def test_true_gradient_proposal_and_current_geometry_unique_active(mode):
    torch.manual_seed(777)
    base = grid(4, 5)
    base[:, 1:-1, 1:-1] += 0.01 * torch.randn(1, 2, 3, 2, dtype=base.dtype)
    raw = 2 * torch.randn(base.shape[:-1], dtype=base.dtype)
    base.requires_grad_()
    raw.requires_grad_()
    layer = CoordinatedQ1Update((0.8, 0.6), mode=mode)
    weight = torch.randn(base.shape, dtype=base.dtype)
    result = layer(base, raw)
    assert result.gauge.item() > 1
    slack, delta, _ = layer._constraints(base, layer._mask(raw), grid(4, 5))
    active = ((-delta).clamp_min(0) / slack).flatten().sort(descending=True).values
    assert active[0] - active[1] > 1e-3  # no max/min branch ambiguity
    loss = (result.vertices * weight).sum()
    gradients = torch.autograd.grad(loss, (base, raw))
    for index, variable in enumerate((base, raw)):
        tangent = torch.randn_like(variable)
        step = 1e-6
        args_plus = [base.detach(), raw.detach()]
        args_minus = [base.detach(), raw.detach()]
        args_plus[index] = variable.detach() + step * tangent
        args_minus[index] = variable.detach() - step * tangent
        plus = (layer(*args_plus).vertices * weight).sum()
        minus = (layer(*args_minus).vertices * weight).sum()
        fd = (plus - minus) / (2 * step)
        ad = (gradients[index] * tangent).sum()
        torch.testing.assert_close(ad, fd, atol=2e-8, rtol=2e-6)


def test_analytic_limited_and_unlimited_step_branches():
    base = grid()
    raw = torch.zeros(base.shape[:-1], dtype=base.dtype)
    raw[:, 1, 1] = 1
    layer = CoordinatedQ1Update(mode="analytic", theta=0.9, minimum_jacobian=0.1)
    result = layer(base, raw, alpha_trial=2)
    torch.testing.assert_close(result.alpha_max, torch.tensor([0.3], dtype=base.dtype))
    torch.testing.assert_close(result.scale, torch.tensor([0.27], dtype=base.dtype))
    result = layer(base, raw, alpha_trial=0.05)
    assert result.scale.item() == 0.05


def test_analytic_tiny_proposals_do_not_differentiate_inactive_huge_reciprocal():
    # Batch mixes tiny anchor slack, a huge but inactive finite step limit,
    # an infinite step limit, and an active finite step limit.
    base = torch.cat((grid(4, 4)*1e-150, grid(4, 4), grid(4, 4), grid(4, 4))).requires_grad_()
    raw = torch.zeros(4, 4, 4, dtype=base.dtype)
    raw[1, 1, 1] = 1e-305
    raw[3, 1, 1] = 1.
    raw.requires_grad_()
    result = CoordinatedQ1Update(mode="analytic", minimum_jacobian=0)(base, raw)
    assert torch.isinf(result.alpha_max[0]) and torch.isinf(result.alpha_max[2])
    assert torch.isfinite(result.alpha_max[1]) and result.alpha_max[1] > 1e300
    assert torch.equal(result.scale[:3], torch.ones(3, dtype=base.dtype))
    assert result.scale[3] < 1
    grad_base, grad_raw = torch.autograd.grad(result.vertices[:, 1, 1, 0].sum(), (base, raw))
    assert torch.isfinite(grad_base).all() and torch.isfinite(grad_raw).all()
    torch.testing.assert_close(grad_raw[:3, 1, 1], torch.ones(3, dtype=base.dtype))


def test_analytic_batched_trial_steps_match_the_declared_minimum():
    base = grid().expand(3, -1, -1, -1).clone()
    raw = torch.zeros(base.shape[:-1], dtype=base.dtype)
    raw[:, 1, 1] = torch.tensor((0., .1, 1.), dtype=base.dtype)
    raw.requires_grad_()
    trial = torch.tensor((0., .05, 2.), dtype=base.dtype, requires_grad=True)
    result = CoordinatedQ1Update(mode="analytic", theta=.9)(base, raw, alpha_trial=trial)
    expected = torch.minimum(trial.detach(), .9*result.alpha_max.detach())
    torch.testing.assert_close(result.scale, expected)
    gradient = torch.autograd.grad(result.vertices.sum(), (raw, trial))
    assert all(torch.isfinite(part).all() for part in gradient)


def test_invalid_anchor_nonfinite_and_incompatible_sliding_fail():
    base = grid()
    raw = torch.zeros(base.shape[:-1], dtype=base.dtype)
    bad = base.clone()
    bad[:, 1, 1] = -1
    with pytest.raises(ValueError, match="strictly positive"):
        CoordinatedQ1Update()(bad, raw)
    raw[:, 1, 1] = float("nan")
    with pytest.raises(ValueError, match="finite inputs"):
        CoordinatedQ1Update()(base, raw)
    with pytest.raises(ValueError, match="axis direction"):
        CoordinatedQ1Update((1, 1), boundary="sliding")


def test_proposal_interpolation_keeps_physical_units_on_non_square_grid():
    coarse = torch.full((1, 3, 4), 0.012, dtype=torch.float64, requires_grad=True)
    dense = interpolate_proposal(coarse, (17, 23))
    torch.testing.assert_close(dense, torch.full_like(dense, 0.012))
    dense.mean().backward()
    assert torch.isfinite(coarse.grad).all()
    torch.testing.assert_close(coarse.grad.sum(), torch.tensor(1., dtype=coarse.dtype))


def test_coarse_boundary_mask_gives_coarse_width_transition():
    coarse = torch.ones(1, 3, 3, dtype=torch.float64, requires_grad=True)
    dense = interpolate_proposal(coarse, (9, 9), boundary="fixed")
    assert dense[0, 0].count_nonzero() == 0
    assert dense[0, :, 0].count_nonzero() == 0
    assert dense[0, 1, 4] == .25
    assert dense[0, 4, 4] == 1
    dense.sum().backward()
    assert coarse.grad[0, 0].count_nonzero() == 0
    assert coarse.grad[0, :, 0].count_nonzero() == 0
