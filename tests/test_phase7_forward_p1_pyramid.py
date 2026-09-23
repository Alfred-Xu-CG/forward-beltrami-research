"""Fixed-grid topology, exact prolongation, and smooth-target reachability."""

import math

import torch

from qcopt.neural_bijection.dense import ForwardP1Pyramid, exact_dyadic_p1_refine
from qcopt.neural_bijection.dense.colored_vertex_relaxation import SafeColoredVertexRelaxation
from qcopt.neural_bijection.dense.convex_quad import certify_convex_quad_output
from qcopt.neural_bijection.dense import evaluate_structured_p1_with_jacobian


def _identity(side: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    axis = torch.arange(side, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None]


def _smooth_target(grid: torch.Tensor) -> torch.Tensor:
    x, y = grid[..., 0], grid[..., 1]
    bump = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    return torch.stack((x + 0.02 * bump, y - 0.015 * bump), dim=-1)


def test_exact_refine_preserves_continuous_coarse_p1_and_vjp() -> None:
    coarse = _smooth_target(_identity(9)).requires_grad_()
    fine = exact_dyadic_p1_refine(coarse)
    assert fine.shape == (1, 17, 17, 2)
    assert torch.equal(fine[:, ::2, ::2], coarse)
    query = torch.rand(1, 200, 2, generator=torch.Generator().manual_seed(701), dtype=torch.float64)
    coarse_value, coarse_jac = evaluate_structured_p1_with_jacobian(coarse, query)
    fine_value, fine_jac = evaluate_structured_p1_with_jacobian(fine, query)
    assert (coarse_value - fine_value).abs().max() < 1e-14
    assert (coarse_jac - fine_jac).abs().max() < 1e-12
    fine.square().mean().backward()
    assert torch.isfinite(coarse.grad).all() and coarse.grad.abs().sum() > 0


def test_one_refinement_safe_logits_reach_independent_smooth_target() -> None:
    coarse = _smooth_target(_identity(17))
    fine_base = exact_dyadic_p1_refine(coarse)
    fine_target = _smooth_target(_identity(33))
    delta = fine_target - fine_base
    alpha, h = 2.0, 1.0 / 32
    assert delta.abs().max() < alpha * h
    logits = torch.atanh(delta[:, 1:-1, 1:-1] / (alpha * h)).detach().requires_grad_()
    output = SafeColoredVertexRelaxation(33, safety_fraction=0.85, motion_mode="radial", raw_span=alpha)(
        fine_base, logits,
    )
    assert (output - fine_target).abs().max() < 1e-14
    assert certify_convex_quad_output(output) > 0
    objective = (output * torch.randn_like(output)).sum()
    gradient = torch.autograd.grad(objective, logits)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_pyramid_all_finite_latents_preserve_faces_and_boundary() -> None:
    torch.manual_seed(7781)
    decoder = ForwardP1Pyramid(5, 33, seed_passes=2)
    seed = [torch.randn(2, 3, 3, 2, dtype=torch.float64) for _ in range(2)]
    levels = [torch.randn(2, n - 2, n - 2, 2, dtype=torch.float64) for n in decoder.level_sides]
    levels[-1].requires_grad_()
    output = decoder(seed, levels)
    assert output.shape == (2, 33, 33, 2)
    assert certify_convex_quad_output(output) > 0
    output.square().mean().backward()
    assert torch.isfinite(levels[-1].grad).all()
    assert levels[-1].grad.abs().sum() > 0


def test_pyramid_mask_ignores_old_vertex_logits() -> None:
    decoder = ForwardP1Pyramid(5, 9, seed_passes=0)
    base = torch.zeros(1, 7, 7, 2, dtype=torch.float64)
    changed = base.clone()
    changed[:, 1::2, 1::2] = 7.0  # interior positions of even/even global vertices
    original = decoder([], [base])
    output = decoder([], [changed])
    assert torch.equal(original, output)


def test_small_pyramid_vjp_matches_finite_difference() -> None:
    torch.manual_seed(429)
    decoder = ForwardP1Pyramid(5, 9, seed_passes=1)
    seed = [0.1 * torch.randn(1, 3, 3, 2, dtype=torch.float64)]
    level = (0.1 * torch.randn(1, 7, 7, 2, dtype=torch.float64)).requires_grad_()
    cotangent = torch.randn(1, 9, 9, 2, dtype=torch.float64)
    direction = torch.randn_like(level)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return (decoder(seed, [value]) * cotangent).sum()

    gradient = torch.autograd.grad(objective(level), level)[0]
    step = 1e-6
    numerical = (objective(level + step * direction) - objective(level - step * direction)) / (2 * step)
    analytic = (gradient * direction).sum()
    assert torch.allclose(numerical, analytic, atol=1e-6, rtol=1e-5)
