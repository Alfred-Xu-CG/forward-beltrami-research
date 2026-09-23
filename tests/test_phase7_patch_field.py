"""Quadratic simultaneous-face bound and original-grid P1 checks."""

import torch

from qcopt.neural_bijection.dense import ForwardPatchP1Pyramid, SafePatchFieldPass, StaggeredPatchP1Layer
from qcopt.neural_bijection.dense.convex_quad import certify_convex_quad_output


def _identity(side: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    axis = torch.arange(side, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None]


def test_one_patch_updates_multiple_vertices_of_same_triangle() -> None:
    base = _identity(17)
    torch.manual_seed(771)
    latent = torch.randn(1, 15, 15, 2, dtype=torch.float64)
    layer = SafePatchFieldPass(17, 8)
    output = layer(base, latent)
    assert (output[:, 2, 2] - base[:, 2, 2]).abs().sum() > 0
    assert (output[:, 2, 3] - base[:, 2, 3]).abs().sum() > 0
    assert (output[:, 3, 3] - base[:, 3, 3]).abs().sum() > 0
    assert certify_convex_quad_output(output) >= 0.05 - 1e-12


def test_four_staggered_patch_passes_preserve_all_faces_under_large_latents() -> None:
    torch.manual_seed(8501)
    base = _identity(33, torch.float32).expand(2, -1, -1, -1)
    logits = tuple(5 * torch.randn(2, 31, 31, 2) for _ in range(4))
    output = StaggeredPatchP1Layer(33, 8)(base, logits)
    assert certify_convex_quad_output(output) > 0
    assert torch.equal(output[:, 0], base[:, 0])
    assert torch.equal(output[:, -1], base[:, -1])
    assert torch.equal(output[:, :, 0], base[:, :, 0])
    assert torch.equal(output[:, :, -1], base[:, :, -1])


def test_staggered_patch_interiors_cover_every_nonboundary_vertex() -> None:
    for side in (17, 33, 65):
        layer = StaggeredPatchP1Layer(side, 8)
        observed = set()
        for patch_pass in layer.passes:
            observed.update(patch_pass.interior_ids.tolist())
        expected = {
            row * side + column
            for row in range(1, side - 1) for column in range(1, side - 1)
        }
        assert observed == expected


def test_patch_field_vjp_matches_directional_difference() -> None:
    torch.manual_seed(5177)
    layer = SafePatchFieldPass(17, 8)
    base = _identity(17)
    logits = (0.03 * torch.randn(1, 15, 15, 2, dtype=torch.float64)).requires_grad_()
    direction = torch.randn_like(logits)
    cotangent = torch.randn_like(base)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return (layer(base, value) * cotangent).sum()

    gradient = torch.autograd.grad(objective(logits), logits)[0]
    step = 1e-6
    numerical = (objective(logits + step * direction) - objective(logits - step * direction)) / (2 * step)
    analytic = (gradient * direction).sum()
    assert torch.allclose(analytic, numerical, atol=1e-6, rtol=2e-4)
    assert gradient.abs().sum() > 0


def test_near_floor_patch_corner_has_bounded_vjp() -> None:
    side = 17
    base = _identity(side, torch.float32).clone()
    # (row=1,col=8) is on a patch boundary; a corner face of that patch has
    # all three displacement entries zero, even while other patch faces move.
    base[0, 1, 8, 1] = 0.05 / (side - 1)
    assert abs(certify_convex_quad_output(base) - 0.05) < 2e-6
    base.requires_grad_()
    torch.manual_seed(662)
    logits = (0.1 * torch.randn(1, side - 2, side - 2, 2)).requires_grad_()
    layer = SafePatchFieldPass(side, 8, minimum_jacobian=0.05)
    output = layer(base, logits)
    # A zero-displacement face exactly on the floor must not freeze the
    # otherwise safe motion of a whole patch.
    assert (output[0, 2, 2] - base[0, 2, 2]).abs().sum() > 1e-5
    objective = (output * torch.randn_like(output)).sum()
    base_grad, latent_grad = torch.autograd.grad(objective, (base, logits))
    assert torch.isfinite(base_grad).all()
    assert torch.isfinite(latent_grad).all()
    assert base_grad.abs().max() < 1e8


def test_patch_pyramid_stays_one_fixed_grid_p1_and_backpropagates() -> None:
    torch.manual_seed(7512)
    decoder = ForwardPatchP1Pyramid(17, 33, patch_cells=8)
    seed = tuple(0.03 * torch.randn(1, 15, 15, 2, dtype=torch.float64) for _ in range(4))
    level = tuple((0.03 * torch.randn(1, 31, 31, 2, dtype=torch.float64)).requires_grad_()
                  for _ in range(4))
    output = decoder(seed, (level,))
    assert output.shape == (1, 33, 33, 2)
    assert certify_convex_quad_output(output) >= 0.05 - 1e-12
    output.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in level)
