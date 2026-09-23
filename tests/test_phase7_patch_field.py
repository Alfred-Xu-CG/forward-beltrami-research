"""Quadratic simultaneous-face bound and original-grid P1 checks."""

import math
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


def test_floor_free_patch_passes_exactly_reach_small_independent_target() -> None:
    side = 17
    base = _identity(side)
    x, y = base[..., 0], base[..., 1]
    bump = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    target = base + torch.stack((0.002 * bump, -0.001 * bump), dim=-1)
    layer = StaggeredPatchP1Layer(side, 8, minimum_jacobian=None)
    assigned = torch.zeros(side * side, dtype=torch.bool)
    fields = []
    for patch_pass in layer.passes:
        ids = patch_pass.interior_ids
        new = ~assigned[ids]
        raw = torch.zeros(1, ids.numel(), 2, dtype=base.dtype)
        raw[:, new] = (target - base).reshape(1, -1, 2)[:, ids[new]]
        span = patch_pass.raw_span * patch_pass.patch_cells / (side - 1)
        fields.append(torch.atanh(raw / span))
        assigned[ids] = True
    output = layer(base, tuple(fields))
    assert assigned.reshape(side, side)[1:-1, 1:-1].all()
    assert (output - target).abs().max() < 1e-14
    assert certify_convex_quad_output(output) > 0


def test_split_patch_pyramid_matches_full_output_and_vjp() -> None:
    torch.manual_seed(930)
    whole = ForwardPatchP1Pyramid(17, 65, patch_cells=8)
    coarse = ForwardPatchP1Pyramid(17, 33, patch_cells=8)
    seed = tuple(
        0.02 * torch.randn(1, 15, 15, 2, dtype=torch.float64, requires_grad=True)
        for _ in range(4)
    )
    levels = tuple(
        tuple(
            0.02 * torch.randn(1, side - 2, side - 2, 2,
                               dtype=torch.float64, requires_grad=True)
            for _ in range(4)
        )
        for side in whole.level_sides
    )
    direct = whole(seed, levels)
    first = coarse(seed, levels[:1])
    resumed = whole.forward_from(first, levels[1:], start_index=1)
    assert torch.equal(direct, resumed)
    inputs = seed + tuple(z for group in levels for z in group)
    weights = torch.randn_like(direct)
    grad_direct = torch.autograd.grad((direct * weights).sum(), inputs, retain_graph=True)
    grad_resumed = torch.autograd.grad((resumed * weights).sum(), inputs)
    assert all(torch.allclose(a, b, atol=1e-14, rtol=1e-14)
               for a, b in zip(grad_direct, grad_resumed))


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


def test_compact_patch_latents_match_full_field_and_receive_gradients() -> None:
    torch.manual_seed(716)
    base = _identity(17)
    layer = SafePatchFieldPass(17, 8, offset_row=4, offset_column=4)
    full = 0.05 * torch.randn(1, 15, 15, 2, dtype=torch.float64)
    compact = full.reshape(1, -1, 2)[:, layer.latent_ids.reshape(-1)].clone().requires_grad_()
    full_output = layer(base, full)
    compact_output = layer(base, compact)
    assert torch.allclose(full_output, compact_output, atol=0, rtol=0)
    compact_output.square().mean().backward()
    assert compact.grad is not None and compact.grad.abs().sum() > 0


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


def test_patch_pyramid_accepts_compact_active_latents() -> None:
    torch.manual_seed(7601)
    decoder = ForwardPatchP1Pyramid(17, 33, patch_cells=8)
    seed = tuple(
        0.02 * torch.randn(1, patch_pass.interior_ids.numel(), 2, dtype=torch.float64)
        for patch_pass in decoder.seed_layer.passes
    )
    level = tuple(
        (0.02 * torch.randn(1, patch_pass.interior_ids.numel(), 2,
                            dtype=torch.float64)).requires_grad_()
        for patch_pass in decoder.level_layers[0].passes
    )
    output = decoder(seed, (level,))
    assert output.shape == (1, 33, 33, 2)
    assert certify_convex_quad_output(output) > 0.05 - 1e-12
    output.square().mean().backward()
    assert all(field.grad is not None and torch.isfinite(field.grad).all() for field in level)
