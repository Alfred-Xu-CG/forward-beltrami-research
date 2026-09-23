"""Quadratic simultaneous-face bound and original-grid P1 checks."""

import math
import torch

from qcopt.neural_bijection.dense import ForwardPatchP1Pyramid, SafePatchFieldPass, StaggeredPatchP1Layer
import qcopt.neural_bijection.dense as dense
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


def test_tied_staggered_patch_latent_reaches_small_shifted_target_and_backpropagates() -> None:
    assert hasattr(dense, "TiedStaggeredPatchP1Layer")
    side = 33
    base = _identity(side)
    x, y = base[..., 0], base[..., 1]
    window_x = torch.sin(math.pi * (x - 0.3125) / 0.25).square()
    window_y = torch.sin(math.pi * (y - 0.3125) / 0.25).square()
    mask = ((x >= 0.3125) & (x <= 0.5625) &
            (y >= 0.3125) & (y <= 0.5625))
    displacement = 0.001 * window_x * window_y * mask
    target = base + torch.stack((displacement, displacement), dim=-1)
    layer = dense.TiedStaggeredPatchP1Layer(side, patch_cells=8, cycles=2,
                                           minimum_jacobian=0.05)
    span = 0.5 * 8 / (side - 1)
    latent = torch.atanh(((target - base)[:, 1:-1, 1:-1] / span)).requires_grad_()
    output = layer(base, latent)
    assert (output - target).abs().amax() < 1e-12
    assert certify_convex_quad_output(output) >= 0.05 - 1e-12
    cotangent = torch.randn_like(output)
    gradient = torch.autograd.grad((output * cotangent).sum(), latent)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().amax() > 0


def test_residual_staggered_patch_latent_catches_up_across_seams() -> None:
    assert hasattr(dense, "ResidualStaggeredPatchP1Layer")
    side = 33
    base = _identity(side)
    x, y = base[..., 0], base[..., 1]
    tx, ty = (x - 0.3125) / 0.25, (y - 0.3125) / 0.25
    wx = torch.where((tx >= 0) & (tx <= 1), torch.sin(math.pi * tx).square(), 0)
    wy = torch.where((ty >= 0) & (ty <= 1), torch.sin(math.pi * ty).square(), 0)
    displacement = 0.006 * wx * wy
    target = base + torch.stack((displacement, displacement), dim=-1)
    layer = dense.ResidualStaggeredPatchP1Layer(
        side, patch_cells=8, cycles=2, minimum_jacobian=0.05,
    )
    span = 0.5 * 8 / (side - 1)
    latent = torch.atanh((target - base)[:, 1:-1, 1:-1] / span).requires_grad_()
    output = layer(base, latent)
    assert (output - target).abs().amax() < 1e-12
    assert certify_convex_quad_output(output) > 0.05
    gradient = torch.autograd.grad((output * torch.randn_like(output)).sum(), latent)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().amax() > 0


def test_residual_staggered_patch_extreme_latents_and_directional_vjp() -> None:
    side = 33
    base = _identity(side)
    layer = dense.ResidualStaggeredPatchP1Layer(
        side, patch_cells=8, cycles=2, minimum_jacobian=0.05,
    )
    torch.manual_seed(9127)
    extreme = 50 * torch.randn(1, side - 2, side - 2, 2, dtype=base.dtype)
    output = layer(base, extreme)
    assert torch.isfinite(output).all()
    assert certify_convex_quad_output(output) >= 0.05 - 1e-10
    assert torch.equal(output[:, 0], base[:, 0])
    assert torch.equal(output[:, -1], base[:, -1])
    small = (0.002 * torch.randn_like(extreme)).requires_grad_()
    direction = torch.randn_like(small)
    cotangent = torch.randn_like(base)
    objective = lambda value: (layer(base, value) * cotangent).sum()
    analytic = (torch.autograd.grad(objective(small), small)[0] * direction).sum()
    step = 1e-6
    numerical = (objective(small + step * direction)
                 - objective(small - step * direction)) / (2 * step)
    assert torch.allclose(analytic, numerical, atol=1e-6, rtol=2e-4)


def test_coarse_patch_fine_vertex_layer_keeps_absolute_floor_and_mixed_dtype_vjp() -> None:
    assert hasattr(dense, "CoarsePatchFineVertexP1Layer")
    torch.manual_seed(1182)
    layer = dense.CoarsePatchFineVertexP1Layer(
        17, 65, coarse_patch_cells=4, coarse_cycles=2,
        minimum_jacobian=0.05, compute_dtype=torch.float64,
    )
    coarse_latent = (0.1 * torch.randn(2, 15, 15, 2)).requires_grad_()
    fine_latent = (0.1 * torch.randn(2, 63, 63, 2)).requires_grad_()
    output = layer(coarse_latent, fine_latent)
    identity = _identity(65, torch.float64).expand(2, -1, -1, -1)
    assert output.dtype == torch.float64 and output.shape == (2, 65, 65, 2)
    assert torch.equal(output[:, 0], identity[:, 0])
    assert torch.equal(output[:, -1], identity[:, -1])
    assert certify_convex_quad_output(output) >= 0.05 - 1e-10
    grads = torch.autograd.grad((output * torch.randn_like(output)).sum(),
                                (coarse_latent, fine_latent))
    assert all(torch.isfinite(g).all() and g.abs().amax() > 0 for g in grads)


def test_fixed_p1_output_filter_rejects_folds_and_preserves_valid_gradients() -> None:
    assert hasattr(dense, "certify_p1_or_identity")
    identity = _identity(5, torch.float64).expand(2, -1, -1, -1)
    candidate = identity.clone().detach()
    candidate[0, 2, 2, 0] += 0.01
    candidate[1, 2, 2, 0] += 0.8
    candidate.requires_grad_()
    filtered, valid = dense.certify_p1_or_identity(candidate, identity)
    assert valid.tolist() == [True, False]
    assert torch.equal(filtered[1], identity[1])
    loss = filtered[:, 2, 2, 0].sum()
    grad = torch.autograd.grad(loss, candidate)[0]
    assert grad[0, 2, 2, 0] == 1 and grad[1].abs().amax() == 0


def test_joint_coarse_fine_p1_vjp_matches_central_differences() -> None:
    torch.manual_seed(1183)
    layer = dense.CoarsePatchFineVertexP1Layer(
        17, 65, coarse_patch_cells=4, coarse_cycles=2,
        minimum_jacobian=0.05, compute_dtype=torch.float64,
    )
    coarse = (0.02 * torch.randn(1, 15, 15, 2, dtype=torch.float64)).requires_grad_()
    fine = (0.02 * torch.randn(1, 63, 63, 2, dtype=torch.float64)).requires_grad_()
    cotangent = torch.randn(1, 65, 65, 2, dtype=torch.float64)
    def objective(zc: torch.Tensor, zf: torch.Tensor) -> torch.Tensor:
        return (layer(zc, zf) * cotangent).sum()
    analytic = torch.autograd.grad(objective(coarse, fine), (coarse, fine))
    step = 1e-5
    for tensor, other, gradient, at_coarse in (
        (coarse, fine, analytic[0], True),
        (fine, coarse, analytic[1], False),
    ):
        direction = torch.zeros_like(tensor)
        direction[(0, 7, 7, 0) if at_coarse else (0, 31, 31, 1)] = 1.0
        with torch.no_grad():
            plus = objective(tensor + step * direction, other) if at_coarse else objective(other, tensor + step * direction)
            minus = objective(tensor - step * direction, other) if at_coarse else objective(other, tensor - step * direction)
        numerical = (plus - minus) / (2 * step)
        directional = (gradient * direction).sum()
        assert torch.allclose(directional, numerical, atol=1e-6, rtol=1e-4)


def test_joint_coarse_fine_checkpoint_preserves_output_and_vjp() -> None:
    torch.manual_seed(1184)
    plain = dense.CoarsePatchFineVertexP1Layer(
        17, 65, coarse_patch_cells=4, compute_dtype=torch.float64,
    )
    saved = dense.CoarsePatchFineVertexP1Layer(
        17, 65, coarse_patch_cells=4, compute_dtype=torch.float64,
        checkpoint_fine=True,
    )
    zc = (0.05 * torch.randn(1, 15, 15, 2)).requires_grad_()
    zf = (0.05 * torch.randn(1, 63, 63, 2)).requires_grad_()
    cotangent = torch.randn(1, 65, 65, 2, dtype=torch.float64)
    a = plain(zc, zf)
    ga = torch.autograd.grad((a * cotangent).sum(), (zc, zf))
    b = saved(zc, zf)
    gb = torch.autograd.grad((b * cotangent).sum(), (zc, zf))
    assert torch.equal(a, b)
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_certified_pyramid_preserves_valid_output_and_gradients() -> None:
    assert hasattr(dense, "CertifiedForwardP1Pyramid")
    torch.manual_seed(1185)
    plain = dense.ForwardP1Pyramid(5, 17, seed_passes=2)
    certified = dense.CertifiedForwardP1Pyramid(5, 17, seed_passes=2)
    seed = [(0.03 * torch.randn(1, 3, 3, 2, dtype=torch.float64)).requires_grad_()
            for _ in range(2)]
    levels = [(0.03 * torch.randn(1, n - 2, n - 2, 2, dtype=torch.float64))
              .requires_grad_() for n in (9, 17)]
    a = plain(seed, levels)
    b = certified(seed, levels)
    cotangent = torch.randn_like(a)
    ga = torch.autograd.grad((a * cotangent).sum(), seed + levels)
    gb = torch.autograd.grad((b * cotangent).sum(), seed + levels)
    assert torch.equal(a, b)
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_residual_patch_pyramid_reaches_nonseam_smooth_target() -> None:
    assert hasattr(dense, "ResidualPatchP1Pyramid")
    strength = 0.02
    def target(side: int) -> torch.Tensor:
        base = _identity(side)
        x, y = base[0, ..., 0], base[0, ..., 1]
        u = strength * torch.sin(2 * math.pi * x) * torch.sin(math.pi * y)
        v = -0.8 * strength * torch.sin(math.pi * x) * torch.sin(2 * math.pi * y)
        return base + torch.stack((u, v), dim=-1)[None]
    layer = dense.ResidualPatchP1Pyramid(
        17, 65, patch_cells=4, seed_passes=4,
        minimum_jacobian=0.05,
    )
    old = _identity(17)
    goal = target(17)
    seed = []
    for step in range(4):
        raw = ((goal - old) / 4)[:, 1:-1, 1:-1] / (0.5 * 4 / 16)
        seed.append(torch.atanh(raw).requires_grad_())
    levels = []
    prior = goal
    for side in (33, 65):
        next_goal = target(side)
        prolonged = dense.exact_dyadic_p1_refine(prior)
        raw = (next_goal - prolonged)[:, 1:-1, 1:-1] / (0.5 * 4 / (side - 1))
        levels.append(torch.atanh(raw).requires_grad_())
        prior = next_goal
    output = layer(seed, levels)
    assert torch.allclose(output, prior, atol=1e-14, rtol=0)
    assert certify_convex_quad_output(output) > 0.05
    grads = torch.autograd.grad((output * torch.randn_like(output)).sum(), seed + levels)
    assert all(torch.isfinite(g).all() and g.abs().amax() > 0 for g in grads)
