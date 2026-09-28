"""A mixed F2-D seed/F1-D refinement must retain one Q1 output map."""

from __future__ import annotations

import torch

from qcopt.neural_bijection.dense import HybridPatchSeedVertexQ1Pyramid
from qcopt.neural_bijection.dense.digital_q1 import (
    q1_corner_determinants,
    q1_dyadic_refine,
)
from qcopt.neural_bijection.dense.forward_p1_encoder import ForwardP1ImageEncoder


def _identity(side: int) -> torch.Tensor:
    axis = torch.arange(side, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None]


def test_zero_latents_refine_identity_and_preserve_boundary() -> None:
    layer = HybridPatchSeedVertexQ1Pyramid(5, 17, patch_cells=4)
    seed = tuple(torch.zeros((1, 3, 3, 2), dtype=torch.float64) for _ in range(1))
    levels = tuple(torch.zeros((1, side - 2, side - 2, 2), dtype=torch.float64)
                   for side in layer.level_sides)
    result = layer(seed, levels)
    torch.testing.assert_close(result, _identity(17), atol=0, rtol=0)
    assert torch.all(q1_corner_determinants(result) > 0)


def test_mixed_seed_and_new_vertex_levels_have_finite_true_vjp() -> None:
    layer = HybridPatchSeedVertexQ1Pyramid(9, 33, patch_cells=4)
    generator = torch.Generator().manual_seed(290931)
    seed = tuple((.1 * torch.randn((1, 7, 7, 2), generator=generator,
                                     dtype=torch.float64)).requires_grad_()
                 for _ in range(4))
    levels = tuple((.1 * torch.randn((1, side - 2, side - 2, 2),
                                       generator=generator, dtype=torch.float64)).requires_grad_()
                   for side in layer.level_sides)
    mapped = layer(seed, levels)
    assert mapped.shape == (1, 33, 33, 2)
    assert torch.all(q1_corner_determinants(mapped) > 0)
    identity = _identity(33)
    assert torch.equal(mapped[:, 0], identity[:, 0])
    assert torch.equal(mapped[:, -1], identity[:, -1])
    cotangent = torch.randn(mapped.shape, generator=generator, dtype=mapped.dtype)
    grads = torch.autograd.grad((mapped * cotangent).sum(), (*seed, *levels))
    assert all(torch.isfinite(grad).all() and grad.abs().amax() > 0 for grad in grads)


def test_image_encoder_latents_connect_to_q1_pyramid() -> None:
    encoder = ForwardP1ImageEncoder(5, (9, 17), seed_passes=1, width=4)
    layer = HybridPatchSeedVertexQ1Pyramid(5, 17, patch_cells=4)
    fixed = torch.rand((1, 1, 32, 32))
    moving = torch.rand((1, 1, 32, 32))
    seed, levels = encoder(fixed, moving)
    output = layer(seed, levels)
    assert output.shape == (1, 17, 17, 2)
    torch.testing.assert_close(output, _identity(17).float(), atol=0, rtol=0)
    target = _identity(17).float().clone()
    # (7,8) is new at the 17-grid level; the even/even (8,8) vertex is
    # intentionally frozen there and would give this head zero gradient.
    target[:, 7, 8, 0] += .02
    loss = (output - target).square().sum()
    loss.backward()
    assert encoder.level_heads[-1].weight.grad is not None
    assert torch.isfinite(encoder.level_heads[-1].weight.grad).all()
    assert encoder.level_heads[-1].weight.grad.abs().amax() > 0


def test_q1_refinement_keeps_a_non_affine_q1_cell() -> None:
    coarse = _identity(3)
    coarse[:, 1, 1] += torch.tensor([.03, -.02], dtype=coarse.dtype)
    refined = q1_dyadic_refine(coarse)
    layer = HybridPatchSeedVertexQ1Pyramid(3, 5, patch_cells=2)
    zero = torch.zeros((1, 3, 3, 2), dtype=coarse.dtype)
    result = layer.forward_from(coarse, (zero,), start_index=0)
    torch.testing.assert_close(result, refined, atol=0, rtol=0)
