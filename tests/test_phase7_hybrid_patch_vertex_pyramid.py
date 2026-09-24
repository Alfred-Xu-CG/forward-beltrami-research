"""Joint F2 seed and F1 dyadic levels retain the fixed-grid P1 contract."""
from __future__ import annotations

import math

import torch

from qcopt.neural_bijection.dense import (
    HybridPatchSeedVertexP1Pyramid, exact_dyadic_p1_refine,
)


def _target(side: int, strength: float) -> torch.Tensor:
    axis = torch.arange(side, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((
        xx + strength * torch.sin(2 * math.pi * xx) * torch.sin(math.pi * yy),
        yy - .8 * strength * torch.sin(math.pi * xx) * torch.sin(2 * math.pi * yy),
    ), dim=-1)[None]


def _min_j(mapped: torch.Tensor) -> float:
    sw = mapped[:, :-1, :-1]
    se = mapped[:, :-1, 1:]
    ne = mapped[:, 1:, 1:]
    nw = mapped[:, 1:, :-1]
    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return float(torch.minimum(
        cross(se - sw, ne - sw).amin(),
        cross(ne - sw, nw - sw).amin(),
    ) * (mapped.shape[1] - 1) ** 2)


def test_hybrid_patch_seed_and_vertex_refinement_have_joint_vjp() -> None:
    layer = HybridPatchSeedVertexP1Pyramid(
        17, 65, patch_cells=4, seed_cycles=4,
    )
    identity = _target(17, 0.)
    seed_target = _target(17, .03)
    seed_span = .5 * 4 / 16
    seed = torch.atanh(
        ((seed_target - identity)[:, 1:-1, 1:-1] / seed_span)
    ).detach().requires_grad_()
    levels = []
    previous = seed_target
    for side in layer.level_sides:
        target = _target(side, .03)
        base = exact_dyadic_p1_refine(previous)
        raw = ((target - base)[:, 1:-1, 1:-1] / (2 / (side - 1)))
        levels.append(torch.atanh(raw).detach().requires_grad_())
        previous = target
    mapped = layer(seed, levels)
    assert mapped.shape == (1, 65, 65, 2)
    assert _min_j(mapped) > .05
    assert torch.equal(mapped[:, 0], layer.final_identity[:, 0])
    assert torch.equal(mapped[:, -1], layer.final_identity[:, -1])
    assert torch.equal(mapped[:, :, 0], layer.final_identity[:, :, 0])
    assert torch.equal(mapped[:, :, -1], layer.final_identity[:, :, -1])
    assert float((mapped - layer.final_identity).abs().amax()) > .01
    loss = ((mapped - layer.final_identity).square().sum(dim=-1)).mean()
    gradients = torch.autograd.grad(loss, (seed, *levels))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    assert all(float(g.abs().amax()) > 0 for g in gradients)


def test_hybrid_can_chain_two_coarse_endpoint_fields() -> None:
    layer = HybridPatchSeedVertexP1Pyramid(
        17, 33, patch_cells=4, seed_cycles=4, seed_steps=2,
    )
    start = _target(17, 0.)
    middle = _target(17, .06)
    end = _target(17, .12)
    span = .5 * 4 / 16
    first = torch.atanh(
        ((middle - start)[:, 1:-1, 1:-1] / span)
    ).detach().requires_grad_()
    second = torch.atanh(
        ((end - middle)[:, 1:-1, 1:-1] / span)
    ).detach().requires_grad_()
    fine_target = _target(33, .12)
    fine = torch.atanh(
        ((fine_target - exact_dyadic_p1_refine(end))[:, 1:-1, 1:-1]
         / (2 / 32))
    ).detach().requires_grad_()
    mapped = layer((first, second), (fine,))
    assert float((mapped - fine_target).abs().amax()) < 1e-12
    assert _min_j(mapped) > .05
    gradients = torch.autograd.grad(
        (mapped[..., 0] + .31 * mapped[..., 1]).mean(),
        (first, second, fine),
    )
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    assert all(float(g.abs().amax()) > 0 for g in gradients)


def test_hybrid_checkpointed_refinement_matches_plain_output_and_vjp() -> None:
    generator = torch.Generator().manual_seed(20260924)
    raw = (
        .1 * torch.randn((1, 15, 15, 2), generator=generator, dtype=torch.float64),
        .1 * torch.randn((1, 31, 31, 2), generator=generator, dtype=torch.float64),
        .1 * torch.randn((1, 63, 63, 2), generator=generator, dtype=torch.float64),
    )
    def run(checkpoint_levels: bool) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        layer = HybridPatchSeedVertexP1Pyramid(
            17, 65, patch_cells=4, seed_cycles=4,
            checkpoint_levels=checkpoint_levels,
        )
        fields = tuple(value.detach().clone().requires_grad_() for value in raw)
        mapped = layer(fields[0], fields[1:])
        objective = (mapped[..., 0].square() + .3 * mapped[..., 1].square()).mean()
        return mapped.detach(), torch.autograd.grad(objective, fields)
    plain_map, plain_grads = run(False)
    checked_map, checked_grads = run(True)
    torch.testing.assert_close(checked_map, plain_map, atol=0, rtol=0)
    for checked, plain in zip(checked_grads, plain_grads):
        torch.testing.assert_close(checked, plain, atol=1e-14, rtol=1e-12)
