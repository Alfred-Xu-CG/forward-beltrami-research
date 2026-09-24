"""Image-derived residual steps remain differentiable and P1-safe."""
from __future__ import annotations

import math

import torch

from qcopt.neural_bijection.dense import (
    SafeColoredVertexRelaxation, certify_p1_or_identity,
    local_photometric_logits,
)


def test_local_residual_hint_safe_update_has_image_vjp() -> None:
    side = 33
    axis = torch.arange(side, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    moving = (
        torch.sin(2 * math.pi * xx)
        + .7 * torch.cos(2 * math.pi * yy)
        + .2 * torch.sin(2 * math.pi * (xx + yy))
    )[None, None].float().requires_grad_()
    fixed = (moving.detach() + .002 * torch.sin(4 * math.pi * xx)[None, None].float())
    hint = local_photometric_logits(
        fixed, moving, identity.float(),
        window=5, ridge=1., raw_span=2.,
    ).double()
    safe = SafeColoredVertexRelaxation(side, motion_mode="radial", raw_span=2.)
    floor = identity.new_full((1,), .05 / (side - 1) ** 2)
    updated = safe(identity, hint, area_floor=floor)
    certified, accepted = certify_p1_or_identity(updated, identity)
    assert bool(accepted.item())
    sw, se = certified[:, :-1, :-1], certified[:, :-1, 1:]
    ne, nw = certified[:, 1:, 1:], certified[:, 1:, :-1]
    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    assert float(torch.minimum(
        cross(se - sw, ne - sw).amin(), cross(ne - sw, nw - sw).amin(),
    )) > 0
    weighted = (certified[..., 0] * xx[None]).mean()
    image_gradient = torch.autograd.grad(weighted, moving)[0]
    assert bool(torch.isfinite(image_gradient).all())
    assert float(image_gradient.abs().amax()) > 0


def test_float32_repeated_local_updates_certify_and_backpropagate() -> None:
    torch.manual_seed(20260924)
    side = 65
    axis = torch.arange(side, dtype=torch.float32) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None].expand(2, -1, -1, -1)
    safe = SafeColoredVertexRelaxation(
        side, motion_mode="radial", raw_span=2.,
    )
    floor = identity.new_full((2,), .05 / (side - 1) ** 2)
    proposals = [
        (3 * torch.randn(2, side - 2, side - 2, 2)).requires_grad_()
        for _ in range(3)
    ]
    current = identity
    for proposal in proposals:
        current = safe(current, proposal, area_floor=floor)
    certified, accepted = certify_p1_or_identity(current, identity)
    assert bool(accepted.all())
    sw, se = certified[:, :-1, :-1].double(), certified[:, :-1, 1:].double()
    ne, nw = certified[:, 1:, 1:].double(), certified[:, 1:, :-1].double()
    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    assert float(torch.minimum(
        cross(se - sw, ne - sw).amin(), cross(ne - sw, nw - sw).amin(),
    )) > 0
    loss = (certified[..., 0] * xx[None]).mean()
    gradients = torch.autograd.grad(loss, proposals)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert any(float(gradient.abs().amax()) > 0 for gradient in gradients)


def test_multichannel_photometric_hint_preserves_duplicate_channel_result() -> None:
    side = 17
    axis = torch.linspace(0, 1, side)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    moving = (torch.sin(5 * xx + 3 * yy) + .3 * torch.cos(2 * xx - yy))[None, None]
    fixed = moving + .01 * (xx - yy)[None, None]
    single = local_photometric_logits(
        fixed, moving, identity, window=3, ridge=.7, raw_span=2.,
    )
    duplicate = local_photometric_logits(
        fixed.repeat(1, 2, 1, 1), moving.repeat(1, 2, 1, 1),
        identity, window=3, ridge=.7, raw_span=2.,
    )
    torch.testing.assert_close(single, duplicate, atol=1e-6, rtol=1e-6)
    moving_pair = torch.cat((moving, torch.cos(7 * xx - 2 * yy)[None, None]), dim=1)
    moving_pair.requires_grad_()
    fixed_pair = torch.cat((fixed, fixed + .02 * yy[None, None]), dim=1)
    logits = local_photometric_logits(
        fixed_pair, moving_pair, identity, window=3, ridge=.7, raw_span=2.,
    )
    gradient = torch.autograd.grad(logits.square().mean(), moving_pair)[0]
    assert bool(torch.isfinite(gradient).all())
    assert bool((gradient.abs().sum(dim=(0, 2, 3)) > 0).all())
