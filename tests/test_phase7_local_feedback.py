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
