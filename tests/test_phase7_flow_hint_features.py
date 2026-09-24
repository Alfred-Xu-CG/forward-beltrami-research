"""The local ridge-flow feature is a differentiable image cue, not a map."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.forward_p1_encoder import (
    ForwardP1ImageEncoder, ridge_local_flow_features,
)


def test_local_flow_feature_direction_and_image_gradient() -> None:
    axis = torch.linspace(0, 1, 65)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    moving = (
        torch.sin(2 * math.pi * xx)
        + torch.cos(2 * math.pi * yy)
        + .3 * torch.sin(2 * math.pi * (xx + yy))
    )[None, None].requires_grad_()
    displacement = (0.003, -0.004)
    query = torch.stack((
        xx + displacement[0], yy + displacement[1],
    ), dim=-1)[None]
    fixed = F.grid_sample(
        moving, 2 * query - 1,
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    features = ridge_local_flow_features(
        fixed, moving, window=9, ridge=.1,
    )
    assert features.shape == (1, 2, 65, 65)
    assert bool(torch.isfinite(features).all())
    center = features[:, :, 8:-8, 8:-8].mean(dim=(0, 2, 3))
    assert float(center[0]) > 0
    assert float(center[1]) < 0
    grad = torch.autograd.grad(features.square().mean(), moving)[0]
    assert bool(torch.isfinite(grad).all())
    assert float(grad.abs().amax()) > 0


def test_image_encoder_accepts_flow_feature_without_changing_output_contract() -> None:
    model = ForwardP1ImageEncoder(
        17, (33, 65), seed_passes=1,
        feature_side=65, width=4, flow_hint=True,
    )
    fixed = torch.rand(2, 1, 65, 65)
    moving = torch.rand(2, 1, 65, 65)
    seed, levels = model(fixed, moving)
    assert seed[0].shape == (2, 15, 15, 2)
    assert levels[0].shape == (2, 31, 31, 2)
    assert levels[1].shape == (2, 63, 63, 2)
    assert model.stem[0].in_channels == 6
