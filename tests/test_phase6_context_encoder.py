"""Shape, topology and gradient smoke for the optional contextual A2 encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from phase6_train_multisample_image import ConvexQuadImageEncoder  # noqa: E402
from qcopt.neural_bijection.dense import HierarchicalConvexQuadFreeCenterLayer, certify_convex_quad_output


def test_contextual_encoder_supplies_fine_grid_homeomorphism_and_gradient() -> None:
    torch.manual_seed(20260923)
    side = 17
    encoder = ConvexQuadImageEncoder(side, body_mode="context")
    decoder = HierarchicalConvexQuadFreeCenterLayer(side)
    pair = torch.randn(2, 2, 32, 32)
    control = decoder(*encoder(pair))
    assert control.shape == (2, side, side, 2)
    assert certify_convex_quad_output(control) > 0
    weight = torch.randn_like(control)
    (control * weight).sum().backward()
    head_gradient = encoder.center_heads[-1].weight.grad
    assert head_gradient is not None and torch.isfinite(head_gradient).all()
    assert head_gradient.abs().sum() > 0
