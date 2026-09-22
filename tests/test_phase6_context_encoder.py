"""Shape, topology and gradient smoke for the optional contextual A2 encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from phase6_train_multisample_image import CoarseFineConvexQuadImageEncoder, ConvexQuadImageEncoder  # noqa: E402
from qcopt.neural_bijection.dense import CoarseFineConvexQuadComposition, HierarchicalConvexQuadFreeCenterLayer, certify_convex_quad_output


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


def test_image_encoder_vjp_through_exact_coarse_fine_composition() -> None:
    torch.manual_seed(217)
    encoder = CoarseFineConvexQuadImageEncoder(5, 17, head_mode="multilevel", body_mode="local")
    decoder = CoarseFineConvexQuadComposition(5, 17, 32)
    decoder.prepare(device="cpu", dtype=torch.float32)
    pair = torch.randn(2, 2, 32, 32)
    coarse, fine = encoder(pair)
    result = decoder(*coarse, *fine)
    assert result.dense.shape == (2, 32, 32, 2)
    assert all(certify_convex_quad_output(control) > 0 for control in result.controls)
    (result.dense * torch.randn_like(result.dense)).sum().backward()
    for head in (encoder.coarse.center_heads[0], encoder.fine.center_heads[-1]):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
