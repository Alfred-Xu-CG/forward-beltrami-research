"""Tiny structural-image CNN plumbing, safety and VJP fixture."""

import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_mind_amortized_network import (
    MindSafeImageNetwork, image_features, structural_loss,
)
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_affine_prewarp import warp_moving_to_fixed


def test_translated_image_cnn_has_safe_finite_vjp():
    torch.manual_seed(5)
    fixed = torch.rand((1, 1, 128, 128))
    moving = torch.roll(fixed, 2, dims=-1)
    features, fixed_descriptor, moving_descriptor, mask = image_features(fixed, moving)
    assert features.shape == (1, 43, 128, 128)
    network = MindSafeImageNetwork(width=8)
    mapped = network(features)
    assert mapped.shape == (1, 65, 65, 2)
    assert validate_q1_map(mapped, identity_vertices(65, device=fixed.device))["valid"]
    loss = structural_loss(fixed_descriptor, moving_descriptor, mask, mapped)
    loss = loss + strain_penalty(mapped)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in network.parameters())
    assert sum(float(p.grad.abs().sum()) for p in network.parameters()) > 0


def test_image_only_predictor_import_and_pure_torch_affine_warp():
    from tools.digital_mind_amortized_predict import predict

    assert callable(predict)
    moving = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    warped = warp_moving_to_fixed(moving, torch.eye(2), torch.zeros(2),
                                  height=8, width=8)
    torch.testing.assert_close(warped, moving, atol=1e-5, rtol=0)
