"""Synthetic image shift has a known fixed-to-moving map and live VJP."""

import torch

from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from tools.digital_q1_synthetic_translation import (
    evaluate, make_translated_moving, translation_target,
)


def test_shift_sign_and_exact_target():
    side = 33
    ramp = torch.linspace(0, 1, side)[None, None, None].expand(1, 1, side, side)
    shift = torch.tensor([[2 / (side - 1), 0.0]])
    moving = make_translated_moving(ramp, shift)
    assert torch.allclose(moving[0, 0, 16, 4:29], ramp[0, 0, 16, 2:27], atol=1e-6)
    target = translation_target(17, shift)
    assert target.shape == (1, 17, 17, 2)
    assert torch.allclose(target[0, 8, 8], torch.tensor([.5625, .5]))


def test_small_full_network_gradient_is_finite():
    torch.manual_seed(9)
    fixed = torch.rand((1, 1, 33, 33))
    shift = torch.tensor([[.025, -.03]])
    moving = make_translated_moving(fixed, shift)
    model = Q1ImageRegistrationNetwork(seed_side=17, final_side=33,
                                       feature_side=33, width=4)
    residual, matrix, offset = model(fixed, moving)
    target = translation_target(33, shift)
    loss = (model.apply_affine(residual, matrix, offset) - target).square().sum(-1).mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_batched_evaluation_checks_each_residual(tmp_path):
    texture = torch.rand((1, 1, 33, 33))
    model = Q1ImageRegistrationNetwork(seed_side=17, final_side=33,
                                       feature_side=33, width=4)
    result = evaluate(model, texture, samples=2, batch_size=2, seed=123,
                      shift_bound=.06, output_map=tmp_path / "sample.npz")
    assert result["heldout_examples"] == 2
    assert result["nonpositive_residual_corners"] == 0
    assert result["saved_example_certificate"]["saved_residual_valid"]
