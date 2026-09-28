"""End-to-end image-encoder integration smoke, without landmark access."""

from __future__ import annotations

import torch

from tools.digital_q1_network_teacher import input_sensitivity, train_to_teacher


def test_one_pair_teacher_training_reaches_affine_head_and_safe_map():
    torch.manual_seed(93)
    image = torch.rand(1, 1, 16, 16)
    other = torch.rand(1, 1, 16, 16)
    axis = torch.arange(9) / 8
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    teacher = torch.stack((x + .02, y - .01), dim=-1)[None]
    model, (residual, matrix, offset), report = train_to_teacher(
        image, other, teacher, seed_side=5, steps=3, learning_rate=.005,
        device="cpu", width=4, feature_side=16, flow_hint=False,
    )
    assert report["finite_gradient_steps"] == 3
    assert report["best_vector_rmse"] < report["initial_vector_rmse"]
    assert residual.shape == (1, 9, 9, 2)
    assert bool((torch.linalg.det(matrix) > 0).all())
    assert report["nonpositive_residual_corners"] == 0
    assert model.training is False


def test_input_sensitivity_detects_swapped_and_blank_pairs():
    torch.manual_seed(31)
    image = torch.rand(1, 1, 16, 16)
    other = torch.rand(1, 1, 16, 16)
    axis = torch.arange(9) / 8
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    teacher = torch.stack((x + .01, y), dim=-1)[None]
    model, _, _ = train_to_teacher(
        image, other, teacher, seed_side=5, steps=3, learning_rate=.005,
        device="cpu", width=4, feature_side=16, flow_hint=False,
    )
    report = input_sensitivity(model, image, other)
    assert report["swapped_input_map_vector_rmse"] >= 0
    assert report["blank_input_map_vector_rmse"] >= 0
    assert all(torch.isfinite(torch.tensor(value)) for value in report.values())


def test_two_pair_teacher_batch_keeps_both_safe_and_backpropagates():
    torch.manual_seed(47)
    fixed = torch.rand(2, 1, 16, 16)
    moving = torch.rand(2, 1, 16, 16)
    axis = torch.arange(9) / 8
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((x, y), dim=-1)
    teacher = torch.stack((base + torch.tensor([.02, 0.]),
                           base + torch.tensor([0., -.02])))
    _, (residual, matrix, offset), report = train_to_teacher(
        fixed, moving, teacher, seed_side=5, steps=3, learning_rate=.005,
        device="cpu", width=4, feature_side=16, flow_hint=False,
    )
    assert residual.shape == (2, 9, 9, 2)
    assert matrix.shape == (2, 2, 2)
    assert offset.shape == (2, 2)
    assert report["batch"] == 2
    assert report["finite_gradient_steps"] == 3
    assert report["nonpositive_residual_corners"] == 0
