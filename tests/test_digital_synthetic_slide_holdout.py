"""Small protocol and coordinate checks for the slide-texture capability probe."""

from __future__ import annotations

import pytest
import numpy as np
import torch

from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from tools import digital_q1_synthetic_slide_holdout as holdout
from tools.digital_q1_network_teacher import _save_checkpoint
from tools.digital_q1_synthetic_slide_holdout import (
    analytic_map, make_coefficients, point_grid, run, synthesize,
)
from tools.digital_q1_synthetic_slide_baseline import optimize_baseline
from tools.digital_q1_synthetic_p1_eval import evaluate


def test_identity_sampled_at_pixel_centers() -> None:
    device = torch.device("cpu")
    centers = point_grid(8, centers=True, device=device)
    nodes = point_grid(17, centers=False, device=device)
    assert torch.equal(centers[0, 0, 0], torch.tensor((1 / 16, 1 / 16)))
    assert torch.equal(nodes[0, 0, 0], torch.tensor((0., 0.)))
    image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
    fixed = synthesize(image, torch.zeros((1, 6)), centers)
    assert torch.allclose(fixed, image, atol=2e-7, rtol=0)


def test_analytic_map_has_exact_fixed_boundary() -> None:
    grid = point_grid(33, centers=False, device=torch.device("cpu"))
    coefficients = torch.tensor(((.018, .008, -.008, -.018, -.008, .008),))
    mapped = analytic_map(grid, coefficients)
    assert torch.equal(mapped[:, 0, :, :], grid[:, 0, :, :])
    assert torch.equal(mapped[:, -1, :, :], grid[:, -1, :, :])
    assert torch.equal(mapped[:, :, 0, :], grid[:, :, 0, :])
    assert torch.equal(mapped[:, :, -1, :], grid[:, :, -1, :])


def test_negative_texture_alias_cannot_bypass_holdout() -> None:
    textures = torch.rand((3, 1, 32, 32))
    with pytest.raises(ValueError, match="canonical nonnegative"):
        run(textures=textures, train_texture_indices=[2],
            test_texture_indices=[-1], side=33, steps=1, batch=1,
            train_count=1, test_count=1, device="cpu")
    with pytest.raises(ValueError, match="canonical nonnegative"):
        optimize_baseline(textures=textures, test_texture_indices=[-1],
                          side=33, steps=1, batch=1, test_count=1,
                          device="cpu", learning_rate=.01)


def test_p1_diagonal_branches_have_same_value_and_vertex_vjp() -> None:
    corners = torch.tensor(((0., 0.), (1., 0.), (1.1, 1.), (0., 1.)),
                           dtype=torch.float64, requires_grad=True)
    a, b, c, d = corners.unbind(0)
    s = t = .37
    lower = a + s * (b - a) + t * (c - b)
    upper = a + s * (c - d) + t * (d - a)
    assert torch.allclose(lower, upper, atol=1e-15, rtol=0)
    lower_vjp, = torch.autograd.grad(lower.sum(), corners, retain_graph=True)
    upper_vjp, = torch.autograd.grad(upper.sum(), corners)
    assert torch.allclose(lower_vjp, upper_vjp, atol=1e-15, rtol=0)


def test_p1_image_warp_identity_and_vertex_gradient() -> None:
    moving = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
    vertices = point_grid(3, centers=False, device=torch.device("cpu")).requires_grad_()
    warped = holdout.warp_moving_at_p1_map(moving, vertices)
    assert torch.allclose(warped, moving, atol=2e-7, rtol=0)
    gradient, = torch.autograd.grad(warped[:, :, 2:6, 2:6].sum(), vertices)
    assert bool(torch.isfinite(gradient).all())
    assert bool((gradient != 0).any())


def test_p1_image_loss_trains_encoder_on_small_batch() -> None:
    textures = torch.rand((3, 1, 32, 32))
    report = run(textures=textures, train_texture_indices=[0, 1],
                 test_texture_indices=[2], side=33, steps=2, batch=1,
                 train_count=3, test_count=2, device="cpu",
                 loss_mode="p1_image")
    assert report["training_loss_mode"] == "p1_image"
    assert report["model_seed"] == 291001
    assert report["minibatch_seed"] == 291004
    assert report["train_coefficient_scale"] == 1.0
    assert report["test_coefficient_scale"] == 1.0
    assert report["test_image_mse_mean"] >= 0
    assert report["test_map_rmse_mean"] >= 0


def test_p1_report_names_twice_triangle_area_as_determinant(tmp_path) -> None:
    weights = tmp_path / "tiny_model.npz"
    model = Q1ImageRegistrationNetwork(seed_side=17, final_side=33,
                                       width=16, feature_side=33)
    _save_checkpoint(weights, model)
    report = evaluate(textures=torch.rand((2, 1, 32, 32)),
                      test_texture_indices=[1], weights=weights,
                      side=33, test_count=1, batch=1, device="cpu")
    assert report["minimum_unnormalized_p1_triangle_determinant_float32"] > 0
    assert report["full_p1_inference_seconds_per_image_median"] >= 0


def test_p1_ood_scale_requires_positive_bounded_deformation(tmp_path) -> None:
    with pytest.raises(ValueError, match="coefficient scale"):
        evaluate(textures=torch.rand((2, 1, 32, 32)),
                 test_texture_indices=[1], weights=tmp_path / "unused.npz",
                 side=33, test_count=1, batch=1, device="cpu",
                 coefficient_scale=0)


def test_no_network_baseline_can_use_matching_p1_sampling_and_scale(tmp_path) -> None:
    report = optimize_baseline(
        textures=torch.rand((2, 1, 32, 32)), test_texture_indices=[1],
        side=33, steps=1, batch=1, test_count=1, device="cpu",
        learning_rate=.01, regularization=.1, coefficient_scale=2.0,
        image_interpolation="p1", output_example=tmp_path / "example.npz",
    )
    assert report["coefficient_scale"] == 2.0
    assert report["image_interpolation"] == "fixed_SW_NE_P1"
    assert report["saved_example_certificate"]["valid"]
    assert report["mean_map_rmse"] >= 0
    assert report["mean_query_map_rmse"] >= 0
    with np.load(tmp_path / "example.npz") as archive:
        vertices = archive["vertices"][0]
    line = (np.arange(32) + .5) / 32
    qx, qy = np.meshgrid(line, line)
    sx, sy = qx * 32, qy * 32
    col, row = np.floor(sx).astype(int), np.floor(sy).astype(int)
    tx, ty = sx - col, sy - row
    a, b = vertices[row, col], vertices[row, col + 1]
    c, d = vertices[row + 1, col + 1], vertices[row + 1, col]
    lower = a + tx[..., None] * (b - a) + ty[..., None] * (c - b)
    upper = a + tx[..., None] * (c - d) + ty[..., None] * (d - a)
    mapped = np.where((tx >= ty)[..., None], lower, upper)
    truth = analytic_map(
        point_grid(32, centers=True, device=torch.device("cpu")),
        2 * make_coefficients(1, 291003),
    )[0].numpy()
    expected = np.sqrt(np.mean(np.sum((mapped - truth) ** 2, axis=-1)))
    assert abs(expected - report["mean_query_map_rmse"]) < 1e-6


def test_no_network_baseline_rejects_out_of_family_scale() -> None:
    with pytest.raises(ValueError, match="coefficient scale"):
        optimize_baseline(
            textures=torch.rand((2, 1, 32, 32)), test_texture_indices=[1],
            side=33, steps=1, batch=1, test_count=1, device="cpu",
            learning_rate=.01, coefficient_scale=0,
        )
