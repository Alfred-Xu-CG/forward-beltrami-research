"""Image-only O-mode optimisation uses the same safe Q1 decoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers
from tools.digital_q1_real_optimize import apply_post_affine, optimize_tensors


def test_image_only_latent_optimisation_reduces_appearance_loss_without_labels() -> None:
    torch.manual_seed(41321)
    q = fixed_pixel_centers(32, 32, dtype=torch.float32, device=torch.device("cpu"))
    x, y = q[..., 0], q[..., 1]
    moving = (torch.sin(8 * torch.pi * x + 3 * torch.pi * y)
              + .7 * torch.cos(4 * torch.pi * y - x))[:, None]
    interior = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    truth = torch.stack((x + .015 * interior, y - .01 * interior), dim=-1)
    fixed = F.grid_sample(
        moving, 2 * truth - 1, mode="bilinear",
        padding_mode="border", align_corners=False,
    ).detach()
    mapped, result = optimize_tensors(
        fixed, moving, final_side=17, steps=8,
        learning_rate=.04, device="cpu",
    )
    assert mapped.shape == (1, 17, 17, 2)
    assert result["best_image_loss"] < result["initial_image_loss"]
    assert result["finite_gradient_steps"] == 8
    assert result["nonpositive_corners"] == 0
    assert result["boundary_ordered_rectangle"]
    assert math.isfinite(result["step_seconds_median"])


def test_post_affine_acts_on_output_of_safe_residual_and_keeps_vjp() -> None:
    q = fixed_pixel_centers(21, 19, dtype=torch.float32, device=torch.device("cpu"))
    residual = q.clone().requires_grad_(True)
    matrix = torch.tensor([[1.02, .13], [-.04, .98]], dtype=torch.float32)
    offset = torch.tensor([.025, -.035], dtype=torch.float32)
    composed = apply_post_affine(residual, matrix, offset)
    torch.testing.assert_close(composed, q @ matrix.T + offset)
    assert float(torch.linalg.det(matrix)) > 0
    composed.square().mean().backward()
    assert residual.grad is not None and bool(torch.isfinite(residual.grad).all())


def test_affine_initialized_optimization_preserves_factorized_residual() -> None:
    q = fixed_pixel_centers(32, 32, dtype=torch.float32, device=torch.device("cpu"))
    x, y = q[..., 0], q[..., 1]
    moving = (torch.sin(7 * torch.pi * x) + torch.cos(5 * torch.pi * y))[:, None]
    matrix = torch.tensor([[1.0, .08], [0.0, 1.0]], dtype=torch.float32)
    offset = torch.tensor([.02, -.015], dtype=torch.float32)
    fixed = F.grid_sample(moving, 2 * (q @ matrix.T + offset) - 1,
                          mode="bilinear", padding_mode="border",
                          align_corners=False).detach()
    residual, report = optimize_tensors(
        fixed, moving, final_side=17, steps=2, learning_rate=.02,
        device="cpu", post_affine_matrix=matrix, post_affine_offset=offset,
    )
    assert residual.shape == (1, 17, 17, 2)
    assert report["post_affine_det"] > 0
    assert report["finite_gradient_steps"] == 2
    assert report["nonpositive_corners"] == 0
