"""Image-only O-mode optimisation uses the same safe Q1 decoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers
from tools.digital_q1_real_optimize import optimize_tensors


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
