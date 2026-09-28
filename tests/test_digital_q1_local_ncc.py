"""DHR-style local image objective must be finite and differentiable."""

from __future__ import annotations

import pytest
import torch

from tools.digital_q1_real_optimize import _local_ncc_loss, optimize_tensors


def test_local_ncc_identical_random_image_and_gradient():
    torch.manual_seed(305)
    fixed = torch.rand(1, 1, 16, 16)
    warped = fixed.clone().requires_grad_()
    value = _local_ncc_loss(fixed, warped)
    assert torch.isfinite(value)
    assert value.item() < 1e-3
    value.backward()
    assert warped.grad is not None and bool(torch.isfinite(warped.grad).all())


def test_local_ncc_mode_runs_safe_decoder():
    torch.manual_seed(806)
    fixed = torch.rand(1, 1, 16, 16)
    moving = torch.rand(1, 1, 16, 16)
    mapped, report = optimize_tensors(
        fixed, moving, final_side=17, steps=2, learning_rate=.001,
        device="cpu", image_loss="local_ncc",
    )
    assert mapped.shape == (1, 17, 17, 2)
    assert report["image_loss"] == "local_ncc"
    assert report["finite_gradient_steps"] == 2
    assert report["nonpositive_corners"] == 0
    with pytest.raises(ValueError, match="image_loss"):
        optimize_tensors(fixed, moving, final_side=17, steps=1,
                         learning_rate=.001, device="cpu", image_loss="invalid")
