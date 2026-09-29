"""Tiny objective-semantic tests, not anatomy validation."""

import torch

from tools.digital_mind_objective_probe import objective, self_similarity
from tools.digital_q1_dhr_distill import identity_vertices


def test_translated_self_similarity_prefers_correct_shift_and_has_vjp():
    torch.manual_seed(4)
    fixed = torch.rand((1, 1, 64, 64))
    moving = torch.roll(fixed, 2, dims=-1)
    identity = identity_vertices(17, device=fixed.device)
    correct = (identity + torch.tensor([2 / 64, 0])).requires_grad_(True)
    initial_loss, _ = objective(fixed, moving, identity)
    correct_loss, _ = objective(fixed, moving, correct)
    assert correct_loss < initial_loss / 3
    correct_loss.backward()
    assert correct.grad is not None and torch.isfinite(correct.grad).all()


def test_self_similarity_is_finite_on_constant_image():
    descriptor, scale = self_similarity(torch.zeros((1, 1, 16, 16)))
    assert descriptor.shape == (1, 8, 16, 16)
    assert scale.shape == (1, 1, 16, 16)
    assert torch.isfinite(descriptor).all() and torch.isfinite(scale).all()
