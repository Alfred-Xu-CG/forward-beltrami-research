"""Frozen-feature optimizer mechanics without network download."""

import torch
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_dinov2_safe_optimize import patch_loss
from tools.digital_mind_safe_optimize import SafeThreeLevelF1
from tools.digital_q1_dhr_distill import identity_vertices


def test_shifted_patch_field_has_finite_safe_latent_vjp():
    torch.manual_seed(18)
    fixed = F.normalize(torch.rand((1, 8, 32, 32)), dim=1)
    moving = torch.roll(fixed, 1, dims=-1)
    mask = torch.ones((1, 1, 32, 32))
    model = SafeThreeLevelF1()
    mapped = model()
    assert validate_q1_map(mapped, identity_vertices(65, device=mapped.device))["valid"]
    loss = patch_loss(fixed, moving, mask, mapped)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.logits)
    assert sum(float(p.grad.abs().sum()) for p in model.logits) > 0
