"""Geometry and gradient checks for the small-kernel safe-P1 decoder."""

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_kernel_match_safe257 import kernel_target, steer
from tools.digital_mind_sparse_match_finetune import p1_at_points
from tools.digital_q1_dhr_distill import identity_vertices


def test_kernel_decoder_dense_topology_and_three_input_vjp():
    baseline = identity_vertices(257, device=torch.device("cpu")).double()
    baseline.requires_grad_(True)
    rng = np.random.default_rng(41)
    source = torch.tensor(rng.uniform(.1, .9, (16, 2)),
                          dtype=torch.float64, requires_grad=True)
    target = (source.detach() + torch.tensor([.004, -.003], dtype=torch.float64))
    target.requires_grad_(True)
    goal = kernel_target(baseline, source, target)
    mapped = steer(baseline, goal, passes=4)
    assert validate_q1_map(
        mapped.detach(), identity_vertices(257, device=torch.device("cpu")).double()
    )["valid"]
    loss = (p1_at_points(mapped, source) - target).square().mean()
    gradients = torch.autograd.grad(loss, (baseline, source, target))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert mapped.shape == (1, 257, 257, 2)
    np.testing.assert_allclose(mapped.detach()[0, 0, :, :].numpy(),
                               baseline.detach()[0, 0, :, :].numpy(), atol=0)


def test_kernel_target_ridge_reduces_exact_match_residual():
    baseline = identity_vertices(257, device=torch.device("cpu")).double()
    source = torch.tensor([[.1 + .05 * i, .24 + .006 * i]
                           for i in range(16)], dtype=torch.float64)
    target = source + torch.tensor([.002, .001], dtype=torch.float64)
    desired = kernel_target(baseline, source, target, sigma=.1, ridge=.1)
    initial = (p1_at_points(baseline, source) - target).norm(dim=-1).mean()
    fitted = (p1_at_points(desired, source) - target).norm(dim=-1).mean()
    assert fitted < initial
