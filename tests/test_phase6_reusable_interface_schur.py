"""Independent full-solve and directional-gradient checks for interface-only C."""

from __future__ import annotations

import numpy as np
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ReusableInterfaceSchurTutteLayer


def test_reusable_interface_matches_new_full_sparse_factorization() -> None:
    layer = ReusableInterfaceSchurTutteLayer(structured_rectangle(16, 16), patch_cells=4)
    torch.manual_seed(817)
    logits = 0.7 * torch.randn(layer.n_variable_edges, dtype=torch.float64)
    mapped = layer(logits).detach().numpy()
    oracle, oracle_residual = layer.independent_full_solve(logits.numpy())
    assert layer.n_variable_edges > 0
    assert layer.n_variable_edges < layer.n_conductances
    assert oracle_residual < 1e-12
    assert np.max(np.abs(mapped - oracle)) < 2e-12
    assert layer.last_forward_stats[0].relative_residual < 1e-12
    assert layer.last_forward_stats[0].minimum_signed_area_ratio > 0


def test_reusable_interface_vjp_matches_central_difference() -> None:
    layer = ReusableInterfaceSchurTutteLayer(structured_rectangle(16, 16), patch_cells=4)
    torch.manual_seed(199)
    logits = (0.3 * torch.randn(layer.n_variable_edges, dtype=torch.float64)).requires_grad_()
    direction = torch.randn_like(logits)
    direction = direction / torch.linalg.vector_norm(direction)
    cotangent = torch.randn(17**2, 2, dtype=torch.float64)
    gradient = torch.autograd.grad((layer(logits) * cotangent).sum(), logits)[0]
    step = 1e-5
    plus = (layer((logits + step * direction).detach()) * cotangent).sum().item()
    minus = (layer((logits - step * direction).detach()) * cotangent).sum().item()
    observed = (plus - minus) / (2 * step)
    predicted = gradient.dot(direction).item()
    assert abs(predicted - observed) < 1e-7


def test_reusable_interface_batched_backward_is_finite() -> None:
    layer = ReusableInterfaceSchurTutteLayer(structured_rectangle(8, 8), patch_cells=2)
    logits = torch.zeros(2, layer.n_variable_edges, dtype=torch.float64, requires_grad=True)
    output = layer(logits)
    assert output.shape == (2, 9**2, 2)
    output.square().mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
