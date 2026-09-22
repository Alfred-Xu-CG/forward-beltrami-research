"""Full-system and derivative checks for exact patch-interior elimination."""

from __future__ import annotations

import numpy as np
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ExactBlockSchurTutteLayer


def test_block_schur_matches_full_direct_for_random_all_edge_weights() -> None:
    mesh = structured_rectangle(16, 16)
    layer = ExactBlockSchurTutteLayer(mesh, 4)
    rng = np.random.default_rng(20260923)
    logits = rng.normal(size=layer.n_conductances)
    mapped = layer(torch.tensor(logits, dtype=torch.float64)).detach().numpy()
    direct, direct_residual = layer.independent_full_solve(logits)
    assert direct_residual < 1.0e-13
    assert layer.last_forward_stats[0].relative_residual < 1.0e-13
    assert np.max(np.abs(mapped - direct)) < 1.0e-13
    assert layer.last_forward_stats[0].minimum_signed_area_ratio > 0.0


def test_block_schur_implicit_vjp_matches_central_difference() -> None:
    mesh = structured_rectangle(8, 8)
    layer = ExactBlockSchurTutteLayer(mesh, 2)
    rng = np.random.default_rng(20260923)
    latent = torch.tensor(rng.normal(size=layer.n_conductances), dtype=torch.float64, requires_grad=True)
    direction = torch.tensor(rng.normal(size=layer.n_conductances), dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)
    cotangent = torch.tensor(rng.normal(size=(mesh.n_vertices, 2)), dtype=torch.float64)
    objective = (layer(latent) * cotangent).sum()
    gradient = torch.autograd.grad(objective, latent)[0]
    step = 1.0e-5
    plus = (layer((latent + step * direction).detach()) * cotangent).sum().item()
    minus = (layer((latent - step * direction).detach()) * cotangent).sum().item()
    finite_difference = (plus - minus) / (2.0 * step)
    assert abs(gradient.dot(direction).item() - finite_difference) < 1.0e-8


def test_block_schur_batch_outputs_and_gradients() -> None:
    mesh = structured_rectangle(8, 8)
    layer = ExactBlockSchurTutteLayer(mesh, 2)
    logits = torch.randn((2, layer.n_conductances), dtype=torch.float64, requires_grad=True)
    mapped = layer(logits)
    assert mapped.shape == (2, mesh.n_vertices, 2)
    mapped.square().mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
