"""Independent algebra/gradient checks for the fine-grid edge Woodbury layer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import SparseEdgeWoodburyTutteLayer
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


def _selected_indices(side: int) -> tuple[int, int, int]:
    reference = MatrixFreeSymmetricTutteLayer(structured_rectangle(side - 1, side - 1))
    ii = reference._ii_edges.detach().cpu().numpy()
    ib = reference._ib_edges.detach().cpu().numpy()
    return int(ii[0]), int(ii[len(ii) // 2]), int(ib[len(ib) // 2])


def test_woodbury_matches_independent_full_solve_including_boundary_edge() -> None:
    side = 9
    mesh = structured_rectangle(side - 1, side - 1)
    layer = SparseEdgeWoodburyTutteLayer(mesh, _selected_indices(side)).to(dtype=torch.float64)
    latent = torch.tensor([-1.2, 0.5, 1.7], dtype=torch.float64)
    map_woodbury = layer(latent).detach().numpy()
    increments = (layer.minimum_increment + F.softplus(latent)).detach().numpy()
    map_sparse, residual = layer.independent_sparse_solve(increments)
    assert residual < 1.0e-13
    assert np.max(np.abs(map_woodbury - map_sparse)) < 2.0e-13
    assert np.max(np.abs(map_woodbury[mesh.boundary_loops[0]] - mesh.vertices[mesh.boundary_loops[0]])) == 0.0


def test_woodbury_latent_vjp_matches_directional_difference() -> None:
    side = 9
    layer = SparseEdgeWoodburyTutteLayer(structured_rectangle(side - 1, side - 1), _selected_indices(side)).to(dtype=torch.float64)
    latent = torch.tensor([-0.7, 0.2, 0.9], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.3, -0.8, 0.5], dtype=torch.float64)
    cotangent = torch.randn((side**2, 2), dtype=torch.float64)
    scalar = (layer(latent) * cotangent).sum()
    gradient = torch.autograd.grad(scalar, latent)[0]
    step = 1.0e-5
    plus = (layer((latent + step * direction).detach()) * cotangent).sum().item()
    minus = (layer((latent - step * direction).detach()) * cotangent).sum().item()
    finite_difference = (plus - minus) / (2 * step)
    assert abs(gradient.dot(direction).item() - finite_difference) < 1.0e-8


def test_woodbury_dense_output_has_positive_faces() -> None:
    side = 17
    selected = _selected_indices(side)
    layer = SparseEdgeWoodburyTutteLayer(structured_rectangle(side - 1, side - 1), selected).to(dtype=torch.float32)
    latent = torch.randn((2, len(selected)), dtype=torch.float32, requires_grad=True)
    mapped = layer(latent)
    assert mapped.shape == (2, side**2, 2)
    mapped.square().sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
