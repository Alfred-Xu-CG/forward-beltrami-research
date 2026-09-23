"""Optional exact SPD backend tests; skipped where PyPardiso is unavailable."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as sparse_linalg
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import PardisoAllEdgeTutteLayer


pypardiso = pytest.importorskip("pypardiso")


@pytest.mark.parametrize("side", [17, 33])
def test_pardiso_full_system_and_implicit_vjp(side: int) -> None:
    mesh = structured_rectangle(side - 1, side - 1)
    layer = PardisoAllEdgeTutteLayer(mesh, patch_cells=4)
    generator = np.random.default_rng(20260923 + side)
    initial = -0.8 + 0.2 * generator.standard_normal(layer.n_conductances)
    logits = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    mapped = layer(logits)
    conductance = layer.minimum_conductance + np.logaddexp(0.0, initial)
    matrix, rhs = layer._assemble(conductance)
    separate_solution = sparse_linalg.spsolve(matrix.tocsc(), rhs)
    assert np.max(np.abs(mapped.detach().numpy()[layer._interior] - separate_solution)) < 2e-12
    assert np.max(np.abs(mapped.detach().numpy()[np.setdiff1d(np.arange(mesh.n_vertices), layer._interior)] - mesh.vertices[np.setdiff1d(np.arange(mesh.n_vertices), layer._interior)])) == 0
    cotangent = torch.tensor(generator.standard_normal(mapped.shape), dtype=torch.float64)
    objective = (mapped * cotangent).sum()
    objective.backward()
    assert np.all(np.isfinite(logits.grad.detach().numpy()))
    direction = generator.standard_normal(layer.n_conductances)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-4
    with torch.no_grad():
        plus = (layer(torch.tensor(initial + epsilon * direction, dtype=torch.float64)) * cotangent).sum().item()
        minus = (layer(torch.tensor(initial - epsilon * direction, dtype=torch.float64)) * cotangent).sum().item()
    finite_difference = (plus - minus) / (2 * epsilon)
    analytic = float(logits.grad.detach().numpy() @ direction)
    assert abs(finite_difference - analytic) < 2e-7
    assert layer.last_forward_stats[0]["minimum_signed_area_ratio"] > 0
