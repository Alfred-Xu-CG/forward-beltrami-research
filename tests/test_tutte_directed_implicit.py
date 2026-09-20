import numpy as np
import torch

from qcopt.forward.tutte_directed_implicit import (
    DirectedTutteSystem,
    directed_tutte_embedding_torch_implicit,
)
from qcopt.mesh import structured_rectangle


def test_directed_implicit_boundary_and_logit_gradients_match_finite_difference():
    torch.set_default_dtype(torch.float64)
    mesh = structured_rectangle(8, 8)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.as_tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64).requires_grad_(True)
    logits = torch.zeros(system.n_rows, system.max_degree, dtype=torch.float64, requires_grad=True)
    output = directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)
    loss = (output * output).sum()
    loss.backward()
    assert torch.isfinite(boundary.grad).all()
    assert torch.isfinite(logits.grad).all()
    row = system.n_rows // 2
    col = int(np.flatnonzero(system.valid_mask[row])[0])
    eps = 1e-5
    with torch.no_grad():
        plus = logits.detach().clone()
        minus = logits.detach().clone()
        plus[row, col] += eps
        minus[row, col] -= eps
        lp = (directed_tutte_embedding_torch_implicit(mesh, boundary.detach(), plus, system) ** 2).sum()
        lm = (directed_tutte_embedding_torch_implicit(mesh, boundary.detach(), minus, system) ** 2).sum()
    finite_difference = (lp - lm) / (2.0 * eps)
    assert torch.allclose(logits.grad[row, col], finite_difference, rtol=2e-4, atol=2e-6)
