import numpy as np
import torch

from qcopt.forward.tutte_torch import tutte_embedding_torch
from qcopt.forward.tutte import tutte_weights
from qcopt.mesh import structured_rectangle


def test_tutte_boundary_layer_has_a_finite_autograd_gradient():
    mesh = structured_rectangle(6, 5)
    loop = mesh.boundary_loops[0]
    target = torch.tensor(mesh.vertices[loop], dtype=torch.double, requires_grad=True)

    output = tutte_embedding_torch(mesh, target)
    loss = (output.square()).mean()
    loss.backward()

    assert target.grad is not None
    assert target.grad.shape == target.shape
    assert torch.isfinite(target.grad).all()


def test_tutte_boundary_layer_matches_explicit_weight_jacobian():
    mesh = structured_rectangle(5, 4)
    loop = mesh.boundary_loops[0]
    target = torch.tensor(1.2 * mesh.vertices[loop], dtype=torch.double, requires_grad=True)

    output = tutte_embedding_torch(mesh, target)
    upstream = torch.arange(output.numel(), dtype=torch.double).reshape_as(output)
    output.backward(upstream)
    expected = torch.tensor(tutte_weights(mesh).T @ upstream.numpy(), dtype=torch.double)
    assert torch.allclose(target.grad, expected, atol=1e-12, rtol=1e-12)
