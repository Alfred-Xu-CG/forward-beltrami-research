import numpy as np
import torch

from qcopt.forward.tutte import tutte_embedding
from qcopt.forward.tutte_implicit import tutte_embedding_torch_implicit
from qcopt.mesh import structured_rectangle


def test_implicit_tutte_layer_matches_dense_forward_and_gradient():
    mesh = structured_rectangle(8, 7)
    loop = mesh.boundary_loops[0]
    boundary = torch.tensor(
        1.1 * mesh.vertices[loop] + np.asarray([0.2, -0.1]),
        dtype=torch.double,
        requires_grad=True,
    )

    implicit = tutte_embedding_torch_implicit(mesh, boundary)
    expected = torch.tensor(tutte_embedding(mesh, boundary.detach().numpy()), dtype=torch.double)
    assert torch.allclose(implicit.detach(), expected, atol=1e-11, rtol=1e-11)
    implicit.square().mean().backward()
    assert torch.isfinite(boundary.grad).all()


def test_implicit_tutte_layer_runs_on_a_high_resolution_mesh():
    mesh = structured_rectangle(128, 128)
    loop = mesh.boundary_loops[0]
    boundary = torch.tensor(mesh.vertices[loop], dtype=torch.double, requires_grad=True)

    output = tutte_embedding_torch_implicit(mesh, boundary)
    output.square().mean().backward()

    assert output.shape == (mesh.n_vertices, 2)
    assert torch.isfinite(boundary.grad).all()
