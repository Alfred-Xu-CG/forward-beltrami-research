import numpy as np
import pytest
import torch

from qcopt.autograd import lbs_from_raw, lsqc_fast_from_raw, lsqc_from_raw
from qcopt.constraints import fixed_vertex_constraints, two_pin_constraints
from qcopt.mesh import structured_rectangle


def test_lbs_custom_layer_passes_double_precision_gradcheck():
    mesh = structured_rectangle(2, 2)
    boundary = mesh.boundary_loops[0]
    constraints = fixed_vertex_constraints(
        mesh.n_vertices, boundary, mesh.vertices[boundary]
    )
    generator = torch.Generator().manual_seed(31)
    raw = (0.05 * torch.randn(mesh.n_faces, 2, generator=generator, dtype=torch.double)).requires_grad_()

    assert torch.autograd.gradcheck(
        lambda value: lbs_from_raw(value, mesh, constraints, k_max=0.8),
        (raw,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_lsqc_custom_layer_passes_double_precision_gradcheck(weighted):
    mesh = structured_rectangle(1, 1)
    constraints = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    generator = torch.Generator().manual_seed(41 if weighted else 37)
    raw = (0.04 * torch.randn(mesh.n_faces, 2, generator=generator, dtype=torch.double)).requires_grad_()

    assert torch.autograd.gradcheck(
        lambda value: lsqc_from_raw(
            value, mesh, constraints, k_max=0.8, weighted=weighted
        ),
        (raw,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


def test_layer_produces_one_gradient_vector_per_face():
    mesh = structured_rectangle(2, 2)
    constraints = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    raw = torch.zeros(mesh.n_faces, 2, dtype=torch.double, requires_grad=True)
    target = torch.as_tensor(mesh.vertices + np.array([0.1, -0.05]), dtype=torch.double)
    loss = (lsqc_from_raw(raw, mesh, constraints) - target).square().mean()
    loss.backward()

    assert raw.grad is not None
    assert raw.grad.shape == (mesh.n_faces, 2)
    assert torch.all(torch.isfinite(raw.grad))


def test_fast_weighted_lsqc_layer_passes_double_precision_gradcheck():
    mesh = structured_rectangle(1, 1)
    constraints = two_pin_constraints(
        mesh.n_vertices,
        [0, mesh.n_vertices - 1],
        mesh.vertices[[0, mesh.n_vertices - 1]],
    )
    generator = torch.Generator().manual_seed(131)
    raw = (
        0.04 * torch.randn(mesh.n_faces, 2, generator=generator, dtype=torch.double)
    ).requires_grad_()

    assert torch.autograd.gradcheck(
        lambda value: lsqc_fast_from_raw(value, mesh, constraints, k_max=0.8),
        (raw,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )
