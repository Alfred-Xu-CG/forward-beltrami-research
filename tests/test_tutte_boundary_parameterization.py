import torch
import pytest

from qcopt.forward.tutte_directed_implicit import (
    rectangle_boundary_from_logits,
    rectangle_boundary_from_modulus_logits,
)
from qcopt.forward.tutte_directed_implicit import DirectedTutteSystem, directed_tutte_embedding_torch_implicit
from qcopt.mesh import structured_rectangle


def test_rectangle_boundary_logits_are_strictly_ordered_closed_and_differentiable():
    torch.set_default_dtype(torch.float64)
    logits = torch.zeros((4, 5), requires_grad=True)
    boundary = rectangle_boundary_from_logits(logits, width=2.0, height=3.0)

    assert boundary.shape == (20, 2)
    assert torch.allclose(boundary[0], torch.tensor([0.0, 0.0]))
    # Four sides have strictly positive coordinate increments.
    assert torch.all(boundary[:5, 0][1:] > boundary[:5, 0][:-1])
    assert torch.all(boundary[5:10, 1][1:] > boundary[5:10, 1][:-1])
    assert torch.all(boundary[10:15, 0][1:] < boundary[10:15, 0][:-1])
    assert torch.all(boundary[15:20, 1][1:] < boundary[15:20, 1][:-1])
    assert torch.allclose(boundary[5], torch.tensor([2.0, 0.0]))
    assert torch.allclose(boundary[10], torch.tensor([2.0, 3.0]))
    assert torch.allclose(boundary[15], torch.tensor([0.0, 3.0]))
    # The final left-side segment closes back to the first point.
    assert torch.allclose(boundary[-1, 0], torch.tensor(0.0))
    assert boundary[-1, 1] > 0.0
    assert torch.isfinite(boundary).all()
    boundary.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_rectangle_boundary_rejects_wrong_logit_shape():
    with pytest.raises(ValueError):
        rectangle_boundary_from_logits(torch.zeros((3, 4)))


def test_parameterized_boundary_composes_with_directed_tutte_and_preserves_face_orientation():
    torch.set_default_dtype(torch.float64)
    mesh = structured_rectangle(8, 8)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary_logits = torch.randn((4, 8), requires_grad=True)
    row_logits = torch.zeros((system.n_rows, system.max_degree), requires_grad=True)
    boundary = rectangle_boundary_from_logits(boundary_logits)
    output = directed_tutte_embedding_torch_implicit(mesh, boundary, row_logits, system)
    tri = torch.tensor(mesh.faces, dtype=torch.long)
    points = output[tri]
    determinants = (
        (points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
        - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    )
    assert torch.all(determinants > 0.0)
    output.square().sum().backward()
    assert torch.isfinite(boundary_logits.grad).all()
    assert torch.isfinite(row_logits.grad).all()


def test_learnable_modulus_has_positive_height_exact_closure_and_gradient():
    torch.set_default_dtype(torch.float64)
    boundary_logits = torch.zeros((4, 2), requires_grad=True)
    modulus_logit = torch.tensor(-0.3, requires_grad=True)
    boundary = rectangle_boundary_from_modulus_logits(
        boundary_logits, modulus_logit, width=torch.tensor(2.0)
    )
    expected_height = 1e-6 + torch.nn.functional.softplus(modulus_logit)
    assert torch.allclose(boundary[4, 1], expected_height)
    assert torch.allclose(boundary[6, 0], torch.tensor(0.0), atol=1e-12)
    assert torch.allclose(boundary[-1, 1], expected_height / 2.0, atol=1e-12)
    boundary.square().sum().backward()
    assert torch.isfinite(modulus_logit.grad)
    assert abs(float(modulus_logit.grad)) > 1e-8


def test_boundary_accepts_tensor_width_and_height_gradients():
    torch.set_default_dtype(torch.float64)
    logits = torch.zeros((4, 2))
    width = torch.tensor(2.0, requires_grad=True)
    height = torch.tensor(3.0, requires_grad=True)
    boundary = rectangle_boundary_from_logits(logits, width=width, height=height)
    boundary.square().sum().backward()
    assert torch.isfinite(width.grad) and torch.isfinite(height.grad)
