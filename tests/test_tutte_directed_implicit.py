import numpy as np
import pytest
import torch

from qcopt.forward.tutte_directed_implicit import (
    DirectedTutteSystem,
    _validate_weakly_convex_boundary,
    directed_tutte_embedding_torch_implicit,
)
from qcopt.mesh import TriMesh, structured_rectangle


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


def test_directed_implicit_rejects_a_concave_target_boundary_before_solving():
    torch.set_default_dtype(torch.float64)
    mesh = structured_rectangle(2, 2)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64)
    boundary[3] = torch.tensor([0.35, 0.15])
    logits = torch.zeros(system.n_rows, system.max_degree, dtype=torch.float64)
    with pytest.raises(ValueError, match="convex"):
        directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)


def test_directed_implicit_supports_a_boundary_only_mesh():
    torch.set_default_dtype(torch.float64)
    mesh = structured_rectangle(1, 1)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64)
    logits = torch.zeros((0, 0), dtype=torch.float64)
    output = directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)
    expected = torch.empty_like(output)
    expected[torch.tensor(system.loop.copy())] = boundary
    assert torch.allclose(output, expected)
    triangle = torch.tensor(mesh.faces.copy())
    points = output[triangle]
    signed_double_area = (
        (points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
        - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    )
    assert torch.all(signed_double_area > 0.0)


def test_boundary_only_backward_respects_loop_to_vertex_permutation():
    mesh = structured_rectangle(1, 1)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.tensor(
        mesh.vertices[mesh.boundary_loops[0]].copy(),
        dtype=torch.float64,
        requires_grad=True,
    )
    logits = torch.zeros((0, 0), dtype=torch.float64)
    coefficients = torch.arange(8, dtype=torch.float64).reshape(4, 2)

    output = directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)
    (output * coefficients).sum().backward()

    torch.testing.assert_close(boundary.grad, coefficients[torch.tensor(system.loop.copy())])


def test_directed_tutte_rejects_a_collinear_boundary_ear():
    vertices = np.asarray(
        ((0.0, 0.0), (0.5, -0.2), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 5), (2, 3, 5), (3, 4, 5), (4, 0, 5)))
    mesh = TriMesh(vertices, faces)
    system = DirectedTutteSystem.from_mesh(mesh)
    target = torch.tensor(
        ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=torch.float64,
    )
    logits = torch.zeros((system.n_rows, system.max_degree), dtype=torch.float64)

    with pytest.raises(ValueError, match="strictly positive face"):
        directed_tutte_embedding_torch_implicit(mesh, target, logits, system)


def test_directed_tutte_rejects_numerically_zero_supported_probabilities():
    mesh = structured_rectangle(2, 2)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.tensor(mesh.vertices[system.loop].copy(), dtype=torch.float64)
    logits = torch.zeros((system.n_rows, system.max_degree), dtype=torch.float64)
    logits[0, system.valid_mask[0]] = -1000.0
    logits[0, int(np.flatnonzero(system.valid_mask[0])[0])] = 0.0

    with pytest.raises(ValueError, match="underflowed to zero"):
        directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)


def test_directed_system_rejects_an_unused_isolated_vertex():
    mesh = TriMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.2, 0.2))),
        np.asarray(((0, 1, 2),)),
    )
    with pytest.raises(ValueError, match="every vertex"):
        DirectedTutteSystem.from_mesh(mesh)


def test_boundary_validator_rejects_a_self_intersecting_star_cycle():
    angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    convex = np.column_stack((np.cos(angles), np.sin(angles)))
    star = convex[[0, 2, 4, 1, 3]]
    with pytest.raises(ValueError, match="convex|simple"):
        _validate_weakly_convex_boundary(star)


def test_directed_implicit_requires_boundary_and_logits_on_same_device():
    torch.set_default_dtype(torch.float64)
    mesh = structured_rectangle(2, 2)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.tensor(mesh.vertices[mesh.boundary_loops[0]], dtype=torch.float64)
    logits = torch.zeros(system.n_rows, system.max_degree, dtype=torch.float64, device="meta")
    with pytest.raises(ValueError, match="device"):
        directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)
