"""Independent source-complex and Floater-boundary regression tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qcopt.forward.tutte_directed_implicit import (
    DirectedTutteSystem,
    directed_tutte_embedding_torch_implicit,
)
from qcopt.mesh import TriMesh, structured_rectangle


def raw_mesh(faces, n_vertices=None, vertices=None):
    """Exercise the source gate independently of TriMesh's weaker constructor.

    Nonplanar/nonmanifold abstract complexes cannot necessarily get through
    TriMesh's coordinate checks. These raw arrays deliberately include a bogus
    cached boundary summary: the topology gate must derive its own boundary.
    No solver or geometric predicate is mocked.
    """
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = n_vertices or int(faces.max()) + 1
    if vertices is None:
        vertices = np.column_stack((np.arange(n_vertices), np.zeros(n_vertices)))
    return SimpleNamespace(
        faces=faces,
        vertices=np.asarray(vertices, dtype=np.float64),
        n_vertices=n_vertices,
        boundary_loops=(np.asarray([0, 1, 2]),),
    )


def test_duplicate_face_hidden_from_existing_boundary_summary_is_rejected():
    mesh = TriMesh(
        np.array([[0, 0], [1, 0], [0, 1], [2, 0], [3, 0], [2, 1.0]]),
        np.array([[0, 1, 2], [3, 4, 5], [3, 4, 5]]),
    )
    assert len(mesh.boundary_loops) == 1  # Duplicated component looks closed.
    with pytest.raises(ValueError, match="duplicate"):
        DirectedTutteSystem.from_mesh(mesh)


def test_disconnected_face_complex_is_rejected_without_trusting_cached_loop():
    mesh = raw_mesh([[0, 1, 2], [3, 4, 5]])
    with pytest.raises(ValueError, match="connected"):
        DirectedTutteSystem.from_mesh(mesh)


def test_shared_edge_must_have_opposite_orientations():
    mesh = raw_mesh([[0, 1, 2], [0, 1, 3]])
    with pytest.raises(ValueError, match="orientation"):
        DirectedTutteSystem.from_mesh(mesh)


def test_edge_with_three_incident_faces_is_rejected():
    mesh = raw_mesh([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    with pytest.raises(ValueError, match="incident"):
        DirectedTutteSystem.from_mesh(mesh)


def test_pinched_vertex_with_disconnected_link_is_rejected():
    mesh = raw_mesh([[0, 1, 2], [0, 3, 4]])
    with pytest.raises(ValueError, match="link"):
        DirectedTutteSystem.from_mesh(mesh)


def test_annulus_is_not_a_disk():
    faces = []
    for a in range(4):
        b = (a + 1) % 4
        faces.extend([[a, b, b + 4], [a, b + 4, a + 4]])
    with pytest.raises(ValueError, match="boundary cycle"):
        DirectedTutteSystem.from_mesh(raw_mesh(faces))


def test_one_boundary_punctured_torus_is_not_a_disk():
    faces = []
    for i in range(3):
        for j in range(3):
            a, b = 3 * i + j, 3 * ((i + 1) % 3) + j
            c, d = 3 * ((i + 1) % 3) + (j + 1) % 3, 3 * i + (j + 1) % 3
            faces.extend([[a, b, c], [a, c, d]])
    # Removing one face gives an orientable manifold with one boundary cycle,
    # but Euler characteristic -1, not +1. Geometry must not be reached.
    with pytest.raises(ValueError, match="Euler"):
        DirectedTutteSystem.from_mesh(raw_mesh(faces[1:]))


def test_source_realization_rejects_positive_faces_with_star_boundary():
    angles = np.arange(5) * 4 * np.pi / 5
    vertices = np.vstack(([[0.0, 0.0]], np.column_stack((np.cos(angles), np.sin(angles)))))
    faces = np.array([[0, i + 1, (i + 1) % 5 + 1] for i in range(5)])
    mesh = TriMesh(vertices, faces)
    assert np.all(mesh.areas > 0)
    with pytest.raises(ValueError, match="simple"):
        DirectedTutteSystem.from_mesh(mesh)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_source_realization_requires_finite_coordinates(bad):
    mesh = raw_mesh([[0, 1, 2]], vertices=[[0, 0], [1, 0], [0, bad]])
    with pytest.raises(ValueError, match="finite"):
        DirectedTutteSystem.from_mesh(mesh)


def test_source_realization_requires_positive_faces_independently():
    mesh = raw_mesh([[0, 1, 2]], vertices=[[0, 0], [0, 1], [1, 0]])
    with pytest.raises(ValueError, match="positive"):
        DirectedTutteSystem.from_mesh(mesh)


def test_source_orientation_uses_exact_stored_coordinates_not_cancelled_determinant():
    n = 2 ** 27
    vertices = np.array([[0, 0], [n, n - 1], [n + 1, n]], dtype=np.float64)
    naive = vertices[1, 0] * vertices[2, 1] - vertices[1, 1] * vertices[2, 0]
    assert naive == 0  # Exact integer determinant is n*n - (n-1)*(n+1) = 1.
    system = DirectedTutteSystem.from_mesh(raw_mesh([[0, 1, 2]], vertices=vertices))
    assert system.n_rows == 0


def ear_mesh():
    return TriMesh(
        np.array([[0, 0], [.5, -.2], [1, 0], [1, 1], [0, 1], [.5, .5]]),
        np.array([[0, 1, 2], [0, 2, 5], [2, 3, 5], [3, 4, 5], [4, 0, 5]]),
    )


def test_floater_dividing_edge_is_rejected_before_factorization(monkeypatch):
    mesh = ear_mesh()
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.tensor(mesh.vertices[system.loop].copy(), dtype=torch.float64)
    boundary[1, 1] = 0.0
    logits = torch.zeros((system.n_rows, system.max_degree), dtype=torch.float64)

    def no_solve(*args, **kwargs):
        pytest.fail("forbidden dividing edge must be rejected before solving")

    monkeypatch.setattr("qcopt.forward.tutte_directed_implicit.factorized", no_solve)
    with pytest.raises(ValueError, match="dividing edge"):
        directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)


def test_noncollapsed_dividing_edge_is_allowed():
    mesh = ear_mesh()
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.tensor(mesh.vertices[system.loop].copy(), dtype=torch.float64)
    logits = torch.zeros((system.n_rows, system.max_degree), dtype=torch.float64)
    output = directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)
    np.testing.assert_allclose(output.numpy(), mesh.vertices, atol=1e-14)


@pytest.mark.parametrize("nx,ny", [(1, 1), (1, 3), (3, 1), (2, 2), (3, 2), (3, 3)])
def test_structured_thin_rectangles_keep_identity_in_vertex_order(nx, ny):
    mesh = structured_rectangle(nx, ny)
    system = DirectedTutteSystem.from_mesh(mesh)
    boundary = torch.tensor(mesh.vertices[system.loop].copy(), dtype=torch.float64)
    logits = torch.zeros((system.n_rows, system.max_degree), dtype=torch.float64)
    output = directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)
    np.testing.assert_allclose(output.numpy(), mesh.vertices, atol=1e-14)


def test_valid_nonconvex_source_disk_is_allowed():
    mesh = TriMesh(
        np.array([[0, 0], [2, 0], [1, .5], [0, 1.0]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    system = DirectedTutteSystem.from_mesh(mesh)
    assert system.n_rows == 0


@pytest.mark.parametrize("change", ["faces", "vertices"])
def test_cached_certificate_cannot_be_used_for_another_same_size_mesh(change):
    mesh = structured_rectangle(2, 2)
    system = DirectedTutteSystem.from_mesh(mesh)
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    if change == "faces":
        faces[:2] = [[0, 1, 3], [1, 4, 3]]
    else:
        vertices[4] += [.02, 0]
    other = TriMesh(vertices, faces)
    boundary = torch.tensor(mesh.vertices[system.loop].copy(), dtype=torch.float64)
    logits = torch.zeros((system.n_rows, system.max_degree), dtype=torch.float64)
    with pytest.raises(ValueError, match="different mesh"):
        directed_tutte_embedding_torch_implicit(other, boundary, logits, system)
