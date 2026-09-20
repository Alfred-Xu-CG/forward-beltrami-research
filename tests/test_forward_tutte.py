import numpy as np

from qcopt.forward.tutte import tutte_embedding, tutte_embedding_weighted, tutte_weights
from qcopt.mesh import structured_rectangle
from qcopt.injectivity import audit_injectivity


def test_uniform_tutte_embedding_preserves_a_convex_rectangle_boundary():
    mesh = structured_rectangle(12, 10)
    loop = mesh.boundary_loops[0]
    target_boundary = 1.5 * mesh.vertices[loop] + np.asarray([0.2, -0.1])

    uv = tutte_embedding(mesh, target_boundary)
    report = audit_injectivity(mesh, uv)

    assert report.certified
    assert np.max(np.abs(uv[loop] - target_boundary)) < 1e-12
    assert np.min(np.linalg.det(_face_jacobians(mesh, uv))) > 0.0


def test_tutte_embedding_rejects_a_nonconvex_target_boundary():
    mesh = structured_rectangle(4, 4)
    loop = mesh.boundary_loops[0]
    target = mesh.vertices[loop].copy()
    target[len(target) // 2, 1] = -0.2

    try:
        tutte_embedding(mesh, target)
    except ValueError as error:
        assert "convex" in str(error)
    else:
        raise AssertionError("nonconvex boundary was accepted")


def test_positive_nonuniform_tutte_weights_preserve_hard_injectivity():
    mesh = structured_rectangle(16, 12)
    loop = mesh.boundary_loops[0]
    target = 1.2 * mesh.vertices[loop] + np.asarray([0.1, -0.05])
    edges = set()
    for a, b, c in mesh.faces.tolist():
        edges.update((min(a, b), max(a, b)) for a, b in ((a, b), (b, c), (c, a)))
    weights = {
        edge: float(np.exp(-2.0 * np.linalg.norm(mesh.vertices[edge[0]] - mesh.vertices[edge[1]])))
        for edge in edges
    }
    embedding = tutte_embedding_weighted(mesh, target, weights)
    report = audit_injectivity(mesh, embedding)
    assert report.certified
    assert np.min(np.linalg.det(_face_jacobians(mesh, embedding))) > 0.0


def test_tutte_weights_are_nonnegative_partition_of_unity_and_reproduce_embedding():
    mesh = structured_rectangle(8, 7)
    loop = mesh.boundary_loops[0]
    target = 1.3 * mesh.vertices[loop] + np.asarray([0.1, -0.2])

    weights = tutte_weights(mesh)
    embedding = tutte_embedding(mesh, target)

    assert weights.shape == (mesh.n_vertices, len(loop))
    assert np.all(weights >= -1e-12)
    assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-12)
    assert np.allclose(weights[loop], np.eye(len(loop)), atol=1e-12)
    assert np.allclose(embedding, weights @ target, atol=1e-12)


def test_uniform_tutte_decoder_is_not_expressive_for_nonharmonic_interior_truth():
    mesh = structured_rectangle(20, 20)
    loop = mesh.boundary_loops[0]
    target = mesh.vertices[loop].copy()
    truth = mesh.vertices.copy()
    x, y = truth[:, 0], truth[:, 1]
    truth[:, 0] += 0.12 * np.sin(np.pi * x) ** 2 * np.sin(np.pi * y)
    truth[:, 1] += 0.08 * np.sin(np.pi * x) * np.sin(np.pi * y) ** 2
    decoded = tutte_embedding(mesh, target)
    interior = np.ones(mesh.n_vertices, dtype=bool)
    interior[loop] = False
    assert np.max(np.abs(decoded[loop] - truth[loop])) < 1e-12
    assert np.max(np.linalg.norm(decoded[interior] - truth[interior], axis=1)) > 1e-2


def _face_jacobians(mesh, uv):
    p0, p1, p2 = (uv[mesh.faces[:, i]] for i in range(3))
    return np.stack((p1 - p0, p2 - p0), axis=-1)
