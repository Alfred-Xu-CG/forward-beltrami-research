import numpy as np

from qcopt.forward.tutte import tutte_embedding_weighted
from qcopt.forward.tutte_weighted_implicit import weighted_tutte_vjp
from qcopt.mesh import structured_rectangle


def _edge_weights(mesh):
    edges = set()
    for face in mesh.faces:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = sorted((int(face[i]), int(face[j])))
            edges.add((a, b))
    return {edge: 0.5 + 1.4 * ((edge[0] * 17 + edge[1] * 7) % 11) / 10 for edge in edges}


def test_weighted_tutte_vjp_matches_edge_finite_difference():
    mesh = structured_rectangle(5, 5)
    loop = mesh.boundary_loops[0]
    theta = np.linspace(0.0, 2.0 * np.pi, len(loop), endpoint=False)
    target = np.column_stack((np.cos(theta), np.sin(theta)))
    weights = _edge_weights(mesh)
    rng = np.random.default_rng(4)
    grad = rng.normal(size=(mesh.n_vertices, 2))
    result, edge_grad = weighted_tutte_vjp(mesh, target, weights, grad)
    key = sorted(weights)[len(weights) // 2]
    eps = 1e-6
    plus = dict(weights)
    minus = dict(weights)
    plus[key] += eps
    minus[key] -= eps
    loss_plus = np.sum(grad * tutte_embedding_weighted(mesh, target, plus))
    loss_minus = np.sum(grad * tutte_embedding_weighted(mesh, target, minus))
    finite_difference = (loss_plus - loss_minus) / (2.0 * eps)
    assert np.isfinite(edge_grad[key])
    np.testing.assert_allclose(edge_grad[key], finite_difference, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(result, tutte_embedding_weighted(mesh, target, weights))

