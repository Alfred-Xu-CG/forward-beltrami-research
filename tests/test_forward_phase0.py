import numpy as np

from qcopt.forward.benchmarks import affine_map, manufactured_mu, smooth_twist_map
from qcopt.forward.operators import (
    beltrami_residual,
    evaluate_forward_metrics,
    face_jacobian_determinants,
    periodic_grid_edges,
)
from qcopt.mesh import structured_rectangle


def test_affine_manufactured_map_has_constant_exact_beltrami_coefficient():
    mesh = structured_rectangle(4, 3)
    matrix = np.asarray([[1.25, 0.30], [0.10, 0.90]], dtype=np.float64)
    uv = affine_map(mesh.vertices, matrix, np.asarray([0.2, -0.1]))

    mu = manufactured_mu(mesh, uv)

    assert np.allclose(mu, mu[0], atol=1e-13, rtol=1e-13)
    assert np.all(face_jacobian_determinants(mesh, uv) > 0.0)
    assert np.max(beltrami_residual(mesh, uv, mu)) < 1e-12


def test_smooth_twist_manufactured_map_has_positive_face_jacobians():
    mesh = structured_rectangle(32, 24)
    uv = smooth_twist_map(mesh.vertices, amplitude=0.08)

    determinants = face_jacobian_determinants(mesh, uv)

    assert np.min(determinants) > 0.5
    assert np.max(beltrami_residual(mesh, uv, manufactured_mu(mesh, uv))) < 1e-12


def test_periodic_grid_edges_include_both_coordinate_wraps():
    edges = periodic_grid_edges(3, 2)
    undirected = {tuple(sorted(edge)) for edge in edges.tolist()}

    assert (0, 2) in undirected
    assert (0, 3) in undirected
    assert len(undirected) == len(edges)


def test_topology_metrics_certify_identity_independently_of_zero_residual():
    mesh = structured_rectangle(5, 4)
    target_mu = np.zeros(mesh.n_faces, dtype=np.complex128)

    metrics = evaluate_forward_metrics(mesh, mesh.vertices, target_mu)

    assert metrics.topology_certified
    assert metrics.flipped_faces == ()
    assert metrics.min_det == 1.0
    assert metrics.mu_l2 < 1e-14
    assert metrics.mu_linf < 1e-14
    assert metrics.equation_linf < 1e-14


def test_topology_metrics_reject_a_fold_without_using_solver_residual_as_evidence():
    mesh = structured_rectangle(1, 1)
    folded = mesh.vertices.copy()
    folded[3] = np.asarray([-0.25, 0.25])

    metrics = evaluate_forward_metrics(mesh, folded)

    assert not metrics.topology_certified
    assert metrics.flipped_faces == (1,)
    assert metrics.min_det < 0.0
    assert np.isnan(metrics.mu_l2)
    assert np.isnan(metrics.equation_linf)
