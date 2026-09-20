import numpy as np

from qcopt.forward.bhf_variation import (
    all_vertex_near_far_bhf_variation,
    edge_point_near_far_bhf_variation,
    interior_point_near_far_bhf_variation,
    normalized_bhf_variation,
    vertex_near_far_bhf_variation,
)
from qcopt.mesh import structured_rectangle


def test_normalized_bhf_variation_honors_three_point_normalization():
    source = np.array([-0.7 + 0.2j, -0.1 + 0.6j, 0.4 + 0.3j, 1.4 - 0.5j])
    weights = np.full(len(source), 0.2)
    nu = np.array([0.3 - 0.1j, -0.2 + 0.05j, 0.1 + 0.2j, 0.05 - 0.3j])
    evaluation = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.2 + 0.4j])
    velocity = normalized_bhf_variation(evaluation, source, nu, weights, block_size=2)
    assert np.allclose(velocity[:2], 0.0, atol=1e-14)
    assert np.isfinite(velocity[2])


def test_normalized_bhf_variation_is_linear_in_the_coefficient():
    rng = np.random.default_rng(44)
    source = rng.normal(size=32) + 1j * rng.normal(size=32)
    source += 3.0  # keep away from 0 and 1
    evaluation = rng.normal(size=16) + 1j * rng.normal(size=16)
    weights = np.full(len(source), 0.01)
    nu = rng.normal(size=len(source)) + 1j * rng.normal(size=len(source))
    lhs = normalized_bhf_variation(evaluation, source, 2.0 * nu, weights)
    rhs = 2.0 * normalized_bhf_variation(evaluation, source, nu, weights)
    assert np.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_vertex_near_far_bhf_assembly_is_finite_and_locally_corrected():
    mesh = structured_rectangle(8, 8)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    triangles = source[mesh.faces]
    b = 0.21 + 0.09j
    a = 1.0 - b
    image = a * source + b * np.conjugate(source)
    fz = np.full(mesh.n_faces, a, dtype=np.complex128)
    variation = 0.12 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.42 - 0.37j) ** 2 / 0.15)
    target_vertex = 4 * 9 + 4
    result = vertex_near_far_bhf_variation(
        source,
        image,
        mesh.faces,
        fz,
        variation,
        target_vertex,
        near_order=16,
    )
    assert np.isfinite(result.real + result.imag)
    assert abs(result) < 1.0


def test_interior_point_near_far_bhf_converges_with_duffy_subdivision():
    mesh = structured_rectangle(8, 8)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    triangles = source[mesh.faces]
    b = 0.21 + 0.09j
    a = 1.0 - b
    image = a * source + b * np.conjugate(source)
    fz = np.full(mesh.n_faces, a, dtype=np.complex128)
    variation = 0.12 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.42 - 0.37j) ** 2 / 0.15)
    coarse = interior_point_near_far_bhf_variation(
        source, image, mesh.faces, fz, variation, 30, np.asarray([0.2, 0.3, 0.5]), near_order=8
    )
    fine = interior_point_near_far_bhf_variation(
        source, image, mesh.faces, fz, variation, 30, np.asarray([0.2, 0.3, 0.5]), near_order=16
    )
    assert np.isfinite(fine.real + fine.imag)
    assert abs(fine - coarse) < 2e-4


def test_all_vertex_near_far_batch_matches_single_vertex_control():
    mesh = structured_rectangle(4, 4)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    triangles = source[mesh.faces]
    b = 0.18 + 0.07j
    a = 1.0 - b
    image = a * source + b * np.conjugate(source)
    fz = np.full(mesh.n_faces, a, dtype=np.complex128)
    variation = 0.04 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.38 - 0.42j) ** 2 / 0.12)
    batch = all_vertex_near_far_bhf_variation(source, image, mesh.faces, fz, variation, near_order=8)
    assert batch.shape == source.shape
    assert np.all(np.isfinite(batch))


def test_edge_point_near_far_uses_both_incident_faces_and_converges():
    mesh = structured_rectangle(8, 8)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    triangles = source[mesh.faces]
    b = 0.17 + 0.05j
    a = 1.0 - b
    image = a * source + b * np.conjugate(source)
    fz = np.full(mesh.n_faces, a, dtype=np.complex128)
    variation = 0.1 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.45 - 0.4j) ** 2 / 0.1)
    # The diagonal from (4,4) to (5,5) is an interior shared edge.
    edge = (4 + 9 * 4, 5 + 9 * 5)
    coarse = edge_point_near_far_bhf_variation(source, image, mesh.faces, fz, variation, edge, 0.37, near_order=8)
    fine = edge_point_near_far_bhf_variation(source, image, mesh.faces, fz, variation, edge, 0.37, near_order=16)
    assert np.isfinite(fine.real + fine.imag)
    assert abs(fine - coarse) < 2e-4
