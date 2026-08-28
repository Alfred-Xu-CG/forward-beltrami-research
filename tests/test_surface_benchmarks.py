import numpy as np

from qcopt.surface_benchmarks import (
    deterministic_face_samples,
    make_double_torus,
    random_smooth_tangent_field,
    select_farthest_landmarks,
)


def test_implicit_double_torus_is_closed_oriented_genus_two():
    mesh = make_double_torus(resolution=20)
    assert mesh.topology.closed
    assert mesh.topology.connected
    assert mesh.topology.oriented
    assert mesh.topology.euler_characteristic == -2
    assert mesh.topology.genus == 2


def test_sampling_landmarks_and_random_field_are_deterministic():
    mesh = make_double_torus(resolution=20)
    samples = deterministic_face_samples(mesh, 120)
    first_landmarks = select_farthest_landmarks(mesh, samples, 10)
    second_landmarks = select_farthest_landmarks(mesh, samples, 10)
    first_field = random_smooth_tangent_field(mesh, seed=17, relative_amplitude=1.5)
    second_field = random_smooth_tangent_field(mesh, seed=17, relative_amplitude=1.5)

    assert len(samples) == 120
    assert np.array_equal(first_landmarks, second_landmarks)
    assert len(np.unique(first_landmarks)) == 10
    assert np.allclose(first_field, second_field)
    assert np.all(np.isfinite(first_field))
    assert np.isclose(
        np.max(np.linalg.norm(first_field, axis=1)),
        1.5 * mesh.median_edge_length,
    )
