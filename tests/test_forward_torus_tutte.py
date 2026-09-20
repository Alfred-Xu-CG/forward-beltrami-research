import numpy as np

from qcopt.forward.torus_tutte import (
    periodic_positive_graph_embedding,
    periodic_torus_face_determinants,
    periodic_tutte_embedding,
)


def test_uniform_periodic_tutte_recovers_identity_quasi_periods():
    nx, ny = 12, 10
    values = periodic_tutte_embedding(nx, ny, np.ones((ny, nx)), np.ones((ny, nx)))
    expected = np.column_stack(((np.arange(nx * ny) % nx) / nx, (np.arange(nx * ny) // nx) / ny))
    np.testing.assert_allclose(values, expected, atol=1e-12)
    det = periodic_torus_face_determinants(values, nx, ny)
    assert np.min(det) > 0.0


def test_variable_positive_periodic_weights_keep_a_valid_small_torus_embedding():
    nx, ny = 24, 20
    i = np.arange(nx)[None, :]
    j = np.arange(ny)[:, None]
    wx = 1.0 + 0.25 * np.sin(2.0 * np.pi * (i + 0.3 * j) / nx)
    wy = 1.0 + 0.25 * np.cos(2.0 * np.pi * (j + 0.2 * i) / ny)
    values = periodic_tutte_embedding(nx, ny, wx, wy)
    det = periodic_torus_face_determinants(values, nx, ny)
    assert np.all(np.isfinite(values))
    assert np.min(det) > 0.0


def test_positive_graph_diagonals_preserve_uniform_affine_torus_map():
    nx, ny = 14, 12
    offsets = np.asarray(((1, 0), (0, 1), (1, 1), (1, -1)), dtype=np.int64)
    weights = np.ones((len(offsets), ny, nx), dtype=np.float64)
    values = periodic_positive_graph_embedding(nx, ny, offsets, weights)
    expected = np.column_stack(((np.arange(nx * ny) % nx) / nx, (np.arange(nx * ny) // nx) / ny))
    np.testing.assert_allclose(values, expected, atol=1e-12)
    det = periodic_torus_face_determinants(values, nx, ny)
    assert np.min(det) > 0.0
