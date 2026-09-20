import numpy as np

from qcopt.forward.coupled_decoder import triangular_spatial_monotone_inverse, triangular_spatial_monotone_map


def test_spatial_triangular_decoder_round_trips_and_is_coupled():
    rng = np.random.default_rng(31)
    points = rng.random((2000, 2))
    x_inc = 0.2 + rng.random(24)
    y_inc = 0.2 + rng.random((20, 28))
    mapped = triangular_spatial_monotone_map(points, x_inc, y_inc)
    recovered = triangular_spatial_monotone_inverse(mapped, x_inc, y_inc)
    assert np.max(np.abs(recovered - points)) < 2e-14
    pair = np.asarray([[0.2, 0.5], [0.8, 0.5]])
    pair_mapped = triangular_spatial_monotone_map(pair, x_inc, y_inc)
    assert abs(pair_mapped[0, 1] - pair_mapped[1, 1]) > 1e-5


def test_spatial_triangular_decoder_preserves_rectangle_boundaries():
    x_inc = np.ones(8)
    y_inc = np.ones((7, 9))
    boundary = np.asarray([[0.0, 0.3], [1.0, 0.3], [0.4, 0.0], [0.4, 1.0]])
    mapped = triangular_spatial_monotone_map(boundary, x_inc, y_inc)
    np.testing.assert_allclose(mapped[[0, 1], 0], boundary[[0, 1], 0])
    np.testing.assert_allclose(mapped[[2, 3], 1], boundary[[2, 3], 1])
