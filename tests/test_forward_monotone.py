import numpy as np

from qcopt.forward.monotone import (
    monotone_axis_inverse,
    monotone_axis_map,
    separable_monotone_map,
)


def test_positive_increment_decoder_is_a_rectangle_bijection():
    rng = np.random.default_rng(12)
    points = rng.random((2000, 2))
    dx = np.exp(rng.normal(size=17))
    dy = np.exp(rng.normal(size=13))
    mapped = separable_monotone_map(points, dx, dy)
    recovered = np.column_stack(
        (
            monotone_axis_inverse(mapped[:, 0], dx),
            monotone_axis_inverse(mapped[:, 1], dy),
        )
    )
    assert np.max(np.abs(recovered - points)) < 2e-3
    assert np.all(mapped >= 0.0) and np.all(mapped <= 1.0)


def test_monotone_axis_rejects_nonpositive_increment():
    try:
        monotone_axis_map(np.array([0.2]), np.array([1.0, 0.0]))
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive increment was accepted")
