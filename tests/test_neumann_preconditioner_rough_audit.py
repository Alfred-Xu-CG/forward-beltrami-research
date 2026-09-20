import numpy as np


def test_rough_field_generator_is_deterministic_and_bounded():
    from qcopt.experiments.neumann_preconditioner_rough_audit import _rough_random_field

    first = _rough_random_field(32, 0.6, np.random.default_rng(17))
    second = _rough_random_field(32, 0.6, np.random.default_rng(17))
    assert first.shape == (32, 32)
    assert np.array_equal(first, second)
    assert np.max(np.abs(first)) <= 0.6000000001
    assert np.all(np.isfinite(first))
