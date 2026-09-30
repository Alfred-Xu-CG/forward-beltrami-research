"""Pure numerical checks for the retrospective match-coverage readout."""

import numpy as np

from tools.digital_lung_match_coverage_audit import _correlation, _ranks


def test_ranks_and_spearman_direction() -> None:
    distance = np.array([4., 1., 3., 2.])
    gain = np.array([1., 4., 2., 3.])
    assert np.array_equal(_ranks(distance), np.array([3., 0., 2., 1.]))
    assert np.isclose(_correlation(_ranks(distance), _ranks(gain)), -1.)


def test_constant_direction_reports_undefined_correlation() -> None:
    assert np.isnan(_correlation(np.array([1., 1.]), np.array([2., 3.])))
