import numpy as np

from qcopt.experiments.phase3_mbm_vjp_benchmark import (
    _induced_face_mu_accuracy,
)
from qcopt.forward.mbm_lbs import solve_mbm_lbs


def test_induced_face_mu_accuracy_is_finite_on_small_reference():
    n = 7
    yy, xx = np.mgrid[0:n, 0:n]
    x = xx / (n - 1) - 0.5
    y = yy / (n - 1) - 0.5
    coefficient = 0.12 * np.exp(-(x * x + y * y) / 0.18) * np.exp(0.7j)
    result = solve_mbm_lbs(coefficient)
    metrics = _induced_face_mu_accuracy(coefficient, result)
    assert set(metrics) == {
        "induced_mu_rmse",
        "induced_mu_max_error",
        "induced_mu_max_abs",
    }
    assert all(np.isfinite(value) and value >= 0.0 for value in metrics.values())
