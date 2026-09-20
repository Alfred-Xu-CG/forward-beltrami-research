import numpy as np
import torch

from qcopt.experiments.phase3_mbm_vjp_benchmark import (
    _induced_face_mu_accuracy,
    _mapped_face_metrics,
)
from qcopt.forward.mbm_lbs import solve_mbm_lbs
from qcopt.forward.mbm_lbs_implicit import mbm_lbs_torch_implicit


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


def test_mapped_face_metrics_are_evaluated_on_implicit_output():
    n = 7
    yy, xx = np.mgrid[0:n, 0:n]
    x = xx / (n - 1) - 0.5
    y = yy / (n - 1) - 0.5
    coefficient = 0.12 * np.exp(-(x * x + y * y) / 0.18) * np.exp(0.7j)
    mapped = mbm_lbs_torch_implicit(
        torch.tensor(coefficient.real), torch.tensor(coefficient.imag)
    )
    metrics = _mapped_face_metrics(coefficient, mapped)
    assert set(metrics) == {"implicit_conjugacy_residual", "implicit_min_face_determinant"}
    assert all(np.isfinite(value) for value in metrics.values())
