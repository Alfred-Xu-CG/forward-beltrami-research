import numpy as np
import torch

from qcopt.forward.mbm_lbs_implicit import mbm_lbs_torch_implicit


def test_mbm_implicit_vjp_matches_directional_finite_difference() -> None:
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(3)
    n = 7
    mu_real = torch.tensor(0.12 * rng.normal(size=(n, n)), requires_grad=True)
    mu_imag = torch.tensor(0.08 * rng.normal(size=(n, n)), requires_grad=True)
    output = mbm_lbs_torch_implicit(mu_real, mu_imag)
    loss = (output * output).sum()
    loss.backward()

    direction_real = torch.tensor(rng.normal(size=(n, n)))
    direction_imag = torch.tensor(rng.normal(size=(n, n)))
    predicted = float(
        (mu_real.grad * direction_real + mu_imag.grad * direction_imag).sum()
    )
    epsilon = 1e-6
    with torch.no_grad():
        plus = (
            mbm_lbs_torch_implicit(
                mu_real + epsilon * direction_real,
                mu_imag + epsilon * direction_imag,
            )
            ** 2
        ).sum()
        minus = (
            mbm_lbs_torch_implicit(
                mu_real - epsilon * direction_real,
                mu_imag - epsilon * direction_imag,
            )
            ** 2
        ).sum()
    finite_difference = float((plus - minus) / (2.0 * epsilon))
    relative_error = abs(predicted - finite_difference) / max(
        abs(finite_difference), 1e-12
    )
    assert relative_error < 1e-6


def test_mbm_implicit_rejects_small_grids_and_mismatched_devices() -> None:
    with np.testing.assert_raises(ValueError):
        mbm_lbs_torch_implicit(torch.zeros((2, 3)), torch.zeros((2, 3)))
    with np.testing.assert_raises(ValueError):
        mbm_lbs_torch_implicit(torch.zeros((3, 3)), torch.zeros((3, 4)))
    if torch.cuda.is_available():
        with np.testing.assert_raises(ValueError):
            mbm_lbs_torch_implicit(torch.zeros((3, 3)), torch.zeros((3, 3), device="cuda"))
