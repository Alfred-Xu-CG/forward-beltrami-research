import numpy as np

from qcopt.forward.rectangle_fd import RectangleFDFactorization, rectangle_beltrami_fd


def test_rectangle_fd_recovers_constant_affine_beltrami_map_from_boundary():
    nx, ny = 40, 32
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.25 + 0.12j
    truth = xx + 1j * yy + mu * (xx - 1j * yy)
    coefficient = np.full((ny, nx), mu, dtype=np.complex128)
    result = rectangle_beltrami_fd(coefficient, truth)
    assert result.converged
    assert result.max_equation_residual < 1e-10
    assert np.max(np.abs(result.map - truth)) < 1e-10


def test_rectangle_fd_requires_boundary_values_and_rejects_degenerate_mu():
    coefficient = np.zeros((8, 8), dtype=np.complex128)
    try:
        rectangle_beltrami_fd(coefficient, np.zeros((7, 8), dtype=np.complex128))
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("bad boundary shape was accepted")
    coefficient[2, 2] = 1.0
    try:
        rectangle_beltrami_fd(coefficient, np.zeros((8, 8), dtype=np.complex128))
    except ValueError as error:
        assert "unit disk" in str(error)
    else:
        raise AssertionError("degenerate coefficient was accepted")


def test_rectangle_factorization_reuses_the_operator_for_multiple_boundaries():
    n = 20
    mu = np.full((n, n), 0.15 + 0.04j, dtype=np.complex128)
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    factorization = RectangleFDFactorization(mu)
    first = factorization.solve(xx + 1j * yy)
    second = factorization.solve(2.0 * xx + 1j * yy)
    assert first.converged and second.converged
    assert first.max_equation_residual < 1e-8
    assert second.max_equation_residual < 1e-8
