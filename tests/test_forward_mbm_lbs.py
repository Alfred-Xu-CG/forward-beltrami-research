from qcopt.forward.mbm_lbs import solve_mbm_lbs

import numpy as np


def test_real_constant_coefficient_has_exact_rectangle_solution() -> None:
    n = 12
    coefficient = 0.2
    result = solve_mbm_lbs(np.full((n, n), coefficient, dtype=np.complex128))
    expected_modulus = (1.0 - coefficient) / (1.0 + coefficient)
    assert abs(result.modulus - expected_modulus) < 1e-10
    assert result.conjugacy_residual < 1e-10
    assert result.min_triangle_determinant > 0.0
    assert result.left_monotone and result.right_monotone


def test_zero_coefficient_is_identity_rectangle() -> None:
    result = solve_mbm_lbs(np.zeros((8, 10), dtype=np.complex128))
    assert abs(result.modulus - 1.0) < 1e-10
    assert result.conjugacy_residual < 1e-10
    assert result.min_triangle_determinant > 0.0
