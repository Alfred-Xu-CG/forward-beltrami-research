import numpy as np
import pytest

from tools.digital_boundary_error_floor import boundary_floor


def _identity(side: int) -> np.ndarray:
    axis = np.linspace(0, 1, side)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx, yy), axis=-1)


def test_boundary_error_is_unavoidable_part_of_full_vector_rmse() -> None:
    reference = _identity(3)
    teacher = reference.copy()
    teacher[0, 1, 0] += .2
    candidate = reference.copy()
    candidate[1, 1, 1] += .1
    row = boundary_floor(reference, np.eye(2), np.zeros(2), teacher, candidate)
    np.testing.assert_allclose(row["boundary_floor_vector_rmse"], .2 / 3)
    np.testing.assert_allclose(row["candidate_full_vector_rmse"], 5 ** .5 / 30)
    np.testing.assert_allclose(row["boundary_fraction_of_squared_error"], .8)
    assert row["boundary_vertices"] == 8


def test_boundary_movement_is_rejected_from_fixed_contract() -> None:
    reference = _identity(3)
    candidate = reference.copy()
    candidate[0, 1, 0] += .01
    with pytest.raises(ValueError, match="boundary"):
        boundary_floor(reference, np.eye(2), np.zeros(2), reference, candidate)
