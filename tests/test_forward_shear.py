import numpy as np

from qcopt.forward.shear import (
    affine_shear_log_abs_det,
    affine_shear_matrix,
    apply_affine_shear,
    invert_affine_shear,
)


def test_affine_shear_is_exactly_invertible_and_orientation_preserving():
    rng = np.random.default_rng(8)
    points = rng.normal(size=(1000, 2))
    mapped = apply_affine_shear(points, 2.3, -1.7)
    recovered = invert_affine_shear(mapped, 2.3, -1.7)
    assert np.max(np.abs(recovered - points)) < 1e-12
    assert np.isclose(np.linalg.det(affine_shear_matrix(2.3, -1.7)), 1.0)
    assert affine_shear_log_abs_det(2.3, -1.7) == 0.0
