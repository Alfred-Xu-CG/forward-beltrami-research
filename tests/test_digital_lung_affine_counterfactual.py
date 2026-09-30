"""Check that the no-label affine counterfactual fits vertex positions."""

import numpy as np

from tools.digital_lung_affine_counterfactual import fit_affine_postmap


def test_exact_positive_affine_fit() -> None:
    x, y = np.meshgrid(np.linspace(0., 1., 5), np.linspace(0., 1., 5))
    source = np.stack((x, y), axis=-1)
    matrix = np.array([[1.1, .2], [-.1, .9]])
    offset = np.array([.07, -.03])
    target = source @ matrix.T + offset
    fitted_matrix, fitted_offset, rms = fit_affine_postmap(source, target)
    assert np.allclose(fitted_matrix, matrix, atol=1e-14)
    assert np.allclose(fitted_offset, offset, atol=1e-14)
    assert rms < 1e-11
    assert np.linalg.det(fitted_matrix) > 0


def test_affine_fit_rejects_rank_deficiency() -> None:
    source = np.zeros((3, 3, 2))
    target = source.copy()
    try:
        fit_affine_postmap(source, target)
    except ValueError as exc:
        assert "rank-deficient" in str(exc)
    else:
        raise AssertionError("rank-deficient fit must fail")
