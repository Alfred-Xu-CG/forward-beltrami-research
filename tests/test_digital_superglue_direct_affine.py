"""Direct-similarity algebra does not require the external matcher."""

import numpy as np

from tools.digital_superglue_direct_affine import similarity_from_inlier_units


def test_similarity_recovers_positive_rotated_scale():
    source = np.array([[.1, .2], [.7, .2], [.2, .8], [.9, .9], [.4, .5]])
    truth = np.array([[.94, -.17], [.17, .94]])
    offset = np.array([.03, -.04])
    target = np.column_stack((
        truth[0, 0] * source[:, 0] + truth[0, 1] * source[:, 1],
        truth[1, 0] * source[:, 0] + truth[1, 1] * source[:, 1],
    )) + offset
    matrix, estimate = similarity_from_inlier_units(source, target)
    np.testing.assert_allclose(matrix, truth, rtol=0, atol=1e-15)
    np.testing.assert_allclose(estimate, offset, rtol=0, atol=1e-15)
    assert matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0] > 0
