import numpy as np

from tools.digital_recurrent_round_trace import (
    corner_min_and_nonpositive, vector_rmse,
)


def test_vector_rmse_uses_full_euclidean_vector_and_output_affine():
    vertices = np.array([[[0., 0.], [1., 0.]]])
    matrix = np.array([[2., 0.], [0., 3.]])
    offset = np.array([.5, -.5])
    teacher = np.array([[[.5, -.5], [2.5, 1.5]]])
    np.testing.assert_allclose(vector_rmse(vertices, matrix, offset, teacher),
                               np.sqrt(2.), rtol=0, atol=1e-15)


def test_all_four_corner_signs_on_one_affine_cell():
    vertices = np.array([[[0., 0.], [1., 0.]],
                         [[0., 2.], [1., 2.]]])
    assert corner_min_and_nonpositive(vertices) == (2., 0)
    vertices[1, 1] = (-1., -1.)
    minimum, failures = corner_min_and_nonpositive(vertices)
    assert minimum < 0 and failures > 0
