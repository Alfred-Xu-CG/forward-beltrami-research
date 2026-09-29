import numpy as np

from tools.digital_q1_p1_eval import p1_map_at_unit_queries
from tools.digital_teacher_fine_detail import p1_refine


def test_p1_refinement_preserves_coarse_map_and_uses_diagonal_center() -> None:
    axis = np.linspace(0, 1, 3)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    coarse = np.stack((xx, yy), axis=-1)
    coarse[1, 1] += [.1, -.05]
    fine = p1_refine(coarse)
    np.testing.assert_array_equal(fine[::2, ::2], coarse)
    np.testing.assert_array_equal(fine[1::2, 1::2],
                                  (coarse[:-1, :-1] + coarse[1:, 1:]) / 2)
    fine_axis = np.linspace(0, 1, 5)
    yy_fine, xx_fine = np.meshgrid(fine_axis, fine_axis, indexing="ij")
    queries = np.stack((xx_fine, yy_fine), axis=-1).reshape(-1, 2)
    np.testing.assert_allclose(
        fine.reshape(-1, 2), p1_map_at_unit_queries(coarse, queries), atol=1e-15)
    assert not np.array_equal(fine[1, 1],
                              (coarse[0, 0] + coarse[0, 1]
                               + coarse[1, 0] + coarse[1, 1]) / 4)
