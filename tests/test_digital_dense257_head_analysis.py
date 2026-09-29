import numpy as np

from tools.digital_dense257_head_analysis import _q1_upsample


def test_q1_upsample_preserves_coarse_vertices_and_bilinear_center() -> None:
    coarse = np.arange(18, dtype=np.float64).reshape(3, 3, 2)
    fine = _q1_upsample(coarse)
    assert fine.shape == (5, 5, 2)
    np.testing.assert_array_equal(fine[::2, ::2], coarse)
    np.testing.assert_array_equal(fine[1, 1],
                                  .25 * (coarse[0, 0] + coarse[0, 1]
                                         + coarse[1, 0] + coarse[1, 1]))
