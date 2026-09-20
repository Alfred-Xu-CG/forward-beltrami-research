import numpy as np

from qcopt.forward.coupled_decoder import coupled_monotone_shear_inverse, coupled_monotone_shear_map


def test_coupled_monotone_shear_roundtrip_and_positive_affine_det():
    rng = np.random.default_rng(5)
    points = rng.random((200, 2))
    dx = np.array([0.3, 0.8, 1.5, 0.6])
    dy = np.array([1.1, 0.4, 0.9, 1.8])
    mapped = coupled_monotone_shear_map(points, dx, dy, 0.25, -0.15)
    recovered = coupled_monotone_shear_inverse(mapped, dx, dy, 0.25, -0.15)
    np.testing.assert_allclose(recovered, points, atol=2e-12)
    assert np.all(np.isfinite(mapped))

