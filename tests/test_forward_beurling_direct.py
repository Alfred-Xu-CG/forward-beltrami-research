import numpy as np

from qcopt.forward.beurling_direct import direct_beurling_apply


def test_direct_beurling_quadrature_is_linear_and_handles_blocking():
    rng = np.random.default_rng(3)
    points = rng.random(96) + 1j * rng.random(96)
    weights = np.full(len(points), 1.0 / len(points))
    values = np.exp(-((points.real - 0.5) ** 2 + (points.imag - 0.5) ** 2) / 0.1)
    lhs = direct_beurling_apply(points, 2.0 * values, weights, block_size=17)
    rhs = 2.0 * direct_beurling_apply(points, values, weights, block_size=31)
    assert np.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_direct_beurling_supports_disjoint_targets_without_self_singularity():
    rng = np.random.default_rng(31)
    source = 0.2 + 0.2 * rng.random(64) + 1j * (0.2 + 0.6 * rng.random(64))
    target = 0.6 + 0.2 * rng.random(32) + 1j * (0.2 + 0.6 * rng.random(32))
    values = np.exp(-((source.real - 0.3) ** 2 + (source.imag - 0.5) ** 2) / 0.03)
    weights = np.full(len(source), 0.04 / len(source))
    result = direct_beurling_apply(source, values, weights, target_points=target)
    assert result.shape == target.shape
    assert np.all(np.isfinite(result))
