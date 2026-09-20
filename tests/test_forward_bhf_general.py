import numpy as np

from qcopt.forward.bhf_variation import (
    arbitrary_base_bhf_variation,
    bhf_kernel_R,
    normalized_bhf_variation,
    triangle_bhf_variation,
    duffy_triangle_kernel_integral,
)


def test_arbitrary_base_bhf_variation_has_three_point_normalization_and_linearity():
    rng = np.random.default_rng(12)
    source = (rng.random(128) + 1j * rng.random(128)) * 1.6 - 0.8 - 0.2j
    evaluation = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.2 + 0.3j, -0.35 + 0.1j])
    b = 0.2 + 0.1j
    a = 1.0 - b  # f(0)=0 and f(1)=1 for f(z)=a*z+b*conj(z)
    f_source = a * source + b * np.conjugate(source)
    f_evaluation = a * evaluation + b * np.conjugate(evaluation)
    fz = np.full(source.shape, a, dtype=np.complex128)
    nu = 0.15 * np.exp(-np.abs(source - (0.1 + 0.2j)) ** 2 / 0.6)
    weights = np.full(source.shape, 2.56 / source.size)

    velocity = arbitrary_base_bhf_variation(
        evaluation,
        source,
        f_evaluation,
        f_source,
        fz,
        nu,
        weights,
        block_size=32,
    )
    doubled = arbitrary_base_bhf_variation(
        evaluation,
        source,
        f_evaluation,
        f_source,
        fz,
        2.0 * nu,
        weights,
        block_size=32,
    )
    np.testing.assert_allclose(velocity[:2], 0.0, atol=1e-14)
    np.testing.assert_allclose(doubled, 2.0 * velocity, rtol=1e-13, atol=1e-14)
    assert np.all(np.isfinite(velocity))


def test_bhf_kernel_reduces_to_normalized_rational_kernel_at_identity():
    z = np.array([0.23 + 0.17j, -0.4 + 0.2j])
    w = 0.37 - 0.11j
    expected = w * (w - 1.0) / (z * (z - 1.0) * (z - w))
    np.testing.assert_allclose(bhf_kernel_R(z, w), expected)


def test_general_operator_matches_normalized_operator_at_identity_base():
    rng = np.random.default_rng(8)
    source = (rng.random(96) + 1j * rng.random(96)) * 1.5 - 0.75 + 0.24j
    evaluation = np.array([0.19 + 0.21j, -0.31 + 0.18j])
    nu = 0.2 * np.exp(-np.abs(source - 0.1 - 0.3j) ** 2)
    weights = np.full(source.shape, 2.25 / source.size)
    old = normalized_bhf_variation(evaluation, source, nu, weights)
    new = arbitrary_base_bhf_variation(
        evaluation,
        source,
        evaluation,
        source,
        np.ones(source.shape, dtype=np.complex128),
        nu,
        weights,
    )
    np.testing.assert_allclose(new, old, rtol=2e-13, atol=2e-14)


def test_triangle_quadrature_operator_preserves_normalization():
    triangles = np.array(
        [
            [-0.8 - 0.6j, 0.1 - 0.7j, -0.3 + 0.1j],
            [0.1 - 0.7j, 0.8 + 0.5j, -0.3 + 0.1j],
        ]
    )
    evaluation = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.23 + 0.17j])
    variation = np.array([0.1 + 0.2j, -0.08 + 0.04j])
    velocity = triangle_bhf_variation(
        evaluation,
        evaluation,
        triangles,
        triangles,
        np.ones(2, dtype=np.complex128),
        variation,
    )
    np.testing.assert_allclose(velocity[:2], 0.0, atol=1e-14)
    assert np.all(np.isfinite(velocity))


def test_duffy_triangle_rule_handles_vertex_kernel_pole_with_convergent_values():
    source = np.array([0.2 + 0.1j, 0.9 + 0.15j, 0.25 + 0.8j])
    b = 0.2 + 0.08j
    a = 1.0 - b
    image = a * source + b * np.conjugate(source)
    values = [duffy_triangle_kernel_integral(source, image, image[0], vertex=0, order=o) for o in (8, 16, 24)]
    assert np.all(np.isfinite(values))
    assert abs(values[-1] - values[-2]) < 1e-7
