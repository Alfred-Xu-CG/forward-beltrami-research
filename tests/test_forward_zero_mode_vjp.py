import numpy as np

from qcopt.forward.beurling_implicit import torus_affine_map_vjp, torus_affine_period_vjp


def test_torus_affine_period_vjp_matches_real_finite_difference():
    mu = 0.2 + 0.13j
    gp = -0.7 + 0.4j
    gq = 0.3 - 0.2j
    direction = -0.11 + 0.17j

    def loss(value):
        p = 1.0 + value
        q = 1j * (1.0 - value)
        return float(np.real(np.conjugate(gp) * p + np.conjugate(gq) * q))

    eps = 1e-7
    finite_difference = (loss(mu + eps * direction) - loss(mu - eps * direction)) / (2.0 * eps)
    gradient = torus_affine_period_vjp(gp, gq)
    np.testing.assert_allclose(finite_difference, np.real(np.conjugate(gradient) * direction), rtol=1e-7, atol=1e-9)


def test_torus_affine_map_vjp_includes_map_and_period_zero_modes():
    points = np.asarray([0.0 + 0.0j, 0.25 + 0.1j, 0.7 + 0.6j, 0.9 + 0.2j])
    cotangent = np.asarray([0.2 - 0.4j, -0.1 + 0.3j, 0.5 + 0.2j, -0.6 + 0.1j])
    gp, gq = -0.3 + 0.2j, 0.15 - 0.25j
    mu = 0.21 + 0.17j
    direction = -0.08 + 0.13j

    def loss(value):
        mapped = points + value * np.conjugate(points)
        px, py = 1.0 + value, 1j * (1.0 - value)
        return float(
            np.real(
                np.sum(np.conjugate(cotangent) * mapped)
                + np.conjugate(gp) * px
                + np.conjugate(gq) * py
            )
        )

    eps = 1e-7
    finite_difference = (loss(mu + eps * direction) - loss(mu - eps * direction)) / (2.0 * eps)
    gradient = torus_affine_map_vjp(
        points, cotangent, grad_period_x=gp, grad_period_y=gq
    )
    np.testing.assert_allclose(
        finite_difference,
        np.real(np.conjugate(gradient) * direction),
        rtol=1e-7,
        atol=1e-9,
    )
