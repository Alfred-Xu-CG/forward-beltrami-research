import numpy as np

from qcopt.forward.beurling import periodic_beltrami_map_from_h
from qcopt.forward.beurling_gmres import periodic_beltrami_gmres
from qcopt.forward.beurling_implicit import periodic_beltrami_implicit_vjp, periodic_beltrami_map_h_vjp


def test_periodic_beurling_implicit_vjp_matches_finite_difference():
    nx, ny = 20, 16
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mu = 0.22 * np.exp(2j * np.pi * (2.0 * xx - yy))
    rng = np.random.default_rng(7)
    cotangent = rng.normal(size=mu.shape) + 1j * rng.normal(size=mu.shape)
    h = periodic_beltrami_gmres(mu, rtol=1e-12, maxiter=100).h
    gradient, residual, info = periodic_beltrami_implicit_vjp(
        mu, h, cotangent, rtol=1e-12, maxiter=100
    )
    assert info == 0
    assert residual < 1e-10

    def loss(coefficient):
        trial = periodic_beltrami_gmres(coefficient, rtol=1e-12, maxiter=100).h
        return float(np.real(np.vdot(cotangent, trial)))

    direction = rng.normal(size=mu.shape) + 1j * rng.normal(size=mu.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-5
    finite_difference = (loss(mu + eps * direction) - loss(mu - eps * direction)) / (2.0 * eps)
    predicted = float(np.real(np.vdot(gradient, direction)))
    assert abs(finite_difference - predicted) < 2e-6


def test_zero_mode_safe_map_h_vjp_matches_finite_difference():
    rng = np.random.default_rng(7)
    h = 0.15 + 0.05j + 0.08 * (rng.normal(size=(16, 20)) + 1j * rng.normal(size=(16, 20)))
    cotangent = rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape)
    analytic = periodic_beltrami_map_h_vjp(h, cotangent)
    direction = rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = periodic_beltrami_map_from_h(h + eps * direction)[0]
    minus = periodic_beltrami_map_from_h(h - eps * direction)[0]
    directional = float(np.real(np.sum(np.conjugate(cotangent) * (plus - minus) / (2.0 * eps))))
    predicted = float(np.real(np.sum(np.conjugate(analytic) * direction)))
    assert abs(directional - predicted) < 2e-8
