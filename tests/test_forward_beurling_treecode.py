import numpy as np

from qcopt.forward.beurling_direct import direct_beurling_apply
from qcopt.forward.beurling_treecode import treecode_beurling_apply
from qcopt.forward.beurling_treecode import treecode_beurling_values_vjp


def test_treecode_matches_direct_for_disjoint_scattered_sets():
    rng = np.random.default_rng(12)
    source = 0.08 + 0.35 * rng.random(128) + 1j * (0.1 + 0.8 * rng.random(128))
    target = 0.62 + 0.3 * rng.random(96) + 1j * (0.1 + 0.8 * rng.random(96))
    values = rng.normal(size=source.size) + 1j * rng.normal(size=source.size)
    weights = 0.2 + rng.random(source.size)
    reference = direct_beurling_apply(source, values, weights, target_points=target, block_size=64)
    approximation = treecode_beurling_apply(source, values, weights, target_points=target, theta=0.25, order=10, max_leaf=12)
    assert np.linalg.norm(approximation - reference) / np.linalg.norm(reference) < 2e-6


def test_treecode_omits_coincident_principal_value_terms():
    points = np.array([0.1 + 0.2j, 0.7 + 0.4j, 0.3 + 0.8j])
    values = np.array([1.0 + 0.2j, -0.3 + 0.5j, 0.4 - 0.1j])
    weights = np.ones(3)
    output = treecode_beurling_apply(points, values, weights, theta=0.3, order=6, max_leaf=1)
    assert np.all(np.isfinite(output))
    assert np.allclose(output[0], -1.0 / np.pi * np.sum(values[1:] / (points[0] - points[1:]) ** 2))


def test_treecode_values_vjp_matches_direct_adjoint_directional_derivative():
    rng = np.random.default_rng(13)
    source = 0.08 + 0.3 * rng.random(48) + 1j * (0.1 + 0.8 * rng.random(48))
    target = 0.62 + 0.28 * rng.random(40) + 1j * (0.1 + 0.8 * rng.random(40))
    values = rng.normal(size=source.size) + 1j * rng.normal(size=source.size)
    weights = 0.4 + rng.random(source.size)
    cotangent = rng.normal(size=target.size) + 1j * rng.normal(size=target.size)
    gradient = treecode_beurling_values_vjp(source, values, weights, cotangent, target_points=target, theta=0.22, order=10, max_leaf=8)
    direction = rng.normal(size=source.size) + 1j * rng.normal(size=source.size)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = treecode_beurling_apply(source, values + eps * direction, weights, target_points=target, theta=0.22, order=10, max_leaf=8)
    minus = treecode_beurling_apply(source, values - eps * direction, weights, target_points=target, theta=0.22, order=10, max_leaf=8)
    directional = float(np.real(np.vdot(cotangent, (plus - minus) / (2.0 * eps))))
    predicted = float(np.real(np.vdot(gradient, direction)))
    assert abs(directional - predicted) < 2e-8
