import numpy as np

from qcopt.forward.p1_symbols import (
    p1_beurling_block_symbol,
    p1_block_symbol_grid,
    p1_face_derivative_symbols,
)
from qcopt.forward.p1_symbols_torch import p1_block_symbol_grid_torch


def test_p1_constant_mode_is_a_zero_derivative_mode():
    a, b = p1_face_derivative_symbols(0, 0, 16, 12)
    np.testing.assert_allclose(a, 0.0, atol=1e-14)
    np.testing.assert_allclose(b, 0.0, atol=1e-14)
    _, _, p, s = p1_beurling_block_symbol(0, 0, 16, 12)
    np.testing.assert_allclose(p, 0.0)
    np.testing.assert_allclose(s, 0.0)


def test_p1_weighted_pseudoinverse_projects_onto_derivative_range():
    nx, ny = 20, 16
    symbols, projectors = p1_block_symbol_grid(nx, ny, face_weights=np.array([1.0, 2.0]))
    for ky in range(ny):
        for kx in range(nx):
            projector = projectors[ky, kx]
            np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)
            _, b, p, s = p1_beurling_block_symbol(kx, ky, nx, ny, face_weights=np.array([1.0, 2.0]))
            np.testing.assert_allclose(s, s @ projector, atol=1e-12)
            np.testing.assert_allclose(b[:, None] @ p, projector, atol=1e-12)
    assert np.all(np.isfinite(symbols))


def test_p1_symbol_reproduces_beltrami_ratio_on_compatible_face_data():
    a, b, p, s = p1_beurling_block_symbol(3, -2, 64, 48)
    q = np.asarray([0.7 - 0.2j])
    face_dbar = b * q[0]
    recovered = p @ face_dbar
    np.testing.assert_allclose(recovered, q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s @ face_dbar, a * q[0], rtol=1e-12, atol=1e-12)


def test_torch_batched_p1_symbol_matches_numpy_grid():
    numpy_symbols, numpy_projectors = p1_block_symbol_grid(12, 10, face_weights=np.array([1.0, 2.0]))
    torch_symbols, torch_projectors = p1_block_symbol_grid_torch(12, 10, face_weights=(1.0, 2.0))
    np.testing.assert_allclose(torch_symbols.detach().numpy(), numpy_symbols, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(torch_projectors.detach().numpy(), numpy_projectors, rtol=1e-12, atol=1e-12)
