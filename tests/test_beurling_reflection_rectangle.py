import numpy as np
import pytest

from qcopt.forward.beurling_reflection import reflection_extend_mu


def test_reflection_extension_has_conjugate_quadrant_symmetry():
    rng = np.random.default_rng(4)
    base = 0.2 * (rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7)))
    base[0, :] = base[-1, :] = base[:, 0] = base[:, -1] = np.real(base[0, :].mean())
    extension = reflection_extend_mu(base)
    ny, nx = base.shape
    assert extension.shape == (2 * ny, 2 * nx)
    for j in range(2 * ny):
        for i in range(2 * nx):
            ii = (-i) % (2 * nx)
            jj = (-j) % (2 * ny)
            assert np.allclose(extension[j, ii], np.conjugate(extension[j, i]))
            assert np.allclose(extension[jj, i], np.conjugate(extension[j, i]))
            assert np.allclose(extension[jj, ii], extension[j, i])


def test_reflection_extension_preserves_zero_boundary_imaginary_part():
    base = np.ones((6, 8), dtype=np.complex128) * (0.2 + 0.1j)
    base[0, :] = base[-1, :] = base[:, 0] = base[:, -1] = 0.2
    extension = reflection_extend_mu(base)
    assert np.max(np.abs(extension[0].imag)) == 0.0
    assert np.max(np.abs(extension[:, 0].imag)) == 0.0


def test_reflection_extension_rejects_incompatible_boundary_trace():
    base = np.ones((4, 4), dtype=np.complex128) * (0.1 + 0.2j)
    with pytest.raises(ValueError, match="traces must be real"):
        reflection_extend_mu(base)
