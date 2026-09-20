"""Reflection-compatible rectangle extensions for periodic Beurling solves.

The extension is a numerical boundary model, not an assertion that every
rectangle Beltrami problem has a reflection-symmetric solution.  A coefficient
on a fundamental rectangle is copied to the three reflected quadrants.  One
reflection conjugates the Beltrami coefficient; two reflections return the
original coefficient.  Boundary traces must themselves satisfy the
corresponding real-axis compatibility condition.
"""

from __future__ import annotations

import numpy as np


def reflection_extend_mu(mu: np.ndarray) -> np.ndarray:
    """Return a ``2*ny`` by ``2*nx`` conjugate-reflection extension.

    The input samples the half-open rectangle ``[0,1)^2``.  The extended
    periodic cell uses the same spacing on ``[0,2)^2``; index ``-i`` is the
    reflected coordinate and a single reflection applies complex conjugation.
    This construction is exact on the sampled grid, including both axis
    symmetry identities, and does not invoke a periodic convolution itself.
    """

    values = np.asarray(mu, dtype=np.complex128)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("mu must be a two-dimensional array with both axes >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("mu must be finite")
    # The half-open grid represents both reflection axes (0 and 1) by the
    # duplicated index 0 in the doubled periodic cell; the last sample is an
    # interior point, not the x=1/y=1 boundary trace.
    boundary = np.concatenate((values[0], values[:, 0]))
    if np.max(np.abs(boundary.imag)) > 1e-12:
        raise ValueError("reflection-compatible rectangle traces must be real")
    ny, nx = values.shape
    ii = np.arange(2 * nx, dtype=np.int64)
    jj = np.arange(2 * ny, dtype=np.int64)
    base_i = np.minimum(ii % (2 * nx), (-ii) % (2 * nx)) % nx
    base_j = np.minimum(jj % (2 * ny), (-jj) % (2 * ny)) % ny
    reflected_i = ii >= nx
    reflected_j = jj >= ny
    extension = values[np.ix_(base_j, base_i)].copy()
    conjugate_mask = reflected_i[None, :] ^ reflected_j[:, None]
    extension[conjugate_mask] = np.conjugate(extension[conjugate_mask])
    return np.ascontiguousarray(extension)


__all__ = ["reflection_extend_mu"]
