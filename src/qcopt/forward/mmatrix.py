"""Beltrami conductivity tensors and a simple monotone-stencil diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import lsq_linear
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DirectionalConductances:
    """Weights for axis and diagonal second differences."""

    axis_x: float
    axis_y: float
    diagonal_plus: float
    diagonal_minus: float
    monotone_possible: bool


@dataclass(frozen=True)
class SpectralConductances:
    """Positive directional decomposition in arbitrary (eigenvector) directions."""

    values: NDArray[np.float64]
    directions: NDArray[np.float64]


@dataclass(frozen=True)
class DictionaryConductances:
    """Nonnegative fit over a finite stencil-direction dictionary."""

    values: NDArray[np.float64]
    directions: NDArray[np.float64]
    residual: float


@dataclass(frozen=True)
class BatchDictionaryConductances:
    """Vectorized nonnegative cone fits for a batch of SPD tensors."""

    values: NDArray[np.float64]
    directions: NDArray[np.float64]
    residual: NDArray[np.float64]


def integer_wide_stencil_directions(max_step: int = 4) -> NDArray[np.float64]:
    """Return unoriented directions available as integer grid wide stencils.

    A direction ``(p,q)`` represents the centered second difference between
    vertices separated by the integer vector.  The sign is irrelevant for
    ``d d^T``; only one representative of each unoriented pair is retained.
    """

    if not isinstance(max_step, (int, np.integer)) or max_step < 1:
        raise ValueError("max_step must be a positive integer")
    vectors: list[tuple[int, int]] = []
    for p in range(0, int(max_step) + 1):
        for q in range(-int(max_step), int(max_step) + 1):
            if p == 0 and q <= 0:
                continue
            if p == 0 and q == 0:
                continue
            if np.gcd(p, abs(q)) != 1:
                continue
            vectors.append((p, q))
    values = np.asarray(vectors, dtype=np.float64)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values)


def batch_positive_directional_conductances(
    tensors: FloatArray, directions: FloatArray, *, tolerance: float = 1e-12
) -> BatchDictionaryConductances:
    """Fit many SPD tensors with nonnegative weights over one fixed direction set.

    In two dimensions an exact symmetric tensor has three independent entries,
    so a nonnegative least-squares optimum can be represented by at most three
    active directions. Enumerating those small active sets and vectorizing over
    the batch avoids one SciPy optimizer call per grid node in variable-tensor
    audits and in mesh-native decoder prototypes.
    """

    matrix = np.asarray(tensors, dtype=np.float64)
    vectors = np.asarray(directions, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-2:] != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("tensors must have shape (...,2,2) and be finite")
    if vectors.ndim != 2 or vectors.shape[1] != 2 or vectors.shape[0] < 1 or not np.all(np.isfinite(vectors)):
        raise ValueError("directions must have shape (m,2) and be finite")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-14):
        raise ValueError("directions must be nonzero")
    normalized = np.ascontiguousarray(vectors / norms[:, None])
    dictionary = np.vstack(
        (normalized[:, 0] ** 2, normalized[:, 0] * normalized[:, 1], normalized[:, 1] ** 2)
    )
    target = np.stack((matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 1]), axis=-1)
    shape = target.shape[:-1]
    best_values = np.zeros(shape + (vectors.shape[0],), dtype=np.float64)
    best_residual = np.full(shape, np.inf, dtype=np.float64)
    for active_count in range(1, min(3, vectors.shape[0]) + 1):
        for active in combinations(range(vectors.shape[0]), active_count):
            active_dictionary = dictionary[:, active]
            pseudoinverse = np.linalg.pinv(active_dictionary)
            candidate = np.einsum("...c,kc->...k", target, pseudoinverse)
            feasible = np.all(candidate >= -tolerance, axis=-1)
            reconstructed = np.einsum("...k,ck->...c", candidate, active_dictionary)
            residual = np.linalg.norm(reconstructed - target, axis=-1)
            update = feasible & (residual < best_residual)
            candidate_full = np.zeros_like(best_values)
            candidate_full[..., active] = candidate
            best_values = np.where(update[..., None], candidate_full, best_values)
            best_residual = np.where(update, residual, best_residual)
    # Keep an explicit finite fallback for tensors outside the positive cone.
    # The residual records the approximation; callers must not interpret this
    # fallback as an exact monotone representation.
    unresolved = ~np.isfinite(best_residual)
    if np.any(unresolved):
        pseudoinverse = np.linalg.pinv(dictionary)
        clipped = np.maximum(np.einsum("...c,kc->...k", target, pseudoinverse), 0.0)
        reconstructed = np.einsum("...k,ck->...c", clipped, dictionary)
        fallback_residual = np.linalg.norm(reconstructed - target, axis=-1)
        best_values = np.where(unresolved[..., None], clipped, best_values)
        best_residual = np.where(unresolved, fallback_residual, best_residual)
    return BatchDictionaryConductances(
        np.ascontiguousarray(best_values), normalized, np.ascontiguousarray(best_residual)
    )


def integer_wide_stencil_conductances(
    tensor: FloatArray, *, max_step: int = 4
) -> DictionaryConductances:
    """Fit an SPD tensor using positive directions realizable on a regular grid."""

    matrix = np.asarray(tensor, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("tensor must be a symmetric (2, 2) array")
    if np.linalg.eigvalsh(matrix).min() <= 0.0:
        raise ValueError("tensor must be SPD")
    directions = integer_wide_stencil_directions(max_step)
    dictionary = np.vstack(
        (
            directions[:, 0] ** 2,
            directions[:, 0] * directions[:, 1],
            directions[:, 1] ** 2,
        )
    )
    target = np.asarray([matrix[0, 0], matrix[0, 1], matrix[1, 1]])
    fit = lsq_linear(
        dictionary,
        target,
        bounds=(0.0, np.inf),
        tol=1e-12,
        lsmr_tol="auto",
        max_iter=max(100, 4 * len(directions)),
    )
    residual = float(np.linalg.norm(dictionary @ fit.x - target))
    return DictionaryConductances(
        np.ascontiguousarray(fit.x), np.ascontiguousarray(directions), residual
    )


def edge_direction_conductances(
    tensor: FloatArray, directions: FloatArray
) -> DictionaryConductances:
    """Fit an SPD tensor by nonnegative conductances on mesh edge directions.

    Unlike the integer-grid helper, ``directions`` may come from an arbitrary
    unstructured one-ring or wide graph stencil.  The result is a cone-fit
    diagnostic: a positive residual means that this local edge-direction cone
    cannot represent the requested anisotropy exactly.
    """

    matrix = np.asarray(tensor, dtype=np.float64)
    vectors = np.asarray(directions, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("tensor must be a symmetric (2,2) array")
    if np.linalg.eigvalsh(matrix).min() <= 0.0:
        raise ValueError("tensor must be SPD")
    if vectors.ndim != 2 or vectors.shape[1] != 2 or vectors.shape[0] < 3 or not np.all(np.isfinite(vectors)):
        raise ValueError("directions must have shape (m,2) with m >= 3")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-14):
        raise ValueError("directions must be nonzero")
    normalized = vectors / norms[:, None]
    dictionary = np.vstack((normalized[:, 0] ** 2, normalized[:, 0] * normalized[:, 1], normalized[:, 1] ** 2))
    target = np.asarray([matrix[0, 0], matrix[0, 1], matrix[1, 1]])
    fit = lsq_linear(dictionary, target, bounds=(0.0, np.inf), tol=1e-12, lsmr_tol="auto", max_iter=max(100, 4 * vectors.shape[0]))
    residual = float(np.linalg.norm(dictionary @ fit.x - target))
    return DictionaryConductances(np.ascontiguousarray(fit.x), np.ascontiguousarray(normalized), residual)


def beltrami_conductivity(mu: complex) -> FloatArray:
    """Return the symmetric SPD tensor associated with a Beltrami coefficient."""

    mu = complex(mu)
    magnitude = abs(mu)
    if not np.isfinite(mu.real) or not np.isfinite(mu.imag) or magnitude >= 1.0:
        raise ValueError("mu must be finite and lie strictly inside the unit disk")
    denominator = 1.0 - magnitude**2
    return np.asarray(
        [
            [(1.0 - 2.0 * mu.real + magnitude**2) / denominator,
             -2.0 * mu.imag / denominator],
            [-2.0 * mu.imag / denominator,
             (1.0 + 2.0 * mu.real + magnitude**2) / denominator],
        ],
        dtype=np.float64,
    )


def directional_conductances(
    tensor: FloatArray, tolerance: float = 1e-12
) -> DirectionalConductances:
    """Test a five-point/diagonal nonnegative directional decomposition.

    For ``A=[[a,b],[b,c]]``, the split uses diagonal weights ``|b|`` and leaves
    axis weights ``a-|b|`` and ``c-|b|``. SPD alone does not imply these
    remaining weights are nonnegative.
    """

    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.shape != (2, 2) or not np.allclose(tensor, tensor.T, atol=tolerance):
        raise ValueError("tensor must be a symmetric (2, 2) array")
    a, b, c = float(tensor[0, 0]), float(tensor[0, 1]), float(tensor[1, 1])
    axis_x = a - abs(b)
    axis_y = c - abs(b)
    diagonal_plus = max(b, 0.0)
    diagonal_minus = max(-b, 0.0)
    return DirectionalConductances(
        axis_x,
        axis_y,
        diagonal_plus,
        diagonal_minus,
        axis_x >= -tolerance and axis_y >= -tolerance,
    )


def spectral_conductances(tensor: FloatArray) -> SpectralConductances:
    """Decompose an SPD tensor into positive eigen-direction conductances.

    This always succeeds for SPD tensors, but the eigen-directions generally
    are not edges of a fixed rectangular/triangular mesh. It therefore
    separates continuous ellipticity from a mesh-native M-matrix guarantee.
    """

    matrix = np.asarray(tensor, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("tensor must be a symmetric (2, 2) array")
    values, vectors = np.linalg.eigh(matrix)
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("tensor must be SPD")
    return SpectralConductances(
        np.ascontiguousarray(values), np.ascontiguousarray(vectors.T)
    )


def dictionary_conductances(
    tensor: FloatArray, *, n_directions: int = 16
) -> DictionaryConductances:
    """Fit ``A=sum w_k d_k d_k^T`` with nonnegative finite-stencil weights."""

    matrix = np.asarray(tensor, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("tensor must be a symmetric (2, 2) array")
    if np.linalg.eigvalsh(matrix).min() <= 0.0:
        raise ValueError("tensor must be SPD")
    if not isinstance(n_directions, (int, np.integer)) or n_directions < 3:
        raise ValueError("n_directions must be an integer at least three")
    angles = np.arange(n_directions, dtype=np.float64) * np.pi / n_directions
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    dictionary = np.vstack(
        (
            directions[:, 0] ** 2,
            directions[:, 0] * directions[:, 1],
            directions[:, 1] ** 2,
        )
    )
    target = np.asarray([matrix[0, 0], matrix[0, 1], matrix[1, 1]])
    fit = lsq_linear(
        dictionary,
        target,
        bounds=(0.0, np.inf),
        tol=1e-12,
        lsmr_tol="auto",
        max_iter=max(100, 4 * n_directions),
    )
    values = fit.x
    residual = float(np.linalg.norm(dictionary @ values - target))
    return DictionaryConductances(
        np.ascontiguousarray(values), np.ascontiguousarray(directions), float(residual)
    )


def conditioning_safe_dictionary_conductances(
    tensor: FloatArray, *, residual_tolerance: float = 1e-8, max_directions: int = 32
) -> DictionaryConductances:
    """Choose the smallest well-conditioned angular dictionary meeting a residual target.

    Overcomplete nonnegative least-squares dictionaries can become numerically
    ill-conditioned (the 128-direction audit is a concrete example).  This
    helper therefore grows a modest sequence of angular dictionaries and stops
    at the first residual target, avoiding the false precision of a huge
    nearly-dependent stencil.
    """

    if residual_tolerance <= 0.0 or max_directions < 3:
        raise ValueError("residual_tolerance must be positive and max_directions >= 3")
    candidates = [n for n in (8, 12, 16, 24, 32) if n <= max_directions]
    if not candidates or candidates[-1] != max_directions:
        candidates.append(int(max_directions))
    best = None
    for n in candidates:
        fit = dictionary_conductances(tensor, n_directions=n)
        if best is None or fit.residual < best.residual:
            best = fit
        if fit.residual <= residual_tolerance:
            return fit
    assert best is not None
    return best
