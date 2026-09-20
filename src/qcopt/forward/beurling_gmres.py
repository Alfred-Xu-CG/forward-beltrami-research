"""Matrix-free periodic Beltrami solve with a GMRES reference backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .beurling import periodic_beurling_apply


def periodic_neumann_initial_guess(mu: np.ndarray, order: int = 2) -> np.ndarray:
    """Build a truncated Neumann warm start ``h <- mu(1+B h)``.

    The returned field is only an initializer; the GMRES residual certificate
    remains authoritative. Increasing ``order`` costs one FFT-based operator
    application per term and is useful as a transparent learned-preconditioner
    baseline.
    """

    coefficients = np.asarray(mu, dtype=np.complex128)
    if coefficients.ndim != 2 or min(coefficients.shape) < 2:
        raise ValueError("mu must be a two-dimensional grid")
    if np.max(np.abs(coefficients)) >= 1.0 or not np.all(np.isfinite(coefficients)):
        raise ValueError("mu must be finite and lie inside the unit disk")
    if not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    guess = coefficients.copy()
    for _ in range(int(order)):
        guess = coefficients * (1.0 + periodic_beurling_apply(guess))
    return np.ascontiguousarray(guess)


@dataclass(frozen=True)
class PeriodicAndersonResult:
    h: np.ndarray
    fixed_point_residual: float
    iterations: int
    converged: bool


def periodic_beltrami_anderson(
    mu: np.ndarray,
    *,
    depth: int = 5,
    rtol: float = 1e-10,
    maxiter: int = 200,
    damping: float = 1.0,
) -> PeriodicAndersonResult:
    """Solve ``h=mu*(1+B h)`` as a short-memory Anderson/DEQ iteration."""

    coefficients = np.asarray(mu, dtype=np.complex128)
    if coefficients.ndim != 2 or min(coefficients.shape) < 2 or not np.all(np.isfinite(coefficients)) or np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must be a finite two-dimensional field inside the unit disk")
    if not isinstance(depth, (int, np.integer)) or depth < 1 or rtol <= 0.0 or maxiter < 1 or not 0.0 < damping <= 1.0:
        raise ValueError("depth/maxiter must be positive, rtol positive, and damping in (0,1]")
    history_f: list[np.ndarray] = []
    history_r: list[np.ndarray] = []
    h = np.zeros_like(coefficients)
    residual = float("inf")
    for iteration in range(1, maxiter + 1):
        fixed = coefficients * (1.0 + periodic_beurling_apply(h))
        residual_field = fixed - h
        residual = float(np.max(np.abs(residual_field)))
        if residual <= rtol:
            return PeriodicAndersonResult(np.ascontiguousarray(h), residual, iteration - 1, True)
        history_f.append(fixed.ravel().copy())
        history_r.append(residual_field.ravel().copy())
        count = min(int(depth), len(history_f))
        f_stack = np.column_stack(history_f[-count:])
        r_stack = np.column_stack(history_r[-count:])
        augmented = np.vstack((r_stack, np.ones((1, count), dtype=np.complex128)))
        rhs = np.zeros(augmented.shape[0], dtype=np.complex128)
        rhs[-1] = 1.0
        mix, *_ = np.linalg.lstsq(augmented, rhs, rcond=None)
        accelerated = (f_stack @ mix).reshape(coefficients.shape)
        h = np.ascontiguousarray((1.0 - damping) * h + damping * accelerated)
    return PeriodicAndersonResult(np.ascontiguousarray(h), residual, maxiter, False)


def periodic_polynomial_initial_guess(
    mu: np.ndarray, coefficients: tuple[complex, complex] = (1.0 + 0.0j, 1.0 + 0.0j)
) -> np.ndarray:
    """Apply a learned two-feature initializer ``c0*mu+c1*mu*(B mu)``."""

    coefficients_array = np.asarray(mu, dtype=np.complex128)
    if coefficients_array.ndim != 2 or min(coefficients_array.shape) < 2:
        raise ValueError("mu must be a two-dimensional grid")
    if np.max(np.abs(coefficients_array)) >= 1.0 or not np.all(np.isfinite(coefficients_array)):
        raise ValueError("mu must be finite and lie inside the unit disk")
    c0, c1 = (complex(coefficients[0]), complex(coefficients[1]))
    if not np.isfinite(c0.real + c0.imag + c1.real + c1.imag):
        raise ValueError("initializer coefficients must be finite")
    bmu = periodic_beurling_apply(coefficients_array)
    return np.ascontiguousarray(c0 * coefficients_array + c1 * coefficients_array * bmu)


@dataclass(frozen=True)
class PeriodicGMRESResult:
    h: np.ndarray
    operator_residual: float
    iterations: int
    converged: bool
    info: int


def periodic_beltrami_gmres(
    mu: np.ndarray,
    *,
    rtol: float = 1e-10,
    maxiter: int = 200,
    restart: int | None = None,
    initial_guess: np.ndarray | None = None,
    preconditioner_order: int | None = None,
) -> PeriodicGMRESResult:
    """Solve ``(I - diag(mu) B)h = mu`` without assembling a dense matrix.

    ``preconditioner_order`` enables a transparent right-side truncated
    Neumann preconditioner ``sum_{k=0}^p (diag(mu)B)^k``. It is an actual
    Krylov preconditioner, not merely a warm start; the certified residual and
    root remain those of the original operator.
    """

    coefficients = np.asarray(mu, dtype=np.complex128)
    if coefficients.ndim != 2 or min(coefficients.shape) < 2:
        raise ValueError("mu must be a two-dimensional grid")
    if not np.all(np.isfinite(coefficients)) or np.max(np.abs(coefficients)) >= 1.0:
        raise ValueError("mu must be finite and lie inside the unit disk")
    if rtol <= 0.0 or maxiter < 1:
        raise ValueError("rtol and maxiter must be positive")
    if preconditioner_order is not None and (
        not isinstance(preconditioner_order, (int, np.integer)) or preconditioner_order < 0
    ):
        raise ValueError("preconditioner_order must be a nonnegative integer or None")
    shape = coefficients.shape
    size = coefficients.size
    right_hand_side = coefficients.ravel()
    if initial_guess is None:
        guess = None
    else:
        guess_values = np.asarray(initial_guess, dtype=np.complex128)
        if guess_values.shape != shape or not np.all(np.isfinite(guess_values)):
            raise ValueError("initial_guess must be finite and match mu shape")
        guess = guess_values.ravel()

    def apply(vector: np.ndarray) -> np.ndarray:
        field = vector.reshape(shape)
        return (field - coefficients * periodic_beurling_apply(field)).ravel()

    operator = LinearOperator((size, size), matvec=apply, dtype=np.complex128)
    preconditioner = None
    if preconditioner_order is not None and int(preconditioner_order) > 0:
        order = int(preconditioner_order)

        def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
            term = vector.reshape(shape)
            result = term.copy()
            for _ in range(order):
                term = coefficients * periodic_beurling_apply(term)
                result = result + term
            return result.ravel()

        preconditioner = LinearOperator((size, size), matvec=apply_preconditioner, dtype=np.complex128)
    callback_values: list[float] = []
    solution, info = gmres(
        operator,
        right_hand_side,
        x0=guess,
        rtol=rtol,
        atol=0.0,
        restart=restart,
        M=preconditioner,
        maxiter=maxiter,
        callback=lambda value: callback_values.append(float(np.abs(value))),
        callback_type="pr_norm",
    )
    residual = float(np.max(np.abs(apply(solution) - right_hand_side)))
    return PeriodicGMRESResult(
        solution.reshape(shape),
        residual,
        len(callback_values),
        bool(info == 0),
        int(info),
    )
