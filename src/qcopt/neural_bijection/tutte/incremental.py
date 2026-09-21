"""Algebraically exact incremental diagnostics for directed Tutte systems.

This module is deliberately small.  It does not replace the Route-I decoder
and it does not certify topology on its own.  It exposes the identical reduced
linear system needed to compare a cold Krylov solve, an old-solution warm
start, the exact correction equation, and a row-local Woodbury update.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from ...forward.tutte_directed_implicit import DirectedTutteSystem


@dataclass(frozen=True)
class IterativeSolveResult:
    solution: np.ndarray
    iterations: tuple[int, int]
    relative_residuals: tuple[float, float]


@dataclass(frozen=True)
class WoodburyUpdateResult:
    solution: np.ndarray
    updated_rows: int
    schur_condition: float
    relative_residual: float


def assemble_directed_system(
    system: DirectedTutteSystem,
    logits: np.ndarray,
    boundary: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Return ``A``, ``B`` and supported row probabilities for ``AY=B``.

    The neighbor slots and boundary order are exactly those of
    :class:`DirectedTutteSystem`.  Unsupported probability slots are zero.
    This is an assembly utility, not a boundary-convexity or P1-map checker.
    """

    values = np.asarray(logits, dtype=np.float64)
    points = np.asarray(boundary, dtype=np.float64)
    expected_logits = (system.n_rows, system.max_degree)
    expected_boundary = (len(system.loop), 2)
    if values.shape != expected_logits:
        raise ValueError(f"logits must have shape {expected_logits}")
    if points.shape != expected_boundary:
        raise ValueError(f"boundary must have shape {expected_boundary}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(points)):
        raise ValueError("logits and boundary must be finite")
    if system.n_rows == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64), np.empty((0, 2)), np.empty_like(values)
    probabilities = system._probabilities(values)
    matrix, coupling = system._assemble(probabilities)
    right_hand_side = np.asarray(coupling @ points, dtype=np.float64)
    if not np.all(np.isfinite(right_hand_side)):
        raise ValueError("assembled right-hand side must be finite")
    return matrix.tocsr(), right_hand_side, probabilities


def correction_right_hand_side(
    old_matrix: sparse.spmatrix,
    old_rhs: np.ndarray,
    old_solution: np.ndarray,
    new_matrix: sparse.spmatrix,
    new_rhs: np.ndarray,
) -> np.ndarray:
    """Form the exact shifted equation ``A' delta = (B'-B)-(A'-A)Y``."""

    old_a, new_a = sparse.csr_matrix(old_matrix), sparse.csr_matrix(new_matrix)
    old_b = np.asarray(old_rhs, dtype=np.float64)
    new_b = np.asarray(new_rhs, dtype=np.float64)
    old_y = np.asarray(old_solution, dtype=np.float64)
    if old_a.shape != new_a.shape or old_a.shape[0] != old_a.shape[1]:
        raise ValueError("old and new matrices must be square with the same shape")
    if old_b.shape != new_b.shape or old_y.shape != old_b.shape or old_b.shape != (old_a.shape[0], 2):
        raise ValueError("right-hand sides and old solution must share shape (n, 2)")
    if not all(np.all(np.isfinite(value)) for value in (old_b, new_b, old_y)):
        raise ValueError("right-hand sides and old solution must be finite")
    result = (new_b - old_b) - (new_a - old_a) @ old_y
    if not np.all(np.isfinite(result)):
        raise ValueError("correction right-hand side must be finite")
    return np.asarray(result)


def bicgstab_two_rhs(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    atol: float = 1.0e-12,
    max_iterations: int | None = None,
) -> IterativeSolveResult:
    """Solve two coordinate columns with one absolute residual contract.

    SciPy executes the two columns separately, so the reported iteration tuple
    contains one count per coordinate.  ``rtol=0`` keeps the same externally
    declared absolute threshold when the correction RHS is much smaller than
    the full RHS; this avoids crediting the correction solve with a looser
    stopping rule.
    """

    a = sparse.csr_matrix(matrix, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    guess = None if x0 is None else np.asarray(x0, dtype=np.float64)
    if a.shape[0] != a.shape[1] or b.shape != (a.shape[0], 2):
        raise ValueError("matrix must be square and rhs must have shape (n, 2)")
    if guess is not None and guess.shape != b.shape:
        raise ValueError("x0 must have the same shape as rhs")
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be finite and strictly positive")
    if not np.all(np.isfinite(b)) or (guess is not None and not np.all(np.isfinite(guess))):
        raise ValueError("rhs and x0 must be finite")
    budget = max_iterations if max_iterations is not None else max(20, 10 * max(1, a.shape[0]))
    if budget < 1:
        raise ValueError("max_iterations must be positive")

    parameters = inspect.signature(sparse_linalg.bicgstab).parameters
    if "rtol" in parameters:
        relative_keyword = "rtol"
    elif "tol" in parameters:
        # SciPy < 1.12 used ``tol`` for the same relative term.  Setting it to
        # zero preserves the declared absolute-only stopping contract.
        relative_keyword = "tol"
    else:  # pragma: no cover - protects against an unrecognized future API
        raise RuntimeError("unrecognized scipy.sparse.linalg.bicgstab tolerance API")

    columns: list[np.ndarray] = []
    counts: list[int] = []
    residuals: list[float] = []
    for coordinate in range(2):
        count = 0

        def callback(_iterate: np.ndarray) -> None:
            nonlocal count
            count += 1

        keyword_arguments = {
            "x0": None if guess is None else guess[:, coordinate],
            relative_keyword: 0.0,
            "atol": atol,
            "maxiter": budget,
            "callback": callback,
        }
        solution, info = sparse_linalg.bicgstab(
            a, b[:, coordinate], **keyword_arguments
        )
        residual = float(np.linalg.norm(b[:, coordinate] - a @ solution))
        if info != 0 or not np.all(np.isfinite(solution)) or not np.isfinite(residual) or residual > 1.01 * atol:
            raise RuntimeError(
                f"BiCGStab coordinate {coordinate} failed the true-residual contract "
                f"(info={info}, iterations={count}, residual={residual:.6e}, atol={atol:.6e})"
            )
        denominator = max(float(np.linalg.norm(b[:, coordinate])), np.finfo(np.float64).tiny)
        columns.append(solution)
        counts.append(count)
        residuals.append(residual / denominator)
    return IterativeSolveResult(
        solution=np.column_stack(columns),
        iterations=(counts[0], counts[1]),
        relative_residuals=(residuals[0], residuals[1]),
    )


def woodbury_row_update(
    old_matrix: sparse.spmatrix,
    new_matrix: sparse.spmatrix,
    new_rhs: np.ndarray,
    changed_rows: np.ndarray,
    *,
    old_factor=None,
) -> WoodburyUpdateResult:
    """Solve a row-local matrix update using one factorization of ``old_matrix``.

    If rows ``R`` contain every nonzero row of ``A'-A``, then
    ``A'=A+U V^T`` with columns of ``U`` equal to coordinate vectors for
    ``R``.  The returned solution uses the exact Woodbury identity.  A supplied
    ``old_factor`` is reused; otherwise this function factors ``A`` and its
    timing therefore includes setup.  A global update is algebraically allowed
    but builds a dense Schur system of order ``len(R)`` and therefore carries
    no automatic speed claim.
    """

    old_a = sparse.csc_matrix(old_matrix, dtype=np.float64)
    new_a = sparse.csr_matrix(new_matrix, dtype=np.float64)
    b = np.asarray(new_rhs, dtype=np.float64)
    rows = np.asarray(changed_rows, dtype=np.int64)
    n = old_a.shape[0]
    if old_a.shape != (n, n) or new_a.shape != (n, n) or b.shape != (n, 2):
        raise ValueError("matrices must be matching square systems and new_rhs must have shape (n, 2)")
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= n):
        raise ValueError("changed_rows must contain unique valid row indices")
    if not np.all(np.isfinite(b)):
        raise ValueError("new_rhs must be finite")
    rows = np.sort(rows)
    delta = (new_a - old_a.tocsr()).tocsr()
    keep = np.ones(n, dtype=bool)
    keep[rows] = False
    if delta[keep].nnz:
        outside = delta[keep].data
        if np.any(outside != 0.0):
            raise ValueError("matrix update has nonzero entries outside declared rows")

    if old_factor is None:
        try:
            factor = sparse_linalg.splu(old_a)
        except RuntimeError as error:
            raise ValueError("old matrix factorization failed") from error
    elif not hasattr(old_factor, "solve"):
        raise TypeError("old_factor must expose a solve method")
    else:
        factor = old_factor
    if len(rows) == 0:
        solution = factor.solve(b)
        schur_condition = 1.0
    else:
        u = np.zeros((n, len(rows)), dtype=np.float64)
        u[rows, np.arange(len(rows))] = 1.0
        # One factor solve call with two physical RHS columns plus the low-rank
        # basis.  The dense Schur solve is k-by-k, k=len(changed_rows).
        solved = factor.solve(np.column_stack((b, u)))
        base, inverse_u = solved[:, :2], solved[:, 2:]
        v_transpose = delta[rows].toarray()
        schur = np.eye(len(rows)) + v_transpose @ inverse_u
        schur_condition = float(np.linalg.cond(schur))
        if not np.isfinite(schur_condition):
            raise ValueError("Woodbury Schur system is singular or nonfinite")
        try:
            correction = np.linalg.solve(schur, v_transpose @ base)
        except np.linalg.LinAlgError as error:
            raise ValueError("Woodbury Schur solve failed") from error
        solution = base - inverse_u @ correction
    residual = np.asarray(b - new_a @ solution)
    residual_norm = float(np.linalg.norm(residual))
    denominator = max(float(np.linalg.norm(b)), np.finfo(np.float64).tiny)
    relative = residual_norm / denominator
    scale = float(sparse_linalg.norm(new_a) * np.linalg.norm(solution) + np.linalg.norm(b))
    tolerance = 2048.0 * np.finfo(np.float64).eps * max(1.0, scale)
    if not np.all(np.isfinite(solution)) or not np.isfinite(relative) or residual_norm > tolerance:
        raise RuntimeError(
            f"Woodbury result failed represented residual check ({residual_norm:.6e} > {tolerance:.6e})"
        )
    return WoodburyUpdateResult(solution, len(rows), schur_condition, relative)
