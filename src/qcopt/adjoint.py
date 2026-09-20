"""Exact implicit VJPs for sparse variable-μ QC reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .lbs import beltrami_tensors, validate_mu
from .mesh import TriMesh
from .solver import ReducedSolveState, SolveResult, SparseSolveState

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class AdjointResult:
    gradient: FloatArray
    residual: float


def _adjoint_vector(result: SolveResult, uv_bar: FloatArray) -> tuple[FloatArray, float]:
    uv_bar = np.asarray(uv_bar, dtype=np.float64)
    if uv_bar.shape != result.uv.shape:
        raise ValueError("uv_bar must have the same shape as the solved map")
    if not isinstance(result.state, SparseSolveState):
        raise ValueError("augmented/KKT adjoint requires a SparseSolveState")
    right_hand_side = np.zeros_like(result.state.solution)
    right_hand_side[result.state.coordinate_slice] = uv_bar.T.reshape(-1)
    adjoint = result.state.factor.solve(right_hand_side, trans="T")
    residual = result.state.system.T @ adjoint - right_hand_side
    relative = float(
        np.linalg.norm(residual) / max(np.linalg.norm(right_hand_side), 1.0)
    )
    return adjoint, relative


def beltrami_tensor_derivatives(mu: ComplexArray) -> tuple[FloatArray, FloatArray]:
    """Return analytic derivatives of the LBS tensor with respect to ρ and τ."""

    coefficients = np.asarray(mu, dtype=np.complex128).reshape(-1)
    rho = coefficients.real
    tau = coefficients.imag
    denominator = 1.0 - rho**2 - tau**2
    numerator = beltrami_tensors(coefficients) * denominator[:, None, None]

    numerator_rho = np.zeros_like(numerator)
    numerator_rho[:, 0, 0] = 2.0 * (rho - 1.0)
    numerator_rho[:, 1, 1] = 2.0 * (rho + 1.0)
    numerator_tau = np.zeros_like(numerator)
    numerator_tau[:, 0, 0] = 2.0 * tau
    numerator_tau[:, 0, 1] = -2.0
    numerator_tau[:, 1, 0] = -2.0
    numerator_tau[:, 1, 1] = 2.0 * tau

    derivative_rho = (
        numerator_rho * denominator[:, None, None]
        + 2.0 * rho[:, None, None] * numerator
    ) / denominator[:, None, None] ** 2
    derivative_tau = (
        numerator_tau * denominator[:, None, None]
        + 2.0 * tau[:, None, None] * numerator
    ) / denominator[:, None, None] ** 2
    return derivative_rho, derivative_tau


def _tensor_mu_vjp(
    mesh: TriMesh,
    coefficients: ComplexArray,
    coordinates: FloatArray,
    adjoint_coordinates: FloatArray,
) -> FloatArray:
    """Contract all weighted LSQC/LBS face tensor derivatives in one batch.

    The symmetric 2-by-2 tensor derivatives are expanded componentwise.  This
    avoids materializing an ``(n_faces, 2, 2, 2)`` derivative tensor and also
    gathers all four primal/adjoint scalar fields in one operation.
    """

    n = mesh.n_vertices
    fields = np.column_stack(
        (
            coordinates[:n],
            coordinates[n:],
            adjoint_coordinates[:n],
            adjoint_coordinates[n:],
        )
    )
    face_gradients = np.einsum(
        "fki,fkc->fci", mesh.gradients, fields[mesh.faces], optimize=False
    )
    gradient_u, gradient_v, gradient_pu, gradient_pv = np.moveaxis(
        face_gradients, 1, 0
    )

    rho = coefficients.real
    tau = coefficients.imag
    denominator = 1.0 - rho**2 - tau**2
    denominator_squared = denominator**2
    numerator_00 = (rho - 1.0) ** 2 + tau**2
    numerator_01 = -2.0 * tau
    numerator_11 = (rho + 1.0) ** 2 + tau**2

    twice_rho = 2.0 * rho
    derivative_rho_00 = (
        2.0 * (rho - 1.0) * denominator + twice_rho * numerator_00
    ) / denominator_squared
    derivative_rho_01 = twice_rho * numerator_01 / denominator_squared
    derivative_rho_11 = (
        2.0 * (rho + 1.0) * denominator + twice_rho * numerator_11
    ) / denominator_squared

    twice_tau = 2.0 * tau
    derivative_tau_00 = (
        twice_tau * denominator + twice_tau * numerator_00
    ) / denominator_squared
    derivative_tau_01 = (
        -2.0 * denominator + twice_tau * numerator_01
    ) / denominator_squared
    derivative_tau_11 = (
        twice_tau * denominator + twice_tau * numerator_11
    ) / denominator_squared

    def contract_symmetric(
        left: FloatArray,
        matrix_00: FloatArray,
        matrix_01: FloatArray,
        matrix_11: FloatArray,
        right: FloatArray,
    ) -> FloatArray:
        return left[:, 0] * (
            matrix_00 * right[:, 0] + matrix_01 * right[:, 1]
        ) + left[:, 1] * (
            matrix_01 * right[:, 0] + matrix_11 * right[:, 1]
        )

    contraction_rho = contract_symmetric(
        gradient_pu,
        derivative_rho_00,
        derivative_rho_01,
        derivative_rho_11,
        gradient_u,
    ) + contract_symmetric(
        gradient_pv,
        derivative_rho_00,
        derivative_rho_01,
        derivative_rho_11,
        gradient_v,
    )
    contraction_tau = contract_symmetric(
        gradient_pu,
        derivative_tau_00,
        derivative_tau_01,
        derivative_tau_11,
        gradient_u,
    ) + contract_symmetric(
        gradient_pv,
        derivative_tau_00,
        derivative_tau_01,
        derivative_tau_11,
        gradient_v,
    )
    return -mesh.areas[:, None] * np.column_stack(
        (contraction_rho, contraction_tau)
    )


def lbs_mu_vjp(
    mesh: TriMesh, mu: ComplexArray, result: SolveResult, uv_bar: FloatArray
) -> AdjointResult:
    coefficients = validate_mu(mesh, mu)
    adjoint, residual = _adjoint_vector(result, uv_bar)
    coordinates = result.state.solution[result.state.coordinate_slice]
    adjoint_coordinates = adjoint[result.state.coordinate_slice]
    gradient = _tensor_mu_vjp(
        mesh, coefficients, coordinates, adjoint_coordinates
    )
    return AdjointResult(gradient=gradient, residual=residual)


def lsqc_fast_mu_vjp(
    mesh: TriMesh,
    mu: ComplexArray,
    result: SolveResult,
    uv_bar: FloatArray,
) -> AdjointResult:
    """Exact VJP for :func:`qcopt.lsqc.solve_lsqc_fast`.

    Only the saved free-coordinate factorization is used.  Pinned adjoint
    coordinates are zero because their values are independent of ``mu``.
    """

    coefficients = validate_mu(mesh, mu)
    if not isinstance(result.state, ReducedSolveState):
        raise ValueError("fast LSQC adjoint requires a ReducedSolveState")
    cotangent = np.asarray(uv_bar, dtype=np.float64)
    if cotangent.shape != result.uv.shape:
        raise ValueError("uv_bar must have the same shape as the solved map")
    full_rhs = cotangent.T.reshape(-1)
    free_rhs = full_rhs[result.state.free_indices]
    free_adjoint = result.state.factor.solve(free_rhs, trans="T")
    algebraic_residual = result.state.system.T @ free_adjoint - free_rhs
    relative_residual = float(
        np.linalg.norm(algebraic_residual) / max(np.linalg.norm(free_rhs), 1.0)
    )
    adjoint_coordinates = np.zeros_like(result.state.coordinates)
    adjoint_coordinates[result.state.free_indices] = free_adjoint
    gradient = _weighted_lsqc_mu_contraction(
        mesh, coefficients, result.state.coordinates, adjoint_coordinates
    )
    return AdjointResult(gradient=gradient, residual=relative_residual)


def _lsqc_local_block_and_derivatives(
    mesh: TriMesh, mu: complex, face_index: int, weighted: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    rho, tau = float(mu.real), float(mu.imag)
    gradients = mesh.gradients[face_index]
    base = np.empty((2, 6), dtype=np.float64)
    derivative_rho = np.empty_like(base)
    derivative_tau = np.empty_like(base)
    for local_vertex, (gx, gy) in enumerate(gradients):
        base[:, local_vertex] = (
            (1.0 - rho) * gx - tau * gy,
            -tau * gx + (1.0 + rho) * gy,
        )
        base[:, 3 + local_vertex] = (
            tau * gx - (1.0 + rho) * gy,
            (1.0 - rho) * gx - tau * gy,
        )
        derivative_rho[:, local_vertex] = (-gx, gy)
        derivative_rho[:, 3 + local_vertex] = (-gy, -gx)
        derivative_tau[:, local_vertex] = (-gy, -gx)
        derivative_tau[:, 3 + local_vertex] = (gx, -gy)

    weight = float(np.sqrt(mesh.areas[face_index]))
    if not weighted:
        return weight * base, weight * derivative_rho, weight * derivative_tau
    denominator = 1.0 - rho**2 - tau**2
    qc_weight = weight / np.sqrt(denominator)
    derivative_weight_rho = qc_weight * rho / denominator
    derivative_weight_tau = qc_weight * tau / denominator
    return (
        qc_weight * base,
        qc_weight * derivative_rho + derivative_weight_rho * base,
        qc_weight * derivative_tau + derivative_weight_tau * base,
    )


def _lsqc_residual_and_derivative_actions(
    mesh: TriMesh,
    coefficients: ComplexArray,
    coordinates: FloatArray,
    weighted: bool,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate face residuals and both LSQC derivative actions together.

    The derivative result has shape ``(n_faces, 2, 2)`` ordered as
    ``[face, rho_or_tau, residual_component]``. Computing the residual in this
    same kernel lets the fast weighted-LSQC VJP reuse the already-gathered
    coordinate gradients.
    """

    n = mesh.n_vertices
    u, v = coordinates[:n], coordinates[n:]
    gradient_u = np.einsum(
        "fki,fk->fi", mesh.gradients, u[mesh.faces], optimize=True
    )
    gradient_v = np.einsum(
        "fki,fk->fi", mesh.gradients, v[mesh.faces], optimize=True
    )
    ux, uy = gradient_u[:, 0], gradient_u[:, 1]
    vx, vy = gradient_v[:, 0], gradient_v[:, 1]
    rho, tau = coefficients.real, coefficients.imag

    base_residual = np.column_stack(
        (
            (1.0 - rho) * ux - tau * uy + tau * vx - (1.0 + rho) * vy,
            -tau * ux + (1.0 + rho) * uy + (1.0 - rho) * vx - tau * vy,
        )
    )
    derivative_rho = np.column_stack((-ux - vy, uy - vx))
    derivative_tau = np.column_stack((-uy + vx, -ux - vy))
    base_derivatives = np.stack((derivative_rho, derivative_tau), axis=1)
    area_weight = np.sqrt(mesh.areas)
    if not weighted:
        return (
            area_weight[:, None] * base_residual,
            area_weight[:, None, None] * base_derivatives,
        )

    denominator = 1.0 - rho**2 - tau**2
    weight = area_weight / np.sqrt(denominator)
    derivative_weight = (
        weight[:, None]
        * np.column_stack((rho, tau))
        / denominator[:, None]
    )
    return (
        weight[:, None] * base_residual,
        weight[:, None, None] * base_derivatives
        + derivative_weight[:, :, None] * base_residual[:, None, :],
    )


def _lsqc_derivative_actions(
    mesh: TriMesh,
    coefficients: ComplexArray,
    coordinates: FloatArray,
    weighted: bool,
) -> FloatArray:
    """Apply both facewise LSQC operator derivatives without forming blocks."""

    return _lsqc_residual_and_derivative_actions(
        mesh, coefficients, coordinates, weighted
    )[1]


def _weighted_lsqc_mu_contraction(
    mesh: TriMesh,
    coefficients: ComplexArray,
    coordinates: FloatArray,
    adjoint_coordinates: FloatArray,
) -> FloatArray:
    """Contract ``d(B.T @ B)`` while sharing face-gradient evaluation."""

    primal_residuals, primal_actions = _lsqc_residual_and_derivative_actions(
        mesh, coefficients, coordinates, True
    )
    adjoint_residuals, adjoint_actions = _lsqc_residual_and_derivative_actions(
        mesh, coefficients, adjoint_coordinates, True
    )
    gradient = -np.einsum(
        "fr,fpr->fp", adjoint_residuals, primal_actions, optimize=True
    )
    gradient -= np.einsum(
        "fr,fpr->fp", primal_residuals, adjoint_actions, optimize=True
    )
    return gradient


def _lsqc_augmented_mu_contraction(
    mesh: TriMesh,
    coefficients: ComplexArray,
    result: SolveResult,
    adjoint: FloatArray,
    weighted: bool,
) -> FloatArray:
    """Contract an already-solved augmented LSQC primal/adjoint pair."""

    if not isinstance(result.state, SparseSolveState):
        raise ValueError("augmented LSQC contraction requires a SparseSolveState")
    if result.state.residual_slice is None:
        raise ValueError("result does not contain LSQC residual variables")
    primal_coordinates = result.state.solution[result.state.coordinate_slice]
    primal_residuals = result.state.solution[result.state.residual_slice]
    adjoint_coordinates = adjoint[result.state.coordinate_slice]
    adjoint_residuals = adjoint[result.state.residual_slice]
    primal_actions = _lsqc_derivative_actions(
        mesh, coefficients, primal_coordinates, weighted
    )
    adjoint_actions = _lsqc_derivative_actions(
        mesh, coefficients, adjoint_coordinates, weighted
    )
    face_residuals = primal_residuals.reshape(mesh.n_faces, 2)
    face_adjoint_residuals = adjoint_residuals.reshape(mesh.n_faces, 2)
    gradient = -np.einsum(
        "fr,fpr->fp", face_adjoint_residuals, primal_actions, optimize=True
    )
    gradient -= np.einsum(
        "fr,fpr->fp", face_residuals, adjoint_actions, optimize=True
    )
    return gradient


def lsqc_mu_vjp(
    mesh: TriMesh,
    mu: ComplexArray,
    result: SolveResult,
    uv_bar: FloatArray,
    weighted: bool = False,
) -> AdjointResult:
    coefficients = validate_mu(mesh, mu)
    adjoint, residual = _adjoint_vector(result, uv_bar)
    gradient = _lsqc_augmented_mu_contraction(
        mesh, coefficients, result, adjoint, weighted
    )
    return AdjointResult(gradient=gradient, residual=residual)
