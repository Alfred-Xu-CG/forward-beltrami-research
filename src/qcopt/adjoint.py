"""Exact implicit VJPs for sparse variable-μ QC reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .lbs import beltrami_tensors, validate_mu
from .mesh import TriMesh
from .solver import SolveResult

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


def lbs_mu_vjp(
    mesh: TriMesh, mu: ComplexArray, result: SolveResult, uv_bar: FloatArray
) -> AdjointResult:
    coefficients = validate_mu(mesh, mu)
    adjoint, residual = _adjoint_vector(result, uv_bar)
    coordinates = result.state.solution[result.state.coordinate_slice]
    adjoint_coordinates = adjoint[result.state.coordinate_slice]
    n = mesh.n_vertices
    u, v = coordinates[:n], coordinates[n:]
    zu, zv = adjoint_coordinates[:n], adjoint_coordinates[n:]
    derivative_rho, derivative_tau = beltrami_tensor_derivatives(coefficients)
    gradient = np.empty((mesh.n_faces, 2), dtype=np.float64)
    for face_index, face in enumerate(mesh.faces):
        basis_gradients = mesh.gradients[face_index]
        local_u, local_v = u[face], v[face]
        local_zu, local_zv = zu[face], zv[face]
        for parameter, tensor_derivative in enumerate(
            (derivative_rho[face_index], derivative_tau[face_index])
        ):
            local_derivative = (
                mesh.areas[face_index]
                * basis_gradients
                @ tensor_derivative
                @ basis_gradients.T
            )
            contraction = (
                local_zu @ local_derivative @ local_u
                + local_zv @ local_derivative @ local_v
            )
            gradient[face_index, parameter] = -float(contraction)
    return AdjointResult(gradient=gradient, residual=residual)


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


def lsqc_mu_vjp(
    mesh: TriMesh,
    mu: ComplexArray,
    result: SolveResult,
    uv_bar: FloatArray,
    weighted: bool = False,
) -> AdjointResult:
    coefficients = validate_mu(mesh, mu)
    if result.state.residual_slice is None:
        raise ValueError("result does not contain LSQC residual variables")
    adjoint, residual = _adjoint_vector(result, uv_bar)
    primal_coordinates = result.state.solution[result.state.coordinate_slice]
    primal_residuals = result.state.solution[result.state.residual_slice]
    adjoint_coordinates = adjoint[result.state.coordinate_slice]
    adjoint_residuals = adjoint[result.state.residual_slice]
    n = mesh.n_vertices
    gradient = np.empty((mesh.n_faces, 2), dtype=np.float64)
    for face_index, face in enumerate(mesh.faces):
        local_primal = np.concatenate(
            (primal_coordinates[face], primal_coordinates[n + face])
        )
        local_adjoint = np.concatenate(
            (adjoint_coordinates[face], adjoint_coordinates[n + face])
        )
        face_residual = primal_residuals[2 * face_index : 2 * face_index + 2]
        face_adjoint_residual = adjoint_residuals[
            2 * face_index : 2 * face_index + 2
        ]
        _, derivative_rho, derivative_tau = _lsqc_local_block_and_derivatives(
            mesh, coefficients[face_index], face_index, weighted
        )
        for parameter, derivative in enumerate((derivative_rho, derivative_tau)):
            contraction = (
                face_adjoint_residual @ derivative @ local_primal
                + face_residual @ derivative @ local_adjoint
            )
            gradient[face_index, parameter] = -float(contraction)
    return AdjointResult(gradient=gradient, residual=residual)
