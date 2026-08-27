"""Sparse Linear Beltrami Solver with general linear constraints."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as splinalg

from .constraints import LinearConstraints
from .mesh import TriMesh
from .solver import SolveResult, SparseSolveState, vector_to_uv

ComplexArray = NDArray[np.complex128]


def validate_mu(mesh: TriMesh, mu: ComplexArray) -> ComplexArray:
    coefficients = np.asarray(mu, dtype=np.complex128).reshape(-1)
    if coefficients.shape != (mesh.n_faces,):
        raise ValueError("mu must contain one complex value per face")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("mu must be finite")
    if np.any(np.abs(coefficients) >= 1.0):
        raise ValueError("every |mu| must be strictly below one")
    return coefficients


def beltrami_tensors(mu: ComplexArray) -> NDArray[np.float64]:
    rho = np.real(mu)
    tau = np.imag(mu)
    denominator = 1.0 - np.square(rho) - np.square(tau)
    tensors = np.empty((len(mu), 2, 2), dtype=np.float64)
    tensors[:, 0, 0] = ((1.0 - rho) ** 2 + tau**2) / denominator
    tensors[:, 0, 1] = -2.0 * tau / denominator
    tensors[:, 1, 0] = tensors[:, 0, 1]
    tensors[:, 1, 1] = ((1.0 + rho) ** 2 + tau**2) / denominator
    return tensors


def assemble_lbs_stiffness(mesh: TriMesh, mu: ComplexArray) -> sparse.csr_matrix:
    coefficients = validate_mu(mesh, mu)
    tensors = beltrami_tensors(coefficients)
    local = mesh.areas[:, None, None] * np.einsum(
        "fki,fij,flj->fkl", mesh.gradients, tensors, mesh.gradients
    )
    rows = np.repeat(mesh.faces, 3, axis=1).reshape(-1)
    columns = np.tile(mesh.faces, (1, 3)).reshape(-1)
    matrix = sparse.coo_matrix(
        (local.reshape(-1), (rows, columns)),
        shape=(mesh.n_vertices, mesh.n_vertices),
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def solve_lbs(
    mesh: TriMesh, mu: ComplexArray, constraints: LinearConstraints
) -> SolveResult:
    if constraints.C.shape[1] != 2 * mesh.n_vertices:
        raise ValueError("constraint matrix has incompatible column count")
    scalar = assemble_lbs_stiffness(mesh, mu)
    operator = sparse.block_diag((scalar, scalar), format="csr")
    zero = sparse.csr_matrix((constraints.C.shape[0], constraints.C.shape[0]))
    system = sparse.bmat(
        [[operator, constraints.C.T], [constraints.C, zero]], format="csc"
    )
    rhs = np.concatenate((np.zeros(2 * mesh.n_vertices), constraints.d))
    factor = splinalg.splu(system)
    solution = factor.solve(rhs)
    coordinate_slice = slice(0, 2 * mesh.n_vertices)
    coordinates = solution[coordinate_slice]
    residual = system @ solution - rhs
    relative_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0)
    )
    constraint_residual = float(
        np.linalg.norm(constraints.C @ coordinates - constraints.d, ord=np.inf)
    )
    state = SparseSolveState(
        system=system,
        rhs=rhs,
        solution=solution,
        factor=factor,
        coordinate_slice=coordinate_slice,
        multiplier_slice=slice(2 * mesh.n_vertices, len(solution)),
    )
    return SolveResult(
        uv=vector_to_uv(coordinates, mesh.n_vertices),
        primal_residual=relative_residual,
        constraint_residual=constraint_residual,
        state=state,
    )
