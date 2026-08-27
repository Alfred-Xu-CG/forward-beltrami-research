"""Augmented weighted and unweighted Least-Squares QC reconstruction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as splinalg

from .constraints import LinearConstraints
from .lbs import validate_mu
from .mesh import TriMesh
from .solver import SolveResult, SparseSolveState, vector_to_uv

ComplexArray = NDArray[np.complex128]


def assemble_lsqc_operator(
    mesh: TriMesh, mu: ComplexArray, weighted: bool = False
) -> sparse.csr_matrix:
    coefficients = validate_mu(mesh, mu)
    n = mesh.n_vertices
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for face_index, face in enumerate(mesh.faces):
        rho = float(coefficients[face_index].real)
        tau = float(coefficients[face_index].imag)
        weight = float(np.sqrt(mesh.areas[face_index]))
        if weighted:
            weight /= float(np.sqrt(1.0 - abs(coefficients[face_index]) ** 2))
        row0 = 2 * face_index
        row1 = row0 + 1
        for local_vertex, vertex in enumerate(face):
            gx, gy = mesh.gradients[face_index, local_vertex]
            coefficients0 = (
                (1.0 - rho) * gx - tau * gy,
                tau * gx - (1.0 + rho) * gy,
            )
            coefficients1 = (
                -tau * gx + (1.0 + rho) * gy,
                (1.0 - rho) * gx - tau * gy,
            )
            row_indices.extend((row0, row0, row1, row1))
            column_indices.extend((int(vertex), n + int(vertex), int(vertex), n + int(vertex)))
            values.extend(
                (
                    weight * coefficients0[0],
                    weight * coefficients0[1],
                    weight * coefficients1[0],
                    weight * coefficients1[1],
                )
            )
    operator = sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(2 * mesh.n_faces, 2 * mesh.n_vertices),
    ).tocsr()
    operator.sum_duplicates()
    return operator


def solve_lsqc(
    mesh: TriMesh,
    mu: ComplexArray,
    constraints: LinearConstraints,
    weighted: bool = False,
) -> SolveResult:
    if constraints.C.shape[1] != 2 * mesh.n_vertices:
        raise ValueError("constraint matrix has incompatible column count")
    operator = assemble_lsqc_operator(mesh, mu, weighted=weighted)
    n_residuals, n_coordinates = operator.shape
    n_constraints = constraints.C.shape[0]
    residual_identity = -sparse.eye(n_residuals, format="csr")
    zero_coordinates = sparse.csr_matrix((n_coordinates, n_coordinates))
    zero_rc = sparse.csr_matrix((n_residuals, n_constraints))
    zero_cc = sparse.csr_matrix((n_constraints, n_constraints))
    system = sparse.bmat(
        [
            [residual_identity, operator, zero_rc],
            [operator.T, zero_coordinates, constraints.C.T],
            [zero_rc.T, constraints.C, zero_cc],
        ],
        format="csc",
    )
    rhs = np.concatenate(
        (np.zeros(n_residuals + n_coordinates, dtype=np.float64), constraints.d)
    )
    factor = splinalg.splu(system)
    solution = factor.solve(rhs)
    residual_slice = slice(0, n_residuals)
    coordinate_slice = slice(n_residuals, n_residuals + n_coordinates)
    multiplier_slice = slice(n_residuals + n_coordinates, len(solution))
    coordinates = solution[coordinate_slice]
    algebraic_residual = system @ solution - rhs
    relative_residual = float(
        np.linalg.norm(algebraic_residual) / max(np.linalg.norm(rhs), 1.0)
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
        residual_slice=residual_slice,
        multiplier_slice=multiplier_slice,
    )
    return SolveResult(
        uv=vector_to_uv(coordinates, mesh.n_vertices),
        primal_residual=relative_residual,
        constraint_residual=constraint_residual,
        state=state,
    )
