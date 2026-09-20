"""Augmented weighted and unweighted Least-Squares QC reconstruction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as splinalg

from .constraints import LinearConstraints
from .lbs import assemble_lbs_stiffness, validate_mu
from .mesh import TriMesh
from .solver import ReducedSolveState, SolveResult, SparseSolveState, vector_to_uv

ComplexArray = NDArray[np.complex128]


def assemble_lsqc_area_coupling(mesh: TriMesh) -> sparse.csr_matrix:
    """Assemble the mu-independent LSQC coupling from oriented boundary edges.

    Interior edge contributions of ``area * G @ J @ G.T`` cancel exactly.
    For every oriented boundary edge ``i -> j`` the surviving entries are
    ``S[i, j] = -1/2`` and ``S[j, i] = 1/2``.  This reduces assembly from all
    faces to the boundary and avoids geometry-dependent roundoff cancellation.
    """

    if not mesh.boundary_loops:
        return sparse.csr_matrix((mesh.n_vertices, mesh.n_vertices))
    starts = np.concatenate(mesh.boundary_loops)
    ends = np.concatenate([np.roll(loop, -1) for loop in mesh.boundary_loops])
    rows = np.concatenate((starts, ends))
    columns = np.concatenate((ends, starts))
    values = np.concatenate(
        (
            -0.5 * np.ones_like(starts, dtype=np.float64),
            0.5 * np.ones_like(ends, dtype=np.float64),
        )
    )
    coupling = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(mesh.n_vertices, mesh.n_vertices),
    ).tocsr()
    coupling.sum_duplicates()
    return coupling


def assemble_weighted_lsqc_hessian(
    mesh: TriMesh, mu: ComplexArray
) -> sparse.csr_matrix:
    """Assemble the weighted LSQC paper Hessian without forming ``B.T @ B``.

    For the coordinate ordering ``[u; v]`` the matrix is
    ``[[K(mu), S], [-S, K(mu)]]``, where ``S`` is the mu-independent signed-area
    coupling.  With the residual convention used by :func:`assemble_lsqc_operator`
    this matrix equals its weighted normal equation to roundoff.
    """

    coefficients = validate_mu(mesh, mu)
    stiffness = assemble_lbs_stiffness(mesh, coefficients)
    area_coupling = assemble_lsqc_area_coupling(mesh)
    hessian = sparse.bmat(
        [[stiffness, area_coupling], [-area_coupling, stiffness]], format="csr"
    )
    hessian.sum_duplicates()
    return hessian


def _coordinate_pins(
    constraints: LinearConstraints, n_coordinates: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Extract exact coordinate values from one-nonzero-per-row constraints."""

    if constraints.C.shape[1] != n_coordinates:
        raise ValueError("constraint matrix has incompatible column count")
    matrix = constraints.C.copy()
    matrix.eliminate_zeros()
    if np.any(np.diff(matrix.indptr) != 1):
        raise ValueError(
            "fast LSQC requires one coordinate-selector entry in every constraint row"
        )
    indices = matrix.indices.copy()
    coefficients = matrix.data.copy()
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients == 0.0):
        raise ValueError("coordinate-selector coefficients must be finite and nonzero")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("fast LSQC requires unique coordinate-selector constraints")
    values = constraints.d / coefficients
    if not np.all(np.isfinite(values)):
        raise ValueError("pinned coordinate values must be finite")
    order = np.argsort(indices)
    return indices[order].astype(np.int64, copy=False), values[order]


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


def solve_lsqc_fast(
    mesh: TriMesh,
    mu: ComplexArray,
    constraints: LinearConstraints,
) -> SolveResult:
    """Solve weighted LSQC by direct Hessian assembly and hard-pin elimination.

    This fast path intentionally accepts only exact coordinate-selector
    constraints.  Use :func:`solve_lsqc` as the general augmented reference for
    mixed linear constraints or for unweighted LSQC.
    """

    n_coordinates = 2 * mesh.n_vertices
    pinned_indices, pinned_values = _coordinate_pins(constraints, n_coordinates)
    free_mask = np.ones(n_coordinates, dtype=bool)
    free_mask[pinned_indices] = False
    free_indices = np.flatnonzero(free_mask).astype(np.int64, copy=False)
    if free_indices.size == 0:
        raise ValueError("fast LSQC requires at least one free coordinate")

    hessian = assemble_weighted_lsqc_hessian(mesh, mu)
    free_system = hessian[free_indices][:, free_indices].tocsc()
    free_to_pinned = hessian[free_indices][:, pinned_indices]
    rhs = -np.asarray(free_to_pinned @ pinned_values, dtype=np.float64).reshape(-1)
    factor = splinalg.splu(
        free_system,
        permc_spec="MMD_AT_PLUS_A",
        options={"SymmetricMode": True},
    )
    free_solution = factor.solve(rhs)

    coordinates = np.empty(n_coordinates, dtype=np.float64)
    coordinates[free_indices] = free_solution
    coordinates[pinned_indices] = pinned_values
    reduced_residual = free_system @ free_solution - rhs
    relative_residual = float(
        np.linalg.norm(reduced_residual) / max(np.linalg.norm(rhs), 1.0)
    )
    constraint_residual = float(
        np.linalg.norm(constraints.C @ coordinates - constraints.d, ord=np.inf)
    )
    state = ReducedSolveState(
        system=free_system,
        rhs=rhs,
        solution=free_solution,
        factor=factor,
        coordinates=coordinates,
        free_indices=free_indices,
        pinned_indices=pinned_indices,
        pinned_values=pinned_values,
    )
    return SolveResult(
        uv=vector_to_uv(coordinates, mesh.n_vertices),
        primal_residual=relative_residual,
        constraint_residual=constraint_residual,
        state=state,
    )
