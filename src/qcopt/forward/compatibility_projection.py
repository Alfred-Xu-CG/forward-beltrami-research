"""Small-mesh nonlinear projection onto PL-realizable Beltrami data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from ..beltrami import face_beltrami
from ..mesh import TriMesh


@dataclass(frozen=True)
class BeltramiProjectionResult:
    map: np.ndarray
    projected_mu: np.ndarray
    residual_norm: float
    success: bool
    nfev: int


def projection_mu_jacobian(
    mesh: TriMesh,
    values: np.ndarray,
    *,
    regularization: float = 1e-8,
    interior: np.ndarray | None = None,
) -> csr_matrix:
    """Assemble the sparse real Jacobian for an implicit projection VJP."""
    if regularization < 0.0:
        raise ValueError("regularization must be nonnegative")
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (mesh.n_vertices, 2) or not np.all(np.isfinite(values)):
        raise ValueError("values must have shape (n_vertices,2)")
    if interior is None:
        boundary = np.asarray(mesh.boundary_loops[0], dtype=np.int64)
        boundary_set = set(boundary.tolist())
        interior = np.asarray([v for v in range(mesh.n_vertices) if v not in boundary_set], dtype=np.int64)
    else:
        interior = np.asarray(interior, dtype=np.int64)
    local = values[mesh.faces]
    gradients = mesh.gradients
    ux = np.einsum("fk,fk->f", local[:, :, 0], gradients[:, :, 0])
    uy = np.einsum("fk,fk->f", local[:, :, 0], gradients[:, :, 1])
    vx = np.einsum("fk,fk->f", local[:, :, 1], gradients[:, :, 0])
    vy = np.einsum("fk,fk->f", local[:, :, 1], gradients[:, :, 1])
    fz = 0.5 * ((ux + vy) + 1j * (vx - uy))
    fbar = 0.5 * ((ux - vy) + 1j * (vx + uy))
    if np.any(np.abs(fz) <= 64.0 * np.finfo(np.float64).eps):
        return csr_matrix((2 * mesh.n_faces + (2 * len(interior) if regularization else 0), 2 * len(interior)))
    scale = 1.0 / (fz * fz)
    interior_columns = {int(vertex): 2 * index for index, vertex in enumerate(interior.tolist())}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for face_index, face in enumerate(mesh.faces.tolist()):
        for local_index, vertex in enumerate(face):
            base = interior_columns.get(int(vertex))
            if base is None:
                continue
            gx = gradients[face_index, local_index, 0]
            gy = gradients[face_index, local_index, 1]
            derivatives = (
                scale[face_index] * (0.5 * (gx + 1j * gy) * fz[face_index] - fbar[face_index] * 0.5 * (gx - 1j * gy)),
                scale[face_index] * (0.5 * (-gy + 1j * gx) * fz[face_index] - fbar[face_index] * 0.5 * (gy + 1j * gx)),
            )
            for offset, derivative in enumerate(derivatives):
                rows.append(face_index)
                cols.append(base + offset)
                vals.append(derivative)
    jac = csr_matrix(
        (
            np.concatenate((np.real(vals), np.imag(vals))),
            (np.concatenate((rows, np.asarray(rows) + mesh.n_faces)), np.concatenate((cols, cols))),
        ),
        shape=(2 * mesh.n_faces, 2 * len(interior)),
    )
    if regularization:
        from scipy.sparse import vstack

        count = 2 * len(interior)
        reg_idx = np.arange(count, dtype=np.int64)
        reg = csr_matrix((np.full(count, np.sqrt(regularization)), (reg_idx, reg_idx)), shape=(count, count))
        jac = vstack((jac, reg), format="csr")
    return jac


def project_facewise_mu(
    mesh: TriMesh,
    target_mu: np.ndarray,
    *,
    boundary_map: np.ndarray | None = None,
    initial_map: np.ndarray | None = None,
    max_nfev: int = 300,
    regularization: float = 1e-8,
) -> BeltramiProjectionResult:
    """Fit a continuous P1 map whose facewise coefficient approximates ``target_mu``.

    This is deliberately a small/medium-mesh research prototype. Boundary
    vertices are fixed, interior coordinates are optimized, and the returned
    map is independently audited by callers. It demonstrates a nonlinear
    realizability projection and exposes the residual for generic incompatible
    fields; it is not a global homeomorphism theorem.
    """

    target = np.asarray(target_mu, dtype=np.complex128)
    if target.shape != (mesh.n_faces,) or not np.all(np.isfinite(target)) or np.max(np.abs(target)) >= 1.0:
        raise ValueError("target_mu must be finite, inside the QC disk, and face-sized")
    if regularization < 0.0 or max_nfev < 1:
        raise ValueError("regularization must be nonnegative and max_nfev positive")
    if len(mesh.boundary_loops) != 1:
        raise ValueError("projection prototype requires one boundary loop")
    boundary = np.asarray(mesh.boundary_loops[0], dtype=np.int64)
    boundary_values = mesh.vertices.copy() if boundary_map is None else np.asarray(boundary_map, dtype=np.float64)
    if boundary_values.shape != (mesh.n_vertices, 2) or not np.all(np.isfinite(boundary_values)):
        raise ValueError("boundary_map must have shape (n_vertices,2)")
    interior = np.asarray([v for v in range(mesh.n_vertices) if v not in set(boundary.tolist())], dtype=np.int64)
    start_map = boundary_values.copy() if initial_map is None else np.asarray(initial_map, dtype=np.float64).copy()
    if start_map.shape != (mesh.n_vertices, 2) or not np.all(np.isfinite(start_map)):
        raise ValueError("initial_map must have shape (n_vertices,2)")
    start_map[boundary] = boundary_values[boundary]
    reference = start_map[interior].copy()

    def unpack(vector: np.ndarray) -> np.ndarray:
        values = boundary_values.copy()
        values[interior] = vector.reshape(-1, 2)
        return values

    def objective(vector: np.ndarray) -> np.ndarray:
        values = unpack(vector)
        try:
            current = face_beltrami(mesh, values)
        except ValueError:
            residual = np.full(2 * mesh.n_faces, 1e3, dtype=np.float64)
            if regularization:
                residual = np.concatenate(
                    (residual, np.sqrt(regularization) * (values[interior] - reference).ravel())
                )
            return residual
        mismatch = current - target
        residual = np.concatenate((mismatch.real, mismatch.imag))
        if regularization:
            residual = np.concatenate((residual, np.sqrt(regularization) * (values[interior] - reference).ravel()))
        return residual

    def jacobian(vector: np.ndarray) -> csr_matrix:
        """Assemble the exact sparse real Jacobian of the residual.

        The facewise coefficient is ``mu=fbar/fz``.  Both Wirtinger
        derivatives are linear in the vertex coordinates, so the quotient
        rule gives a sparse analytic Jacobian with at most three local
        vertices per face.  The regularization block is appended explicitly.
        """

        values = unpack(vector)
        local = values[mesh.faces]
        gradients = mesh.gradients
        ux = np.einsum("fk,fk->f", local[:, :, 0], gradients[:, :, 0])
        uy = np.einsum("fk,fk->f", local[:, :, 0], gradients[:, :, 1])
        vx = np.einsum("fk,fk->f", local[:, :, 1], gradients[:, :, 0])
        vy = np.einsum("fk,fk->f", local[:, :, 1], gradients[:, :, 1])
        fz = 0.5 * ((ux + vy) + 1j * (vx - uy))
        fbar = 0.5 * ((ux - vy) + 1j * (vx + uy))
        if np.any(np.abs(fz) <= 64.0 * np.finfo(np.float64).eps):
            # Keep the same large-residual convention as objective().
            return csr_matrix((2 * mesh.n_faces + (2 * len(interior) if regularization else 0), 2 * len(interior)))
        scale = 1.0 / (fz * fz)
        interior_columns = {int(vertex): 2 * index for index, vertex in enumerate(interior.tolist())}
        # The real and imaginary residual halves use the same complex
        # derivative, stored in a real sparse block matrix below.
        complex_rows: list[int] = []
        complex_cols: list[int] = []
        complex_values: list[complex] = []
        for face_index, face in enumerate(mesh.faces.tolist()):
            for local_index, vertex in enumerate(face):
                base = interior_columns.get(int(vertex))
                if base is None:
                    continue
                gx = gradients[face_index, local_index, 0]
                gy = gradients[face_index, local_index, 1]
                derivatives = (
                    scale[face_index] * (
                        0.5 * (gx + 1j * gy) * fz[face_index]
                        - fbar[face_index] * 0.5 * (gx - 1j * gy)
                    ),
                    scale[face_index] * (
                        0.5 * (-gy + 1j * gx) * fz[face_index]
                        - fbar[face_index] * 0.5 * (gy + 1j * gx)
                    ),
                )
                for offset, derivative in enumerate(derivatives):
                    complex_rows.append(face_index)
                    complex_cols.append(base + offset)
                    complex_values.append(derivative)
        jac = csr_matrix(
            (
                np.concatenate((np.real(complex_values), np.imag(complex_values))),
                (
                    np.concatenate((complex_rows, np.asarray(complex_rows) + mesh.n_faces)),
                    np.concatenate((complex_cols, complex_cols)),
                ),
            ),
            shape=(2 * mesh.n_faces, 2 * len(interior)),
        )
        if regularization:
            reg_rows = np.arange(2 * len(interior), dtype=np.int64)
            reg_cols = np.arange(2 * len(interior), dtype=np.int64)
            reg = csr_matrix(
                (np.full(2 * len(interior), np.sqrt(regularization)), (reg_rows, reg_cols)),
                shape=(2 * len(interior), 2 * len(interior)),
            )
            from scipy.sparse import vstack

            jac = vstack((jac, reg), format="csr")
        return jac

    solution = least_squares(
        objective,
        reference.ravel(),
        jac=jacobian,
        max_nfev=max_nfev,
        method="trf",
    )
    fitted = unpack(solution.x)
    projected = face_beltrami(mesh, fitted)
    mismatch = projected - target
    return BeltramiProjectionResult(
        np.ascontiguousarray(fitted),
        np.ascontiguousarray(projected),
        float(np.linalg.norm(np.concatenate((mismatch.real, mismatch.imag)))),
        bool(solution.success),
        int(solution.nfev),
    )
