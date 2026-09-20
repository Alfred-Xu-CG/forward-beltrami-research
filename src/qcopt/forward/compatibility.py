"""Shared-edge compatibility tests for facewise Beltrami coefficients."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from numpy.typing import NDArray

from ..mesh import TriMesh

ComplexArray = NDArray[np.complex128]


def compatibility_matrix(mesh: TriMesh, mu: ComplexArray) -> coo_matrix:
    """Assemble complex edge equations for facewise ``f_z`` values.

    For a shared source edge with complex vector ``dz``, continuity of a P1 map
    requires ``(a_T-a_U)dz + (mu_T*a_T-mu_U*a_U)conj(dz)=0``. The returned
    matrix maps the complex face values ``a_T=f_z`` to these residuals.
    """

    coefficients = np.asarray(mu, dtype=np.complex128)
    if coefficients.shape != (mesh.n_faces,):
        raise ValueError("mu must have shape (mesh.n_faces,)")
    if not np.all(np.isfinite(coefficients)) or np.any(np.abs(coefficients) >= 1.0):
        raise ValueError("mu must be finite and lie inside the unit disk")
    edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, (a, b, c) in enumerate(mesh.faces.tolist()):
        for start, end in ((a, b), (b, c), (c, a)):
            key = (min(start, end), max(start, end))
            edge_faces.setdefault(key, []).append((face_index, start, end))
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    row = 0
    for records in edge_faces.values():
        if len(records) != 2:
            continue
        first, second = records
        dz = complex(*(mesh.vertices[first[2]] - mesh.vertices[first[1]]))
        rows.extend((row, row))
        columns.extend((first[0], second[0]))
        values.extend(
            (
                dz + coefficients[first[0]] * np.conjugate(dz),
                -(dz + coefficients[second[0]] * np.conjugate(dz)),
            )
        )
        row += 1
    return coo_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, columns)),
        shape=(row, mesh.n_faces),
    )


def compatibility_residual(matrix: coo_matrix, face_fz: ComplexArray) -> float:
    """Return the maximum absolute shared-edge residual."""

    values = np.asarray(face_fz, dtype=np.complex128)
    if values.ndim != 1 or values.shape[0] != matrix.shape[1]:
        raise ValueError("face_fz must have one value per face")
    if not np.all(np.isfinite(values)):
        raise ValueError("face_fz must be finite")
    return float(np.max(np.abs(matrix @ values))) if matrix.shape[0] else 0.0


def compatibility_nullity(matrix: coo_matrix, *, tolerance: float = 1e-10) -> int:
    """Estimate the complex nullity of a compatibility matrix.

    This diagnostic is intentionally dense and intended for dimension studies
    on small/medium meshes.  A realizable simply-connected face field should
    generically have the one complex scale degree of freedom in this kernel;
    arbitrary facewise coefficients have no nonzero kernel.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    dense = matrix.toarray()
    singular_values = np.linalg.svd(dense, compute_uv=False)
    return int(np.count_nonzero(singular_values <= tolerance)) + max(
        0, dense.shape[1] - min(dense.shape)
    )
