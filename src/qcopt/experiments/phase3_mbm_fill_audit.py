"""Measure sparse stiffness and LU fill for the MBM structured reference."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

from qcopt.forward.mbm_lbs import _conductivity_tensor, _structured_geometry


def _smooth_coefficient(n: int) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    x = xx / (n - 1) - 0.5
    y = yy / (n - 1) - 0.5
    return 0.25 * np.exp(-(x * x + y * y) / 0.18) * np.exp(0.7j)


def _assemble(n: int):
    coeff = _smooth_coefficient(n)
    triangles, gradients, areas, face_mu = _structured_geometry(coeff)
    face_a = np.stack([_conductivity_tensor(value) for value in face_mu])
    total = n * n
    matrix = lil_matrix((total, total), dtype=np.float64)
    for index, face in enumerate(triangles):
        local = areas[index] * (gradients[index] @ face_a[index] @ gradients[index].T)
        for i, gi in enumerate(face):
            for j, gj in enumerate(face):
                matrix[gi, gj] += local[i, j]
    matrix = matrix.tocsc()
    left = np.arange(0, total, n)
    right = np.arange(n - 1, total, n)
    bottom = np.arange(n)
    top = np.arange((n - 1) * n, n * n)
    free_u = np.setdiff1d(np.arange(total), np.concatenate((left, right)))
    free_w = np.setdiff1d(np.arange(total), np.concatenate((bottom, top)))
    return matrix, matrix[free_u][:, free_u].tocsc(), matrix[free_w][:, free_w].tocsc()


def _one(n: int) -> dict[str, object]:
    started = time.perf_counter()
    full, primary, complementary = _assemble(n)
    assembly = time.perf_counter() - started
    rows = []
    for label, matrix in (("primary", primary), ("complementary", complementary)):
        started = time.perf_counter()
        lu = splu(matrix, permc_spec="COLAMD")
        seconds = time.perf_counter() - started
        factor_nnz = int(lu.L.nnz + lu.U.nnz)
        rows.append(
            {
                "system": label,
                "dimension": int(matrix.shape[0]),
                "matrix_nnz": int(matrix.nnz),
                "factor_nnz_L_plus_U": factor_nnz,
                "fill_ratio": factor_nnz / max(matrix.nnz, 1),
                "factor_seconds": seconds,
            }
        )
    return {
        "vertices_per_axis": n,
        "faces": int(len(_structured_geometry(_smooth_coefficient(n))[0])),
        "full_matrix_nnz": int(full.nnz),
        "assembly_seconds": assembly,
        "systems": rows,
        "ordering": "COLAMD; SciPy SuperLU; CPU reference; one OpenMP thread",
        "cpu": platform.processor() or platform.uname().processor,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertices", type=int, nargs="+", default=[129, 257])
    args = ap.parse_args()
    print(json.dumps({"records": [_one(n) for n in args.vertices]}, indent=2))


if __name__ == "__main__":
    main()
