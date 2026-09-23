"""Optional MKL PARDISO versus SuperLU on identical all-edge Tutte systems."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import numpy as np
import psutil
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ExactBlockSchurTutteLayer
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


def _time_solver(label, solver, matrix, rhs, cotangent, full_matrix):
    began = time.perf_counter()
    if label == "superlu":
        factor = sparse_linalg.splu(matrix.tocsc())
        factor_seconds = time.perf_counter() - began
        mapped = factor.solve(rhs)
        forward_seconds = time.perf_counter() - began - factor_seconds
        adjoint_began = time.perf_counter()
        adjoint = factor.solve(cotangent)
    else:
        solver.factorize(matrix)
        factor_seconds = time.perf_counter() - began
        mapped = solver.solve(matrix, rhs)
        forward_seconds = time.perf_counter() - began - factor_seconds
        adjoint_began = time.perf_counter()
        adjoint = solver.solve(matrix, cotangent)
    adjoint_seconds = time.perf_counter() - adjoint_began
    residual = full_matrix @ mapped - rhs
    adjoint_residual = full_matrix @ adjoint - cotangent
    return {
        "factor_seconds": factor_seconds,
        "forward_substitution_seconds": forward_seconds,
        "adjoint_substitution_seconds": adjoint_seconds,
        "full_relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-300)),
        "adjoint_relative_residual": float(np.linalg.norm(adjoint_residual) / max(np.linalg.norm(cotangent), 1e-300)),
        "mapped_interior": mapped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--patch-cells", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--pardiso-mtype", type=int, choices=(2, 11), default=11)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("repeat must be positive")
    try:
        from pypardiso import PyPardisoSolver
    except ImportError as error:
        raise RuntimeError("pypardiso is an optional dependency needed for this benchmark") from error
    mesh = structured_rectangle(args.side - 1, args.side - 1)
    layer = ExactBlockSchurTutteLayer(mesh, args.patch_cells)
    edges = MatrixFreeSymmetricTutteLayer(mesh).active_edges
    midpoint = mesh.vertices[edges].mean(axis=1)
    x, y = midpoint[:, 0], midpoint[:, 1]
    logits = -0.8 + 0.4 * np.sin(2 * math.pi * x) * np.sin(2 * math.pi * y)
    logits += 0.6 * np.exp(-((x - 0.42)**2 + (y - 0.61)**2) / 0.015)
    logits += 0.15 * np.sin(16 * math.pi * x) * np.sin(16 * math.pi * y)
    solver = PyPardisoSolver(mtype=args.pardiso_mtype)
    process = psutil.Process()
    cases = []
    peak_rss = process.memory_info().rss
    for repeat in range(args.repeat):
        conductance = layer.minimum_conductance + np.logaddexp(0, logits + 0.05 * repeat)
        began = time.perf_counter()
        full_matrix, rhs = layer._assemble(conductance)
        assembly_seconds = time.perf_counter() - began
        matrix_for_pardiso = sparse.triu(full_matrix, format="csr") if args.pardiso_mtype == 2 else full_matrix
        cotangent = np.sin(np.arange(layer.n_interior, dtype=np.float64)[:, None] * np.array([[0.13, 0.17]]))
        reference = _time_solver("superlu", None, full_matrix, rhs, cotangent, full_matrix)
        candidate = _time_solver("pardiso", solver, matrix_for_pardiso, rhs, cotangent, full_matrix)
        difference = float(np.max(np.abs(candidate["mapped_interior"] - reference["mapped_interior"])))
        mapped = layer._vertices.copy()
        mapped[layer._interior] = candidate["mapped_interior"]
        triangles = mapped[layer._faces]
        first = triangles[:, 1] - triangles[:, 0]
        second = triangles[:, 2] - triangles[:, 0]
        areas = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        peak_rss = max(peak_rss, process.memory_info().rss)
        del reference["mapped_interior"], candidate["mapped_interior"]
        cases.append({
            "changed_weight_sample": repeat,
            "assembly_seconds": assembly_seconds,
            "superlu": reference,
            "pardiso": candidate,
            "maximum_map_coordinate_disagreement": difference,
            "minimum_normalized_face_area": float(areas.min() * (args.side - 1)**2),
        })
    solver.free_memory(everything=True)
    print(json.dumps({
        "question": "optional_exact_all_edge_direct_solver_comparison",
        "side": args.side,
        "control_vertices": mesh.n_vertices,
        "control_faces": mesh.n_faces,
        "active_edges": len(edges),
        "interior_vertices": layer.n_interior,
        "dtype": "float64",
        "device": "cpu",
        "pardiso_mtype": args.pardiso_mtype,
        "repeat": args.repeat,
        "cases": cases,
        "median_superlu_factor_seconds": statistics.median(case["superlu"]["factor_seconds"] for case in cases),
        "median_pardiso_factor_seconds": statistics.median(case["pardiso"]["factor_seconds"] for case in cases),
        "sampled_peak_process_rss_bytes": peak_rss,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
