"""Bounded positive-conductance feasibility for a target sampled vertex map."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.optimize
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch

from phase6_fit_dense_map_oracle import target_map
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


def evaluate(side: int, target_kind: str, minimum: float, maximum: float, time_limit: float) -> dict:
    if side < 3 or not (0 < minimum < maximum):
        raise ValueError("invalid grid or conductance bounds")
    began = time.perf_counter()
    mesh = structured_rectangle(side - 1, side - 1)
    reference = MatrixFreeSymmetricTutteLayer(mesh)
    edges = np.asarray(reference.active_edges, dtype=np.int64)
    target = target_map(side, torch.device("cpu"), torch.float64, target_kind)[0].reshape(-1, 2).numpy()
    interior = np.asarray(reference.interior_vertices, dtype=np.int64)
    index = np.full(mesh.n_vertices, -1, dtype=np.int64)
    index[interior] = np.arange(len(interior))
    edge_ids = np.arange(len(edges))
    a, b = edges[:, 0], edges[:, 1]
    displacement = target[a] - target[b]
    row_chunks, col_chunks, value_chunks = [], [], []
    for endpoint, sign in ((a, 1), (b, -1)):
        valid = index[endpoint] >= 0
        current_rows = index[endpoint[valid]]
        current_edges = edge_ids[valid]
        for coordinate in range(2):
            row_chunks.append(2 * current_rows + coordinate)
            col_chunks.append(current_edges)
            value_chunks.append(sign * displacement[valid, coordinate])
    matrix = sparse.coo_matrix(
        (np.concatenate(value_chunks), (np.concatenate(row_chunks), np.concatenate(col_chunks))),
        shape=(2 * len(interior), len(edges)),
    ).tocsr()
    assembled_seconds = time.perf_counter() - began
    result = scipy.optimize.linprog(
        np.zeros(len(edges)), A_eq=matrix, b_eq=np.zeros(matrix.shape[0]),
        bounds=(minimum, maximum), method="highs",
        options={"time_limit": time_limit, "primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8},
    )
    elapsed = time.perf_counter() - began
    weights = result.x
    residual = matrix @ weights if weights is not None else None
    independent_map_max_error = None
    if weights is not None and result.success:
        internal_a, internal_b = index[a], index[b]
        diagonal = np.bincount(index[a][index[a] >= 0], weights=weights[index[a] >= 0], minlength=len(interior))
        diagonal += np.bincount(index[b][index[b] >= 0], weights=weights[index[b] >= 0], minlength=len(interior))
        both = (internal_a >= 0) & (internal_b >= 0)
        operator = sparse.diags(diagonal).tocsr()
        operator += sparse.coo_matrix(
            (np.concatenate((-weights[both], -weights[both])),
             (np.concatenate((internal_a[both], internal_b[both])), np.concatenate((internal_b[both], internal_a[both])))),
            shape=(len(interior), len(interior)),
        ).tocsr()
        rhs = np.zeros((len(interior), 2), dtype=np.float64)
        a_internal_b_boundary = (internal_a >= 0) & (internal_b < 0)
        b_internal_a_boundary = (internal_b >= 0) & (internal_a < 0)
        np.add.at(rhs, internal_a[a_internal_b_boundary], weights[a_internal_b_boundary, None] * target[b[a_internal_b_boundary]])
        np.add.at(rhs, internal_b[b_internal_a_boundary], weights[b_internal_a_boundary, None] * target[a[b_internal_a_boundary]])
        recovered = sparse_linalg.spsolve(operator, rhs)
        independent_map_max_error = float(np.max(np.abs(recovered - target[interior])))
    return {
        "question": "bounded_symmetric_positive_conductance_representation_of_sampled_target",
        "side": side,
        "target_kind": target_kind,
        "control_vertices": mesh.n_vertices,
        "control_faces": mesh.n_faces,
        "interior_vertices": len(interior),
        "active_edges": len(edges),
        "constraint_rows": matrix.shape[0],
        "conductance_bounds": [minimum, maximum],
        "linprog_status": int(result.status),
        "linprog_message": result.message,
        "linprog_success": bool(result.success),
        "min_found_conductance": float(np.min(weights)) if weights is not None else None,
        "max_found_conductance": float(np.max(weights)) if weights is not None else None,
        "equilibrium_max_abs_residual": float(np.max(np.abs(residual))) if residual is not None else None,
        "equilibrium_relative_l2_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(matrix.data), 1e-300)) if residual is not None else None,
        "independent_sparse_recovery_max_coordinate_error": independent_map_max_error,
        "assembly_seconds": assembled_seconds,
        "total_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--target-kind", choices=("base", "high32"), required=True)
    parser.add_argument("--minimum", type=float, default=0.01)
    parser.add_argument("--maximum", type=float, default=100.0)
    parser.add_argument("--time-limit", type=float, default=120.0)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.side, args.target_kind, args.minimum, args.maximum, args.time_limit), sort_keys=True))


if __name__ == "__main__":
    main()
