"""Compare symmetric and directed positive Tutte weights at realistic scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from qcopt.forward.tutte import tutte_embedding_weighted
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def _adjacency(mesh):
    adjacency = [set() for _ in range(mesh.n_vertices)]
    for a, b, c in mesh.faces.tolist():
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return adjacency


def _directed_embedding(mesh, target_boundary, rng, skew: float):
    loop = mesh.boundary_loops[0]
    boundary_set = set(loop.tolist())
    interior = [v for v in range(mesh.n_vertices) if v not in boundary_set]
    index = {v: i for i, v in enumerate(interior)}
    boundary_index = {v: i for i, v in enumerate(loop.tolist())}
    adjacency = _adjacency(mesh)
    system = lil_matrix((len(interior), len(interior)), dtype=np.float64)
    coupling = lil_matrix((len(interior), len(loop)), dtype=np.float64)
    for vertex in interior:
        row = index[vertex]
        neighbors = sorted(adjacency[vertex])
        raw = np.exp(skew * rng.normal(size=len(neighbors)))
        raw /= np.sum(raw)
        system[row, row] = 1.0
        for neighbor, weight in zip(neighbors, raw):
            if neighbor in index:
                system[row, index[neighbor]] -= weight
            else:
                coupling[row, boundary_index[neighbor]] += weight
    result = mesh.vertices.copy()
    result[loop] = target_boundary
    if interior:
        solved = spsolve(system.tocsr(), coupling.tocsr() @ target_boundary)
        result[np.asarray(interior)] = solved
    return np.ascontiguousarray(result)


def _edge_weights(mesh, rng, spread: float):
    edges = set()
    for face in mesh.faces.tolist():
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges.add((min(a, b), max(a, b)))
    return {edge: float(np.exp(spread * rng.normal())) for edge in sorted(edges)}


def _record(name, mesh, mapped):
    report = audit_injectivity(mesh, mapped, rectangle=True)
    return {
        "name": name,
        "finite": bool(np.all(np.isfinite(mapped))),
        "flipped_faces": int(len(report.flipped_faces)),
        "minimum_signed_area": float(report.minimum_signed_area_ratio),
        "injectivity_certified": bool(report.certified),
        "boundary_orientation_ok": bool(report.boundary_orientation_ok),
        "boundary_intersections": int(len(report.boundary_intersections)),
        "bad_branch_vertices": int(len(report.bad_branch_vertices)),
    }


def run(output_dir: Path, n: int = 256, seed: int = 20260919) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    loop = mesh.boundary_loops[0]
    target = mesh.vertices[loop].copy()
    rng = np.random.default_rng(seed)
    symmetric = tutte_embedding_weighted(mesh, target, _edge_weights(mesh, rng, 1.0))
    directed_moderate = _directed_embedding(mesh, target, rng, 1.0)
    directed_heterogeneous = _directed_embedding(mesh, target, rng, 3.0)
    records = [
        _record("symmetric_positive_edge_weights", mesh, symmetric),
        _record("directed_positive_row_weights_logspread_1", mesh, directed_moderate),
        _record("directed_positive_row_weights_logspread_3", mesh, directed_heterogeneous),
    ]
    result = {
        "grid": f"{n}x{n} cells",
        "vertices": int(mesh.n_vertices),
        "faces": int(mesh.n_faces),
        "records": records,
        "scope": "symmetric versus nonsymmetric strictly positive Tutte row weights",
        "interpretation": "tests whether positivity plus convex boundary is numerically sufficient without an SPD/symmetric energy",
        "limitation": "directed row weights are a numerical stress control; no general nonsymmetric Tutte theorem or differentiable directed-weight layer is claimed",
    }
    (output_dir / "tutte_symmetric_nonsymmetric_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.seed), indent=2))


if __name__ == "__main__":
    main()
