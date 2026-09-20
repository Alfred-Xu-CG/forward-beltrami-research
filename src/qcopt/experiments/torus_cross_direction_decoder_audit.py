"""Cross-direction positive-graph torus decoder audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.mmatrix import beltrami_conductivity, edge_direction_conductances
from qcopt.forward.torus_tutte import periodic_positive_graph_embedding, periodic_torus_face_determinants
from qcopt.forward.torus_tutte import periodic_tutte_embedding


_DIRS = np.asarray(((1, 0), (0, 1), (1, 1), (1, -1)), dtype=np.int64)
_DIR_UNIT = _DIRS / np.linalg.norm(_DIRS, axis=1, keepdims=True)


def _target(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(t, t, indexing="xy")
    fx = xx + 0.22 * np.sin(2.0 * np.pi * yy) / (2.0 * np.pi) + 0.12 * np.sin(2.0 * np.pi * (xx + yy)) / (2.0 * np.pi)
    fy = yy + 0.18 * np.sin(2.0 * np.pi * xx) / (2.0 * np.pi) - 0.10 * np.sin(2.0 * np.pi * (xx - yy)) / (2.0 * np.pi)
    return np.stack((fx, fy), axis=-1).reshape(-1, 2)


def _face_mu(values: np.ndarray, n: int) -> np.ndarray:
    p00 = values.reshape(n, n, 2)
    p10 = np.roll(p00, -1, axis=1).copy()
    p01 = np.roll(p00, -1, axis=0).copy()
    p10[:, -1] += np.array([1.0, 0.0])
    p01[-1, :] += np.array([0.0, 1.0])
    fx = (p10 - p00) * n
    fy = (p01 - p00) * n
    zx = fx[..., 0] + 1j * fx[..., 1]
    zy = fy[..., 0] + 1j * fy[..., 1]
    fz = 0.5 * (zx - 1j * zy)
    fzb = 0.5 * (zx + 1j * zy)
    return (fzb / fz).reshape(-1)


def _weights(target: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    mu = _face_mu(target, n).reshape(n, n)
    graph = np.empty((4, n, n), dtype=np.float64)
    axis = np.empty((2, n, n), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            tensor = beltrami_conductivity(complex(mu[j, i]))
            fit = edge_direction_conductances(tensor, _DIR_UNIT)
            vals = np.maximum(fit.values, 1e-8)
            graph[:, j, i] = vals
            axis[:, j, i] = vals[:2]
    return graph, axis


def _case(name: str, solved: np.ndarray, target: np.ndarray, n: int, elapsed: float) -> dict:
    target_mu = _face_mu(target, n)
    solved_mu = _face_mu(solved, n)
    det = periodic_torus_face_determinants(solved, n, n)
    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "map_relative_l2_error": float(np.linalg.norm(solved - target) / max(np.linalg.norm(target), 1e-15)),
        "mu_relative_l2_error": float(np.linalg.norm(solved_mu - target_mu) / max(np.linalg.norm(target_mu), 1e-15)),
        "mu_absolute_linf_error": float(np.max(np.abs(solved_mu - target_mu))),
        "min_lifted_face_determinant": float(np.min(det)),
        "flipped_faces": int(np.count_nonzero(det <= 0.0)),
    }


def run(output_dir: Path, n: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = _target(n)
    graph, axis = _weights(target, n)
    records = []
    t0 = time.perf_counter()
    axis_map = periodic_tutte_embedding(n, n, axis[0], axis[1])
    records.append(_case("axis_only", axis_map, target, n, time.perf_counter() - t0))
    t0 = time.perf_counter()
    cross_map = periodic_positive_graph_embedding(n, n, _DIRS, graph)
    records.append(_case("axis_plus_diagonals", cross_map, target, n, time.perf_counter() - t0))
    result = {
        "grid": f"{n}x{n} periodic vertices",
        "vertices": n * n,
        "periodic_faces": 2 * n * n,
        "directions": _DIRS.tolist(),
        "records": records,
        "weight_min": float(np.min(graph)),
        "weight_max": float(np.max(graph)),
        "scope": "positive periodic graph decoder with diagonal cross-direction edges",
        "limitation": "direction dictionary and graph connectivity are still regular-grid controls; arbitrary triangulated torus and exact variable-coefficient consistency remain open",
    }
    (output_dir / "torus_cross_direction_decoder_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()
