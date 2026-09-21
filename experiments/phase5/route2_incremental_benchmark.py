"""Compact cold/warm/correction/Woodbury benchmark for Phase-V Route II.

Every method in one row solves the same newly represented directed Tutte
system.  Local and global updates are both included.  Woodbury is run only for
declared local updates with at most 64 changed rows; for a global update its
dense Schur problem is reported rather than hidden behind a cherry-picked
local case.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
import scipy
import scipy.sparse.linalg as sparse_linalg


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from qcopt.mesh import structured_rectangle  # noqa: E402
from qcopt.neural_bijection.metrics import compute_p1_map_metrics  # noqa: E402
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer  # noqa: E402
from qcopt.neural_bijection.tutte.incremental import (  # noqa: E402
    assemble_directed_system,
    bicgstab_two_rhs,
    correction_right_hand_side,
    woodbury_row_update,
)


UPDATE_KINDS = ("local_one", "local_five_percent", "global_small", "global_large")


def _git_commit() -> str | None:
    result = subprocess.run(
        ("git", "-C", str(REPOSITORY), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip().lower()
    return value if result.returncode == 0 and len(value) == 40 else None


def _condition_estimate(matrix, factor) -> float:
    """Estimate the one-norm condition without materializing an inverse."""

    n = matrix.shape[0]
    if n == 0:
        return 1.0
    inverse = sparse_linalg.LinearOperator(
        (n, n),
        matvec=lambda value: factor.solve(value),
        rmatvec=lambda value: factor.solve(value, trans="T"),
        matmat=lambda value: factor.solve(value),
        rmatmat=lambda value: factor.solve(value, trans="T"),
        dtype=np.float64,
    )
    return float(sparse_linalg.norm(matrix, ord=1) * sparse_linalg.onenormest(inverse))


def _iterative_record(result, reference: np.ndarray, seconds: float) -> dict[str, Any]:
    return {
        "seconds": seconds,
        "iterations": list(result.iterations),
        "relative_residuals": list(result.relative_residuals),
        "maximum_error_vs_refactor": float(np.max(np.abs(result.solution - reference))),
    }


def run_case(*, control_side: int, seed: int, update_kind: str) -> dict[str, Any]:
    if control_side < 4:
        raise ValueError("control_side must be at least four")
    if update_kind not in UPDATE_KINDS:
        raise ValueError(f"update_kind must be one of {UPDATE_KINDS}")
    mesh = structured_rectangle(control_side - 1, control_side - 1)
    system = DirectTutteLayer(mesh).system
    rng = np.random.default_rng(seed)
    logits = 0.15 * rng.standard_normal((system.n_rows, system.max_degree))
    boundary = np.asarray(mesh.vertices[system.loop], dtype=np.float64).copy()

    old_a, old_b, _ = assemble_directed_system(system, logits, boundary)
    factor_start = perf_counter()
    old_factor = sparse_linalg.splu(old_a.tocsc())
    old_factor_seconds = perf_counter() - factor_start
    old_y = old_factor.solve(old_b)

    new_logits = logits.copy()
    if update_kind == "local_one":
        changed = np.array([int(rng.integers(system.n_rows))], dtype=np.int64)
        amplitude = 0.05
    elif update_kind == "local_five_percent":
        count = max(1, int(np.ceil(0.05 * system.n_rows)))
        changed = np.sort(rng.choice(system.n_rows, size=count, replace=False))
        amplitude = 0.05
    elif update_kind == "global_small":
        changed = np.arange(system.n_rows, dtype=np.int64)
        amplitude = 0.002
    else:
        changed = np.arange(system.n_rows, dtype=np.int64)
        amplitude = 0.15
    new_logits[changed] += amplitude * rng.standard_normal(new_logits[changed].shape)

    assembly_start = perf_counter()
    new_a, new_b, _ = assemble_directed_system(system, new_logits, boundary)
    assembly_seconds = perf_counter() - assembly_start
    refactor_start = perf_counter()
    new_factor = sparse_linalg.splu(new_a.tocsc())
    reference = new_factor.solve(new_b)
    refactor_seconds = perf_counter() - refactor_start
    condition_estimate = _condition_estimate(new_a, new_factor)
    absolute_tolerance = 1.0e-10 * max(1.0, float(np.linalg.norm(new_b)))

    cold_start = perf_counter()
    cold = bicgstab_two_rhs(new_a, new_b, atol=absolute_tolerance)
    cold_seconds = perf_counter() - cold_start
    warm_start = perf_counter()
    warm = bicgstab_two_rhs(new_a, new_b, x0=old_y, atol=absolute_tolerance)
    warm_seconds = perf_counter() - warm_start
    correction_rhs = correction_right_hand_side(old_a, old_b, old_y, new_a, new_b)
    correction_start = perf_counter()
    correction_delta = bicgstab_two_rhs(new_a, correction_rhs, atol=absolute_tolerance)
    correction_seconds = perf_counter() - correction_start
    corrected_solution = old_y + correction_delta.solution
    # Keep the solver's recorded counts/residuals but compare the shifted map.
    correction = {
        "seconds": correction_seconds,
        "iterations": list(correction_delta.iterations),
        "relative_residuals_to_correction_rhs": list(correction_delta.relative_residuals),
        "maximum_error_vs_refactor": float(np.max(np.abs(corrected_solution - reference))),
        "full_system_relative_residual": float(
            np.linalg.norm(new_b - new_a @ corrected_solution)
            / max(np.linalg.norm(new_b), np.finfo(np.float64).tiny)
        ),
    }

    if update_kind.startswith("local_") and len(changed) <= 64:
        woodbury_start = perf_counter()
        woodbury_result = woodbury_row_update(
            old_a, new_a, new_b, changed, old_factor=old_factor
        )
        woodbury_seconds = perf_counter() - woodbury_start
        woodbury: dict[str, Any] = {
            "status": "success",
            "seconds_excluding_old_factorization": woodbury_seconds,
            "updated_rows": woodbury_result.updated_rows,
            "schur_condition": woodbury_result.schur_condition,
            "relative_residual": woodbury_result.relative_residual,
            "maximum_error_vs_refactor": float(
                np.max(np.abs(woodbury_result.solution - reference))
            ),
        }
    else:
        schur_bytes = int(len(changed) * len(changed) * np.dtype(np.float64).itemsize)
        woodbury = {
            "status": "not_run",
            "reason": (
                "global update would require a dense Schur system whose order equals "
                "the number of interior rows; no low-rank acceleration is implied"
            ),
            "projected_dense_schur_bytes": schur_bytes,
        }

    mapped = np.empty((mesh.n_vertices, 2), dtype=np.float64)
    mapped[system.loop] = boundary
    mapped[system.interior] = reference
    topology = compute_p1_map_metrics(mesh, mapped)
    return {
        "control_side": control_side,
        "control_vertices": mesh.n_vertices,
        "interior_rows": system.n_rows,
        "seed": seed,
        "update_kind": update_kind,
        "update_amplitude": amplitude,
        "changed_row_count": int(len(changed)),
        "changed_row_fraction": float(len(changed) / system.n_rows),
        "assembly_seconds": assembly_seconds,
        "old_factorization_seconds": old_factor_seconds,
        "new_factorization_and_solve_seconds": refactor_seconds,
        "new_matrix_condition_one_estimate": condition_estimate,
        "absolute_residual_tolerance_per_coordinate": absolute_tolerance,
        "map_change_l2": float(np.linalg.norm(reference - old_y)),
        "cold": _iterative_record(cold, reference, cold_seconds),
        "warm": _iterative_record(warm, reference, warm_seconds),
        "correction": correction,
        "woodbury": woodbury,
        "topology": {
            "certified": topology.global_injectivity_certificate,
            "flip_count": topology.flip_count,
            "minimum_area_ratio": topology.minimum_area_ratio,
            "minimum_signed_area": topology.minimum_signed_area,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-sides", default="25,49")
    parser.add_argument("--seeds", default="1701,1702,1703")
    parser.add_argument("--update-kinds", default=",".join(UPDATE_KINDS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sides = [int(value) for value in args.control_sides.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    kinds = [value.strip() for value in args.update_kinds.split(",") if value.strip()]
    started = perf_counter()
    rows = [
        run_case(control_side=side, seed=seed, update_kind=kind)
        for side in sides
        for seed in seeds
        for kind in kinds
    ]
    receipt = {
        "schema": "phase5_route2_incremental_v1",
        "research_question": (
            "Do old-solution warm starts, the exact shifted correction equation, or "
            "row-local Woodbury updates reduce work for identical new Tutte systems?"
        ),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "git_commit": _git_commit(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "configuration": {
            "control_sides": sides,
            "seeds": seeds,
            "update_kinds": kinds,
            "arithmetic": "float64",
            "woodbury_local_row_limit": 64,
        },
        "wall_seconds": perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

