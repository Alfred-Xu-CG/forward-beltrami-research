#!/usr/bin/env python3
"""Bounded negative-control validation for the Phase-V O4 trust variant.

This executable is deliberately separate from the frozen O1--O4 comparison.
It writes a compact strict-JSON receipt and never overwrites a baseline receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

# Avoid the Windows NumPy/SciPy/PyTorch OpenMP initialization-order failure
# without enabling duplicate runtimes.
np.linalg.solve(np.eye(1), np.ones(1))

import torch

from qcopt.neural_bijection.tutte.mvc_instance_optimization import (
    O4_TRUST_METHOD_NAME,
    run_fixed_boundary_comparison,
)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is nonfinite")
    return value


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().to(dtype=torch.float64, device="cpu")
    return {
        "shape": list(value.shape),
        "minimum": _finite(value.min(), "tensor minimum"),
        "maximum": _finite(value.max(), "tensor maximum"),
        "sum": _finite(value.sum(), "tensor sum"),
        "l2": _finite(torch.linalg.vector_norm(value), "tensor l2"),
    }


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _solve_budget_audit(result, budget: int = 83) -> dict[str, Any]:
    exact = next((row for row in result.trace if row.global_solves == budget), None)
    first_at_or_above = next(
        (row for row in result.trace if row.global_solves >= budget), None
    )

    def identify(row) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "accepted_outer_step": row.iteration,
            "global_solves": row.global_solves,
            "objective": row.objective,
            "minimum_area_ratio": row.minimum_area_ratio,
            "accepted_retraction_scale": row.accepted_retraction_scale,
        }

    return {
        "requested_exact_completed_global_solves": budget,
        "exact_budget_row_available": exact is not None,
        "exact_budget_accepted_state": identify(exact),
        "first_accepted_state_at_or_above_budget": identify(first_at_or_above),
        "last_accepted_global_solves": result.trace[-1].global_solves,
        "terminal_global_solves_including_post_acceptance_failure_trials": (
            result.global_solves
        ),
        "comparable_to_baseline_exact_budget_slice": exact is not None,
        "warning": (
            "Rejected successful trials consume completed global solves. Accepted-state "
            "solve counts may skip 83; an outer-step row or the first row above 83 is "
            "not an exact-83-solve row."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.task not in ("map", "image"):
        raise ValueError("task must be map or image")
    if args.control_side < 3 or args.resolution < 2 or args.steps < 1:
        raise ValueError("N, R, and steps must be positive valid sizes")
    torch.set_num_threads(args.threads)
    start = perf_counter()
    comparison = run_fixed_boundary_comparison(
        task="supervised_map" if args.task == "map" else "image_registration",
        control_vertices=args.control_side,
        image_resolution=args.resolution,
        steps=args.steps,
        backend="direct",
        dtype=torch.float64,
        seed=args.seed,
        target_strength=args.strength,
        learning_rate=args.learning_rate,
        image_name="medical_phantom",
        methods=(O4_TRUST_METHOD_NAME,),
        objective_threshold=None,
        system_condition_dense_limit=args.system_condition_dense_limit,
        o4_trust_backtrack_factor=args.backtrack_factor,
        o4_trust_max_trials=args.max_trials,
        o4_trust_min_area_fraction=args.min_area_fraction,
    )
    result = comparison.results[O4_TRUST_METHOD_NAME]
    failure = None if result.failure is None else asdict(result.failure)
    receipt = {
        "schema": "phase5_route2_o4_trust_validation_v1",
        "research_question": (
            "Can bounded geometry-screened latent backtracking stabilize the "
            "fixed-boundary O4 covariance retraction without post-hoc repair?"
        ),
        "status": "ok" if failure is None else "complete_with_optimizer_failure",
        "configuration": {
            "task": args.task,
            "control_side": args.control_side,
            "resolution": args.resolution,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "target_strength": args.strength,
            "backend": "direct",
            "dtype": "float64",
            "device": "cpu",
            "method": O4_TRUST_METHOD_NAME,
            "backtrack_factor": args.backtrack_factor,
            "max_trials": args.max_trials,
            "min_area_fraction_of_previous_accepted_state": args.min_area_fraction,
            "threads": args.threads,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--porcelain")),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "target": {
            "setup_primal_solves": comparison.target_setup_primal_solves,
            "control": _tensor_summary(comparison.target_control),
            "dense": _tensor_summary(comparison.target_dense),
            "fixed_boundary": _tensor_summary(comparison.fixed_target_boundary),
        },
        "result": {
            "failure": failure,
            "completed_steps": result.completed_steps,
            "initial_objective": result.initial_objective,
            "final_objective": result.final_objective,
            "best_objective": result.best_objective,
            "all_iterates_certified": result.all_iterates_certified,
            "primal_attempts": result.primal_attempts,
            "primal_solves": result.primal_solves,
            "global_solves": result.global_solves,
            "local_backward_attempts": result.local_backward_attempts,
            "local_backwards": result.local_backwards,
            "retraction_trial_attempts": result.retraction_trial_attempts,
            "retraction_trial_solves": result.retraction_trial_solves,
            "retraction_extra_solves": result.retraction_extra_solves,
            "minimum_accepted_retraction_scale": (
                result.minimum_accepted_retraction_scale
            ),
            "maximum_mvc_canonical_covariance_condition": (
                result.maximum_mvc_canonical_covariance_condition
            ),
            "final_metrics": asdict(result.final_metrics),
            "final_state": {
                "control": _tensor_summary(result.final_control),
                "logits": _tensor_summary(result.final_logits),
            },
            "trace": [asdict(row) for row in result.trace],
        },
        "solve_budget_audit": _solve_budget_audit(result, 83),
        "accounting_scope": (
            "Every trust trial increments retraction_trial_attempts. Decoder calls "
            "increment primal_attempts before execution; successful calls increment "
            "primal_solves/global_solves even when the subsequent geometry screen "
            "rejects the candidate. No independent audit solve is included."
        ),
        "interpretation": (
            "Follow-up negative control only; not a member of the frozen baseline "
            "tuple and not an equal-global-solve comparison after rejected trials."
        ),
        "wall_seconds": perf_counter() - start,
    }
    # Refuse to emit non-standard NaN/Infinity JSON.
    json.dumps(receipt, allow_nan=False)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("map", "image"), required=True)
    parser.add_argument(
        "--N", "--control-side", dest="control_side", type=int, required=True
    )
    parser.add_argument(
        "--R", "--resolution", dest="resolution", type=int, required=True
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--lr", "--learning-rate", dest="learning_rate", type=float, required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--min-area-fraction", type=float, default=0.5)
    parser.add_argument("--system-condition-dense-limit", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": receipt["status"], "output": str(args.output)}))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "execution_failure",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
