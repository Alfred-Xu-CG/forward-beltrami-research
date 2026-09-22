"""Strict-JSON Route-II fixed-boundary O1--O4 instance benchmark.

One deterministic CPU-float64 Route-I target is built once and passed to the
comparison runner.  The receipt keeps every selected method and its full trace,
including failures.  Two predeclared views are deliberately separate:

* primary: exactly 83 completed global solves (steps 41/41/27/81);
* secondary: outer step 40 (solves 81/81/122/42).

For interpretable process memory, launch one method per fresh process, e.g.::

    python experiments/phase5/route2_mvc_instance_benchmark.py \
      --task image --methods O4 --N 17 --R 256 --steps 81 --lr 1e-3 \
      --threshold 1e-4 --seed 20260922 --strength 0.12 \
      --image medical_phantom --backend directed_iterative \
      --dtype float32 --device cuda

The target identity is a compact numeric summary; the full dense 256-square
tensor is never serialized.  CPU HWM is process-lifetime, while CUDA allocator
baseline/peak covers comparison plus independent final re-decode audits.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, is_dataclass
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

# This Anaconda/Windows environment otherwise lets PyTorch initialize its Intel
# OpenMP runtime before NumPy's LAPACK runtime.  Exercising the tiny NumPy path
# first is safe and prevents a later duplicate-runtime abort in fresh processes.
_NUMPY_LINALG_BOOTSTRAP = float(np.linalg.cond(np.eye(1, dtype=np.float64)))

import scipy
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from qcopt.mesh import structured_rectangle  # noqa: E402
from qcopt.neural_bijection.benchmarks import IMAGE_NAMES  # noqa: E402
from qcopt.neural_bijection.metrics import compute_p1_map_metrics  # noqa: E402
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer  # noqa: E402
from qcopt.neural_bijection.tutte.instance_optimization import (  # noqa: E402
    build_directed_target,
)
from qcopt.neural_bijection.tutte.iterative import (  # noqa: E402
    MatrixFreeDirectedTutteLayer,
)
from qcopt.neural_bijection.tutte.mvc_instance_optimization import (  # noqa: E402
    METHOD_NAMES,
    extract_exact_global_solve_slice,
    extract_outer_step_slice,
    run_fixed_boundary_comparison,
)


METHOD_ALIASES = {
    "O1": "O1_sigmoid_positive",
    "O2": "O2_row_softmax",
    "O3": "O3_mvc_adam",
    "O4": "O4_covariance_retraction",
    **{name: name for name in METHOD_NAMES},
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """One comparison process; ``control_side`` means vertices per side."""

    task: str = "map"
    methods: tuple[str, ...] = METHOD_NAMES
    control_side: int = 17
    resolution: int = 256
    steps: int = 81
    learning_rate: float = 1.0e-3
    objective_threshold: float | None = None
    seed: int = 20260922
    target_strength: float = 0.12
    image_name: str = "medical_phantom"
    backend: str = "directed_iterative"
    dtype: str = "float32"
    device: str = "cpu"
    slice_global_solves: int = 83
    slice_outer_step: int | None = 40
    system_condition_dense_limit: int = 256
    max_covariance_condition: float = 1.0e8
    threads: int = 1


def _plain(
    value: Any,
    *,
    nonfinite_paths: list[str] | None = None,
    _path: str = "$",
) -> Any:
    """Convert to JSON values and explicitly inventory every nonfinite scalar."""

    if is_dataclass(value):
        return {
            member.name: _plain(
                getattr(value, member.name),
                nonfinite_paths=nonfinite_paths,
                _path=f"{_path}.{member.name}",
            )
            for member in fields(value)
        }
    if isinstance(value, torch.Tensor):
        return _plain(
            value.detach().cpu().tolist(),
            nonfinite_paths=nonfinite_paths,
            _path=_path,
        )
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), nonfinite_paths=nonfinite_paths, _path=_path)
    if isinstance(value, dict):
        return {
            str(key): _plain(
                item,
                nonfinite_paths=nonfinite_paths,
                _path=f"{_path}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _plain(
                item,
                nonfinite_paths=nonfinite_paths,
                _path=f"{_path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            if nonfinite_paths is not None:
                nonfinite_paths.append(_path)
            return None
        return numeric
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _finalize_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Make strict JSON without ever presenting a silent nonfinite success."""

    nonfinite_paths: list[str] = []
    result = _plain(receipt, nonfinite_paths=nonfinite_paths)
    if nonfinite_paths and result.get("status") == "ok":
        result["status"] = "complete_with_failures"
    result["nonfinite_guard"] = {
        "count": len(nonfinite_paths),
        "paths": nonfinite_paths,
    }
    return result


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ("git", "-C", str(REPOSITORY), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        status = subprocess.run(
            ("git", "-C", str(REPOSITORY), "status", "--porcelain"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    value = commit.stdout.strip().lower()
    valid = (
        commit.returncode == 0
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )
    return {
        "commit": value if valid else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _process_memory() -> dict[str, Any]:
    """Best-effort current RSS and process-lifetime HWM without mutating state."""

    rss = hwm = None
    source = "unavailable"
    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.Process().memory_info()
        rss = int(memory.rss)
        peak = getattr(memory, "peak_wset", None)
        hwm = int(peak) if peak is not None else None
        source = "psutil_rss_peak_wset" if peak is not None else "psutil_rss"
    except (ImportError, OSError, ValueError):
        pass
    status = Path("/proc/self/status")
    if status.is_file():
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) * 1024
            source = "linux_proc_status"
        except (OSError, ValueError, IndexError):
            pass
    return {"rss_bytes": rss, "process_hwm_bytes": hwm, "source": source}


def _environment(device: torch.device | None) -> dict[str, Any]:
    result = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        **_git_state(),
    }
    if device is not None:
        result["device"] = str(device)
        if device.type == "cuda" and torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(device)
            result["gpu_capability"] = list(torch.cuda.get_device_capability(device))
            result["gpu_total_bytes"] = int(
                torch.cuda.get_device_properties(device).total_memory
            )
    return result


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    data = value.detach().to(dtype=torch.float64, device="cpu")
    flat = data.reshape(-1)
    if flat.numel() == 0:
        minimum = maximum = None
    else:
        minimum, maximum = float(flat.min()), float(flat.max())
    return {
        "shape": list(data.shape),
        "sum": float(data.sum()),
        "l2": float(torch.linalg.vector_norm(flat)),
        "minimum": minimum,
        "maximum": maximum,
    }


def _target_identity(target) -> dict[str, Any]:
    return {
        name: _tensor_summary(getattr(target, name))
        for name in (
            "interior_logits",
            "boundary_logits",
            "raw_modulus",
            "control",
            "dense",
        )
    }


def _target_topology(metrics) -> dict[str, Any]:
    """Target-only topology; accuracy-to-another-map is intentionally not defined."""

    return {
        "flip_count": metrics.flip_count,
        "minimum_signed_area": metrics.minimum_signed_area,
        "minimum_area_ratio": metrics.minimum_area_ratio,
        "boundary_order_min_gap": metrics.boundary_order_min_gap,
        "global_injectivity_certificate": metrics.global_injectivity_certificate,
        "scope": "target topology only; target-to-target map and Beltrami errors omitted as N/A",
    }


def _normalize_methods(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        try:
            method = METHOD_ALIASES[value]
        except KeyError as error:
            raise ValueError(f"unknown method {value!r}") from error
        if method in normalized:
            raise ValueError(f"duplicate method {method!r}")
        normalized.append(method)
    if not normalized:
        raise ValueError("methods must be nonempty")
    return tuple(normalized)


def _validate_config(
    config: BenchmarkConfig,
) -> tuple[torch.device, torch.dtype, str, tuple[str, ...]]:
    if config.task not in ("map", "image"):
        raise ValueError("task must be map or image")
    if (
        isinstance(config.control_side, bool)
        or not isinstance(config.control_side, int)
        or config.control_side < 3
    ):
        raise ValueError("control_side must be an integer at least three")
    if (
        isinstance(config.resolution, bool)
        or not isinstance(config.resolution, int)
        or config.resolution < 2
    ):
        raise ValueError("resolution must be an integer at least two")
    if (
        isinstance(config.steps, bool)
        or not isinstance(config.steps, int)
        or config.steps < 1
    ):
        raise ValueError("steps must be a positive integer")
    for name, value, permit_zero in (
        ("learning_rate", config.learning_rate, False),
        ("target_strength", config.target_strength, False),
        ("objective_threshold", config.objective_threshold, True),
    ):
        if value is None and name == "objective_threshold":
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if float(value) < 0.0 or (not permit_zero and float(value) == 0.0):
            raise ValueError(f"{name} has an invalid sign")
    if (
        isinstance(config.seed, bool)
        or not isinstance(config.seed, int)
        or config.seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    methods = _normalize_methods(list(config.methods))
    if (
        isinstance(config.slice_global_solves, bool)
        or not isinstance(config.slice_global_solves, int)
        or config.slice_global_solves < 1
    ):
        raise ValueError("slice_global_solves must be a positive integer")
    requested_steps = _expected_steps_for_global_budget(
        config.slice_global_solves, methods
    )
    for method, required_step in requested_steps.items():
        if config.steps < required_step:
            raise ValueError(
                f"requested exact slice at {config.slice_global_solves} global solves "
                f"requires at least {required_step} steps for selected method {method}; "
                f"got steps={config.steps}"
            )
    if config.slice_outer_step is not None and (
        isinstance(config.slice_outer_step, bool)
        or not isinstance(config.slice_outer_step, int)
        or config.slice_outer_step < 0
    ):
        raise ValueError("slice_outer_step must be nonnegative or None")
    if config.slice_outer_step is not None and config.steps < config.slice_outer_step:
        raise ValueError(
            f"requested slice_outer_step={config.slice_outer_step} requires at least "
            f"that many optimization steps; got steps={config.steps}"
        )
    if (
        isinstance(config.system_condition_dense_limit, bool)
        or not isinstance(config.system_condition_dense_limit, int)
        or config.system_condition_dense_limit < 0
    ):
        raise ValueError("system_condition_dense_limit must be a nonnegative integer")
    if (
        isinstance(config.max_covariance_condition, bool)
        or not isinstance(config.max_covariance_condition, (int, float))
        or not math.isfinite(float(config.max_covariance_condition))
        or config.max_covariance_condition <= 1.0
    ):
        raise ValueError("max_covariance_condition must be finite and greater than one")
    if (
        isinstance(config.threads, bool)
        or not isinstance(config.threads, int)
        or config.threads < 1
    ):
        raise ValueError("threads must be a positive integer")
    if config.dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if config.backend not in ("direct", "directed_iterative"):
        raise ValueError("backend must be direct or directed_iterative")
    if config.task == "image" and config.image_name not in IMAGE_NAMES:
        raise ValueError(f"image_name must be one of {IMAGE_NAMES}")
    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if config.backend == "direct" and device.type != "cpu":
        raise ValueError("direct backend is CPU-only")
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    return device, dtype, config.backend, methods


def _method_receipt(result) -> dict[str, Any]:
    excluded = {
        "trace",
        "final_metrics",
        "failure",
        "initial_control",
        "final_control",
        "final_logits",
        "fixed_boundary",
    }
    row = {
        member.name: getattr(result, member.name)
        for member in fields(result)
        if member.name not in excluded
    }
    row.update(
        {
            "status": "success" if result.failure is None else "optimizer_failure",
            "failure": result.failure,
            "final_metrics": result.final_metrics,
            "trace": list(result.trace),
            "final_state_numeric_identity": {
                "control": _tensor_summary(result.final_control),
                "logits": _tensor_summary(result.final_logits),
                "fixed_boundary": _tensor_summary(result.fixed_boundary),
            },
        }
    )
    return row


def _slice_receipt(result, *, interpretation: str) -> dict[str, Any]:
    return {
        "protocol": result.protocol,
        "budget": result.budget,
        "complete": result.complete,
        "interpretation": interpretation,
        "rows": dict(result.rows),
        "missing": dict(result.missing),
    }


def _krylov_report(solver: torch.nn.Module) -> Any:
    diagnostics = getattr(solver, "last_diagnostics", None)
    report = getattr(diagnostics, "forward", None)
    if report is None:
        return None
    return {
        "iterations": report.iterations,
        "relative_residual": report.relative_residual,
        "absolute_residual": report.absolute_residual,
        "converged": report.converged,
        "method": report.method,
        "primary_failure": report.primary_failure,
    }


def _represented_dense_system(
    system,
    logits: torch.Tensor,
    boundary: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the CPU-float64 system represented by one logits/boundary pair."""

    logits64 = logits.detach().to(dtype=torch.float64, device="cpu")
    boundary64 = boundary.detach().to(dtype=torch.float64, device="cpu")
    valid = torch.as_tensor(system.valid_mask, dtype=torch.bool)
    probabilities = torch.softmax(
        logits64.masked_fill(~valid, -torch.inf), dim=-1
    ).numpy()
    matrix = np.eye(system.n_rows, dtype=np.float64)
    rhs = np.zeros((system.n_rows, 2), dtype=np.float64)
    rows, slots = np.nonzero(system.valid_mask)
    is_boundary = system.neighbor_is_boundary[rows, slots]
    interior_rows, interior_slots = rows[~is_boundary], slots[~is_boundary]
    boundary_rows, boundary_slots = rows[is_boundary], slots[is_boundary]
    np.add.at(
        matrix,
        (
            interior_rows,
            system.neighbors[interior_rows, interior_slots],
        ),
        -probabilities[interior_rows, interior_slots],
    )
    np.add.at(
        rhs,
        boundary_rows,
        probabilities[boundary_rows, boundary_slots, None]
        * boundary64.numpy()[system.neighbors[boundary_rows, boundary_slots]],
    )
    return matrix, rhs


def _residual_statistics(
    matrix: np.ndarray,
    rhs: np.ndarray,
    interior: np.ndarray,
    *,
    matrix_norm: float,
) -> dict[str, Any]:
    residual = rhs - matrix @ interior
    residual_norms = np.linalg.norm(residual, axis=0)
    rhs_norms = np.linalg.norm(rhs, axis=0)
    relative = np.divide(
        residual_norms,
        rhs_norms,
        out=np.where(residual_norms == 0.0, 0.0, np.inf),
        where=rhs_norms > 0.0,
    )
    denominator = matrix_norm * float(np.linalg.norm(interior)) + float(
        np.linalg.norm(rhs)
    )
    joint_scaled_residual = (
        float(np.linalg.norm(residual)) / denominator
        if denominator > 0.0
        else float(np.linalg.norm(residual))
    )
    return {
        "residual_frobenius_norm": float(np.linalg.norm(residual)),
        "residual_column_norms_2": residual_norms.tolist(),
        "relative_residuals_2": relative.tolist(),
        "maximum_relative_residual_2": float(np.max(relative)),
        "joint_scaled_residual_indicator": joint_scaled_residual,
    }


def _agreement_contract(
    *,
    system,
    logits: torch.Tensor,
    boundary: torch.Tensor,
    accepted: torch.Tensor,
    redecoded: torch.Tensor,
    authority: torch.Tensor,
    backend: str,
    dtype: torch.dtype,
    condition_dense_limit: int,
) -> dict[str, Any]:
    """Residual/conditioning contract against a CPU-float64 direct authority."""

    matrix, rhs = _represented_dense_system(system, logits, boundary)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 1.0
    smallest = float(singular_values[-1]) if singular_values.size else 1.0
    if not math.isfinite(smallest) or smallest <= 0.0:
        raise ValueError("represented directed system must be nonsingular")
    inverse_norm = 1.0 / smallest
    condition = largest / smallest
    interior = np.asarray(system.interior, dtype=np.int64)
    accepted_np = accepted.detach().to(dtype=torch.float64, device="cpu").numpy()
    redecoded_np = redecoded.detach().to(dtype=torch.float64, device="cpu").numpy()
    authority_np = authority.detach().to(dtype=torch.float64, device="cpu").numpy()
    accepted_stats = _residual_statistics(
        matrix, rhs, accepted_np[interior], matrix_norm=largest
    )
    redecoded_stats = _residual_statistics(
        matrix, rhs, redecoded_np[interior], matrix_norm=largest
    )
    authority_stats = _residual_statistics(
        matrix, rhs, authority_np[interior], matrix_norm=largest
    )

    maximum_degree = int(np.max(np.sum(system.valid_mask, axis=1), initial=0))
    operation_depth = maximum_degree + 2
    dtype_roundoff = 32.0 * operation_depth * float(torch.finfo(dtype).eps)
    authority_roundoff = 32.0 * operation_depth * float(torch.finfo(torch.float64).eps)
    solver_rtol = (
        None if backend == "direct" else (1.0e-5 if dtype == torch.float32 else 1.0e-10)
    )
    residual_limit = dtype_roundoff + (solver_rtol or 0.0)
    rhs_column_norms = np.linalg.norm(rhs, axis=0)
    candidate_residual_allowance = float(
        np.linalg.norm(residual_limit * rhs_column_norms)
    )
    authority_residual_allowance = float(
        np.linalg.norm(authority_roundoff * rhs_column_norms)
    )
    authority_scale = max(1.0, float(np.linalg.norm(authority_np[interior])))
    output_roundoff_floor = authority_roundoff * authority_scale
    forward_error_bound = (
        inverse_norm * (candidate_residual_allowance + authority_residual_allowance)
        + output_roundoff_floor
    )
    pairwise_error_bound = (
        2.0 * inverse_norm * candidate_residual_allowance + output_roundoff_floor
    )

    accepted_to_authority = float(
        np.max(np.linalg.norm(accepted_np - authority_np, axis=1))
    )
    redecoded_to_authority = float(
        np.max(np.linalg.norm(redecoded_np - authority_np, axis=1))
    )
    accepted_to_redecoded = float(
        np.max(np.linalg.norm(accepted_np - redecoded_np, axis=1))
    )
    accepted_posteriori_bound = (
        inverse_norm
        * (
            accepted_stats["residual_frobenius_norm"]
            + authority_stats["residual_frobenius_norm"]
        )
        + output_roundoff_floor
    )
    redecoded_posteriori_bound = (
        inverse_norm
        * (
            redecoded_stats["residual_frobenius_norm"]
            + authority_stats["residual_frobenius_norm"]
        )
        + output_roundoff_floor
    )
    return {
        "backend": backend,
        "authority": "CPU-float64 DirectTutteLayer replay of final logits and boundary",
        "condition_estimation": {
            "method": "full_dense_svd_exact_2norm",
            "interior_row_count": int(matrix.shape[0]),
            "dense_limit": condition_dense_limit,
            "dense_matrix_storage_bytes": int(matrix.nbytes),
            "asymptotic_storage": "O(I^2)",
            "asymptotic_work": "O(I^3)",
            "scope": (
                "independent final-state agreement audit only; not part of the "
                "forward/backward neural layer"
            ),
        },
        "condition_number_2": condition,
        "matrix_norm_2": largest,
        "inverse_operator_norm_2": inverse_norm,
        "solver_relative_residual_limit": solver_rtol,
        "roundoff_relative_allowance": dtype_roundoff,
        "residual_relative_limit": residual_limit,
        "authority_roundoff_relative_limit": authority_roundoff,
        "accepted": accepted_stats,
        "redecoded": redecoded_stats,
        "direct_authority": authority_stats,
        "accepted_residual_within_limit": (
            accepted_stats["maximum_relative_residual_2"] <= residual_limit
        ),
        "redecoded_residual_within_limit": (
            redecoded_stats["maximum_relative_residual_2"] <= residual_limit
        ),
        "direct_authority_residual_within_limit": (
            authority_stats["maximum_relative_residual_2"] <= authority_roundoff
        ),
        "forward_error_bound": forward_error_bound,
        "pairwise_forward_error_bound": pairwise_error_bound,
        "accepted_a_posteriori_forward_error_bound": accepted_posteriori_bound,
        "redecoded_a_posteriori_forward_error_bound": redecoded_posteriori_bound,
        "accepted_within_forward_error_bound": (
            accepted_to_authority <= forward_error_bound
            and accepted_to_authority <= accepted_posteriori_bound
        ),
        "redecoded_within_forward_error_bound": (
            redecoded_to_authority <= forward_error_bound
            and redecoded_to_authority <= redecoded_posteriori_bound
        ),
        "accepted_redecoded_within_pairwise_bound": (
            accepted_to_redecoded <= pairwise_error_bound
        ),
    }


def _independent_redecode(
    mesh,
    result,
    *,
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
    condition_dense_limit: int,
) -> dict[str, Any]:
    attempts = 0
    completed = 0
    authority_attempts = 0
    authority_completed = 0
    try:
        solver: torch.nn.Module
        if backend == "direct":
            solver = DirectTutteLayer(mesh)
        else:
            solver = MatrixFreeDirectedTutteLayer(mesh)
        solver = solver.to(device=device, dtype=dtype)
        interior_row_count = int(solver.system.n_rows)
        if interior_row_count > condition_dense_limit:
            storage_bytes = interior_row_count * interior_row_count * 8
            return {
                "status": "scale_limit_failure",
                "verified": False,
                "audit_primal_attempts": attempts,
                "audit_primal_solves": completed,
                "authority_direct_primal_attempts": authority_attempts,
                "authority_direct_primal_solves": authority_completed,
                "condition_estimation": {
                    "method": "not_computed_n_rows_gt_dense_limit",
                    "interior_row_count": interior_row_count,
                    "dense_limit": condition_dense_limit,
                    "dense_matrix_storage_bytes_if_computed": storage_bytes,
                    "asymptotic_storage": "O(I^2)",
                    "asymptotic_work": "O(I^3)",
                    "fail_closed": True,
                },
                "error": {
                    "type": "ConditionAuditScaleLimit",
                    "message": (
                        "condition-aware agreement audit was not evaluated: "
                        f"interior rows {interior_row_count} exceed configured "
                        f"dense limit {condition_dense_limit}; full dense SVD is "
                        "intentionally skipped and the audit fails closed"
                    ),
                },
                "scope": (
                    "independent final-accepted-state-only agreement audit; "
                    "full dense SVD is limited because it requires O(I^2) "
                    "storage and O(I^3) work"
                ),
            }
        logits = result.final_logits.to(device=device, dtype=dtype)
        boundary = result.fixed_boundary.to(device=device, dtype=dtype)
        with torch.no_grad():
            attempts += 1
            recovered = solver(logits, boundary)
            completed += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        recovered64 = recovered.detach().to(dtype=torch.float64, device="cpu")
        expected64 = result.final_control.detach().to(dtype=torch.float64, device="cpu")
        authority_solver = DirectTutteLayer(mesh)
        authority_logits = result.final_logits.detach().to(
            dtype=torch.float64, device="cpu"
        )
        authority_boundary = result.fixed_boundary.detach().to(
            dtype=torch.float64, device="cpu"
        )
        with torch.no_grad():
            authority_attempts += 1
            authority = authority_solver(authority_logits, authority_boundary)
            authority_completed += 1
        authority64 = authority.detach().to(dtype=torch.float64, device="cpu")
        agreement = _agreement_contract(
            system=authority_solver.system,
            logits=authority_logits,
            boundary=authority_boundary,
            accepted=expected64,
            redecoded=recovered64,
            authority=authority64,
            backend=backend,
            dtype=dtype,
            condition_dense_limit=condition_dense_limit,
        )
        error = torch.linalg.vector_norm(recovered64 - expected64, dim=-1)
        accepted_authority_error = torch.linalg.vector_norm(
            expected64 - authority64, dim=-1
        )
        redecoded_authority_error = torch.linalg.vector_norm(
            recovered64 - authority64, dim=-1
        )
        loop = torch.as_tensor(mesh.boundary_loops[0].copy(), dtype=torch.int64)
        boundary_error = torch.linalg.vector_norm(
            recovered64.index_select(0, loop) - result.fixed_boundary.to(torch.float64),
            dim=-1,
        )
        accepted_boundary_error = torch.linalg.vector_norm(
            expected64.index_select(0, loop) - authority_boundary,
            dim=-1,
        )
        authority_boundary_error = torch.linalg.vector_norm(
            authority64.index_select(0, loop) - authority_boundary,
            dim=-1,
        )
        topology = compute_p1_map_metrics(
            mesh, recovered64.numpy(), target=expected64.numpy()
        )
        accepted_topology = compute_p1_map_metrics(
            mesh, expected64.numpy(), target=authority64.numpy()
        )
        authority_topology = compute_p1_map_metrics(
            mesh, authority64.numpy(), target=expected64.numpy()
        )
        maximum_vertex_error = float(error.max())
        map_rmse = float(torch.sqrt(torch.mean(error.square())))
        maximum_boundary_error = float(boundary_error.max())
        maximum_accepted_authority_error = float(accepted_authority_error.max())
        maximum_redecoded_authority_error = float(redecoded_authority_error.max())
        maximum_accepted_boundary_error = float(accepted_boundary_error.max())
        maximum_authority_boundary_error = float(authority_boundary_error.max())
        boundary_atol = 4096.0 * float(torch.finfo(dtype).eps)
        diagnostics = getattr(solver, "last_diagnostics", None)
        forward_report = getattr(diagnostics, "forward", None)
        iterative_report_converged = (
            None
            if backend == "direct"
            else (
                forward_report is not None
                and bool(torch.as_tensor(forward_report.converged).all())
            )
        )
        reasons: list[str] = []
        for label in ("accepted", "redecoded", "direct_authority"):
            if not agreement[f"{label}_residual_within_limit"]:
                reasons.append(
                    f"{label} map exceeds the represented-system residual contract"
                )
        for label in ("accepted", "redecoded"):
            if not agreement[f"{label}_within_forward_error_bound"]:
                reasons.append(
                    f"{label} map exceeds the condition-aware direct-authority error bound"
                )
        if not agreement["accepted_redecoded_within_pairwise_bound"]:
            reasons.append(
                "accepted and redecoded maps exceed their joint condition-aware error bound"
            )
        if iterative_report_converged is False:
            reasons.append(
                "iterative re-decode did not report converged true residuals"
            )
        if (
            not math.isfinite(maximum_boundary_error)
            or maximum_boundary_error > boundary_atol
            or not math.isfinite(maximum_accepted_boundary_error)
            or maximum_accepted_boundary_error > boundary_atol
            or not math.isfinite(maximum_authority_boundary_error)
            or maximum_authority_boundary_error > boundary_atol
        ):
            reasons.append(
                "accepted, redecoded, or direct-authority boundary disagrees with the fixed boundary"
            )
        if not (
            topology.global_injectivity_certificate
            and accepted_topology.global_injectivity_certificate
            and authority_topology.global_injectivity_certificate
        ):
            reasons.append(
                "accepted, redecoded, or direct-authority map failed the global injectivity certificate"
            )
        return {
            "status": "ok" if not reasons else "verification_failure",
            "verified": not reasons,
            "verification_failures": reasons,
            "agreement_contract": agreement,
            "boundary_atol": boundary_atol,
            "audit_primal_attempts": attempts,
            "audit_primal_solves": completed,
            "authority_direct_primal_attempts": authority_attempts,
            "authority_direct_primal_solves": authority_completed,
            "maximum_vertex_error": maximum_vertex_error,
            "map_rmse": map_rmse,
            "maximum_boundary_error": maximum_boundary_error,
            "maximum_accepted_boundary_error": maximum_accepted_boundary_error,
            "maximum_authority_boundary_error": maximum_authority_boundary_error,
            "maximum_accepted_to_direct_authority_error": (
                maximum_accepted_authority_error
            ),
            "maximum_redecoded_to_direct_authority_error": (
                maximum_redecoded_authority_error
            ),
            "topology": topology,
            "accepted_topology": accepted_topology,
            "direct_authority_topology": authority_topology,
            "krylov_forward": _krylov_report(solver),
            "iterative_report_converged": iterative_report_converged,
            "scope": (
                "independent final-accepted-state-only audit against a CPU-float64 "
                "direct replay; trace rows lack state snapshots; all audit solves are "
                "excluded from optimizer solve budget"
            ),
        }
    except Exception as error:  # audit failures are preserved per method
        return {
            "status": "failure",
            "verified": False,
            "audit_primal_attempts": attempts,
            "audit_primal_solves": completed,
            "authority_direct_primal_attempts": authority_attempts,
            "authority_direct_primal_solves": authority_completed,
            "error": {"type": type(error).__name__, "message": str(error)},
            "scope": (
                "independent final-accepted-state-only audit; trace rows lack state "
                "snapshots; excluded from optimizer solve budget"
            ),
        }


def _cuda_memory_snapshot(
    device: torch.device | None,
    *,
    baseline_allocated: int | None,
    baseline_reserved: int | None,
    benchmark_complete: bool = True,
    failure_stage: str | None = None,
) -> dict[str, Any] | None:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize(device)
        status = "ok" if benchmark_complete else "partial"
        scope = (
            "comparison plus independent final re-decode audits"
            if benchmark_complete
            else f"partial allocator receipt through benchmark failure stage {failure_stage}"
        )
        return {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "measurement_status": status,
            "scope": scope,
        }
    except Exception as error:  # preserve partial CUDA accounting on failure
        return {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "measurement_status": "failure",
            "measurement_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "scope": (
                f"allocator query failed after benchmark failure stage {failure_stage}"
                if not benchmark_complete
                else "allocator query failed after benchmark completion"
            ),
        }


def _expected_step_for_global_budget(method: str, global_solves: int) -> int | None:
    if method in ("O1_sigmoid_positive", "O2_row_softmax"):
        return (
            (global_solves - 1) // 2
            if global_solves >= 1 and global_solves % 2
            else None
        )
    if method == "O3_mvc_adam":
        return (
            (global_solves - 2) // 3
            if global_solves >= 2 and (global_solves - 2) % 3 == 0
            else None
        )
    if method == "O4_covariance_retraction":
        return global_solves - 2 if global_solves >= 2 else None
    raise ValueError(f"unknown method {method!r}")


def _expected_steps_for_global_budget(
    global_solves: int, methods: tuple[str, ...] = METHOD_NAMES
) -> dict[str, int]:
    result: dict[str, int] = {}
    for method in methods:
        step = _expected_step_for_global_budget(method, global_solves)
        if step is None:
            raise ValueError(
                f"{global_solves} global solves is unreachable for selected method {method}"
            )
        result[method] = step
    return result


def _solver_settings(
    backend: str, dtype: torch.dtype, config: BenchmarkConfig
) -> dict[str, Any]:
    if backend == "direct":
        backend_settings = {
            "implementation": "DirectTutteLayer",
            "linear_solver": "SciPy SuperLU",
            "krylov": None,
        }
    else:
        backend_settings = {
            "implementation": "MatrixFreeDirectedTutteLayer",
            "krylov": {
                "rtol": 1.0e-5 if dtype == torch.float32 else 1.0e-10,
                "atol": 0.0,
                "max_iterations": 500,
                "source": "core MatrixFreeDirectedTutteLayer defaults",
            },
        }
    return {
        "backend": backend,
        "torch_threads": config.threads,
        "system_condition_dense_limit": config.system_condition_dense_limit,
        "max_covariance_condition": config.max_covariance_condition,
        **backend_settings,
    }


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run one process receipt; setup/runtime failures remain strict JSON rows."""

    wall_start = perf_counter()
    stage = "validation"
    before = _process_memory()
    device: torch.device | None = None
    baseline_allocated: int | None = None
    baseline_reserved: int | None = None
    old_threads = torch.get_num_threads()
    receipt: dict[str, Any] = {
        "schema": "phase5_route2_mvc_instance_benchmark_v1",
        "status": "failure",
        "stage": stage,
        "configuration": asdict(config),
    }
    try:
        device, dtype, backend, methods = _validate_config(config)
        torch.set_num_threads(config.threads)
        task = "supervised_map" if config.task == "map" else "image_registration"
        receipt["configuration"] = {
            **asdict(config),
            "methods": list(methods),
            "task_internal": task,
            "warp_convention": "backward_map_fixed_to_moving",
            "fresh_process_single_method": len(methods) == 1,
        }
        receipt["environment"] = _environment(device)
        mesh = structured_rectangle(config.control_side - 1, config.control_side - 1)

        stage = "target_setup"
        target_start = perf_counter()
        target = build_directed_target(
            mesh,
            image_height=config.resolution,
            image_width=config.resolution,
            seed=config.seed,
            strength=config.target_strength,
            height=0.9 if config.task == "map" else 1.0,
        )
        target_seconds = perf_counter() - target_start
        after_target = _process_memory()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))

        stage = "comparison"
        comparison = run_fixed_boundary_comparison(
            task=task,
            control_vertices=config.control_side,
            image_resolution=config.resolution,
            steps=config.steps,
            backend=backend,
            dtype=dtype,
            seed=config.seed,
            target_strength=config.target_strength,
            learning_rate=config.learning_rate,
            image_name=config.image_name,
            device=device,
            methods=methods,
            target=target,
            objective_threshold=config.objective_threshold,
            system_condition_dense_limit=config.system_condition_dense_limit,
            max_covariance_condition=config.max_covariance_condition,
        )

        stage = "independent_final_redecode"
        redecode = {
            name: _independent_redecode(
                mesh,
                result,
                backend=backend,
                dtype=dtype,
                device=device,
                condition_dense_limit=config.system_condition_dense_limit,
            )
            for name, result in comparison.results.items()
        }
        cuda_memory = _cuda_memory_snapshot(
            device,
            baseline_allocated=baseline_allocated,
            baseline_reserved=baseline_reserved,
        )
        after = _process_memory()

        primary = extract_exact_global_solve_slice(
            comparison, comparison.primary_global_solve_budget
        )
        secondary = extract_outer_step_slice(
            comparison, comparison.secondary_outer_step
        )
        requested_exact = extract_exact_global_solve_slice(
            comparison, config.slice_global_solves
        )
        requested_outer = (
            None
            if config.slice_outer_step is None
            else extract_outer_step_slice(comparison, config.slice_outer_step)
        )
        target_control64 = target.control.detach().to(torch.float64)
        target_dense64 = target.dense.detach().to(torch.float64)
        method_failure_count = sum(
            result.failure is not None for result in comparison.results.values()
        )
        audit_failure_count = sum(
            row.get("status") != "ok" for row in redecode.values()
        )
        requested_exact_failed = not requested_exact.complete
        requested_outer_failed = (
            requested_outer is not None and not requested_outer.complete
        )
        protocol_failure_count = int(requested_exact_failed) + int(
            requested_outer_failed
        )
        memory_failure_count = int(
            cuda_memory is not None and cuda_memory.get("measurement_status") != "ok"
        )
        receipt.update(
            {
                "status": (
                    "ok"
                    if method_failure_count == 0
                    and audit_failure_count == 0
                    and protocol_failure_count == 0
                    and memory_failure_count == 0
                    else "complete_with_failures"
                ),
                "stage": "complete",
                "research_question": (
                    "fixed-boundary O1-O4 map fitting"
                    if config.task == "map"
                    else "fixed-boundary O1-O4 256-grid-style backward image registration"
                ),
                "shared_target": {
                    "generated_once": True,
                    "setup_primal_solves": 1,
                    "setup_seconds": target_seconds,
                    "comparison_target_setup_primal_solves": comparison.target_setup_primal_solves,
                    "comparison_target_validation_seconds": comparison.target_setup_seconds,
                    "authority_dtype": str(target.control.dtype).removeprefix("torch."),
                    "authority_device": str(target.control.device),
                    "topology": _target_topology(target.metrics),
                    "numeric_identity": _target_identity(target),
                },
                "target_reuse_audit": {
                    "same_python_object_passed_to_comparison": True,
                    "authority_was_reused_without_regeneration": True,
                    "maximum_control_cast_error_vs_cpu_float64_authority": float(
                        torch.max(
                            torch.abs(
                                comparison.target_control.to(torch.float64)
                                - target_control64
                            )
                        )
                    ),
                    "maximum_dense_cast_error_vs_cpu_float64_authority": float(
                        torch.max(
                            torch.abs(
                                comparison.target_dense.to(torch.float64)
                                - target_dense64
                            )
                        )
                    ),
                    "comparison_cast_dtype": config.dtype,
                    "scope": (
                        "The prebuilt CPU-float64 target is authoritative. Differences here "
                        "only measure its expected cast into the comparison dtype."
                    ),
                },
                "problem_size": {
                    "control_side": config.control_side,
                    "control_vertex_count": mesh.n_vertices,
                    "boundary_vertex_count": len(mesh.boundary_loops[0]),
                    "interior_row_count": mesh.n_vertices - len(mesh.boundary_loops[0]),
                    "face_count": int(mesh.faces.shape[0]),
                    "dense_query_count": config.resolution * config.resolution,
                },
                "solver_settings": _solver_settings(backend, dtype, config),
                "protocol": {
                    "primary_global_solve_budget": comparison.primary_global_solve_budget,
                    "primary_expected_accepted_steps": comparison.primary_expected_accepted_steps,
                    "secondary_outer_step": comparison.secondary_outer_step,
                    "secondary_expected_global_solves": comparison.secondary_expected_global_solves,
                    "shared_learning_rate_semantics": comparison.shared_learning_rate_semantics,
                    "pilot_recommendation": comparison.pilot_recommendation,
                    "requested_global_solve_budget": config.slice_global_solves,
                    "requested_expected_accepted_steps": _expected_steps_for_global_budget(
                        config.slice_global_solves, methods
                    ),
                    "requested_outer_step": config.slice_outer_step,
                },
                "protocol_slices": {
                    "primary_exact_global_solves": _slice_receipt(
                        primary,
                        interpretation=(
                            "primary equal-completed-global-solve comparison; target setup and "
                            "independent re-decode audits excluded"
                        ),
                    ),
                    "secondary_common_outer_step": _slice_receipt(
                        secondary,
                        interpretation=(
                            "secondary equal-update-count trajectory; explicitly not an equal-solve comparison"
                        ),
                    ),
                    "requested_exact_global_solves": _slice_receipt(
                        requested_exact,
                        interpretation=(
                            "caller-requested equal-completed-global-solve slice; target setup "
                            "and independent re-decode audits excluded"
                        ),
                    ),
                    "requested_outer_step": (
                        {
                            "enabled": False,
                            "complete": False,
                            "rows": {},
                            "missing": {},
                            "interpretation": "caller disabled the requested outer-step slice",
                        }
                        if requested_outer is None
                        else {
                            "enabled": True,
                            **_slice_receipt(
                                requested_outer,
                                interpretation=(
                                    "caller-requested equal-update-count slice; not equal-solve evidence"
                                ),
                            ),
                        }
                    ),
                },
                "methods": {
                    name: _method_receipt(result)
                    for name, result in comparison.results.items()
                },
                "method_failure_count": method_failure_count,
                "audit_failure_count": audit_failure_count,
                "independent_audit_failure_count": audit_failure_count,
                "protocol_failure_count": protocol_failure_count,
                "protocol_failures": {
                    **(
                        {"requested_exact_global_solves": dict(requested_exact.missing)}
                        if requested_exact_failed
                        else {}
                    ),
                    **(
                        {"requested_outer_step": dict(requested_outer.missing)}
                        if requested_outer_failed
                        else {}
                    ),
                },
                "memory_failure_count": memory_failure_count,
                "independent_final_redecode": redecode,
                "memory": {
                    "cpu": {
                        "before": before,
                        "after_target": after_target,
                        "after": after,
                        "scope": (
                            "RSS snapshots are approximate; HWM is process-lifetime. Use one method "
                            "per fresh process for method attribution."
                        ),
                    },
                    "cuda": cuda_memory,
                },
                "wrapper_wall_seconds": perf_counter() - wall_start,
            }
        )
        return _finalize_receipt(receipt)
    except Exception as error:
        cuda_memory = _cuda_memory_snapshot(
            device,
            baseline_allocated=baseline_allocated,
            baseline_reserved=baseline_reserved,
            benchmark_complete=False,
            failure_stage=stage,
        )
        receipt.update(
            {
                "status": "failure",
                "stage": stage,
                "environment": _environment(device),
                "error": {"type": type(error).__name__, "message": str(error)},
                "memory": {
                    "cpu": {"before": before, "after": _process_memory()},
                    "cuda": cuda_memory,
                },
                "wrapper_wall_seconds": perf_counter() - wall_start,
            }
        )
        return _finalize_receipt(receipt)
    finally:
        torch.set_num_threads(old_threads)


def _parse_methods(value: str) -> tuple[str, ...]:
    pieces = value.split(",")
    if any(not item.strip() for item in pieces):
        raise argparse.ArgumentTypeError(
            "methods must be a nonempty CSV without empty entries"
        )
    raw = [item.strip() for item in pieces]
    try:
        return _normalize_methods(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_optional_outer_step(value: str) -> int | None:
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "slice outer step must be an integer or 'none'"
        ) from error


class _StrictJSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _StrictJSONArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("map", "image"), required=True)
    parser.add_argument("--methods", type=_parse_methods, required=True)
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
    parser.add_argument("--threshold", dest="objective_threshold", type=float)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--strength", dest="target_strength", type=float, default=0.12)
    parser.add_argument("--image", dest="image_name", default="medical_phantom")
    parser.add_argument(
        "--backend", choices=("direct", "directed_iterative"), required=True
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--slice-global-solves", type=int, default=83)
    parser.add_argument(
        "--slice-outer-step", type=_parse_optional_outer_step, default=40
    )
    parser.add_argument("--system-condition-dense-limit", type=int, default=256)
    parser.add_argument("--max-covariance-condition", type=float, default=1.0e8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    parser = _parser()
    output: Path | None = None
    try:
        args = parser.parse_args()
        output = args.output
        config = BenchmarkConfig(
            task=args.task,
            methods=args.methods,
            control_side=args.control_side,
            resolution=args.resolution,
            steps=args.steps,
            learning_rate=args.learning_rate,
            objective_threshold=args.objective_threshold,
            seed=args.seed,
            target_strength=args.target_strength,
            image_name=args.image_name,
            backend=args.backend,
            dtype=args.dtype,
            device=args.device,
            slice_global_solves=args.slice_global_solves,
            slice_outer_step=args.slice_outer_step,
            system_condition_dense_limit=args.system_condition_dense_limit,
            max_covariance_condition=args.max_covariance_condition,
            threads=args.threads,
        )
        receipt = run_benchmark(config)
    except Exception as error:
        receipt = _finalize_receipt(
            {
                "schema": "phase5_route2_mvc_instance_benchmark_v1",
                "status": "argument_failure",
                "stage": "argument_parsing",
                "error": {"type": type(error).__name__, "message": str(error)},
                "environment": _environment(None),
            }
        )
    text = json.dumps(receipt, sort_keys=True, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if receipt["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
