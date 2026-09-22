"""Common-input P-ref: one fixed-boundary full-Whitney teacher solve.

This benchmark is deliberately *not* instance optimization and it is not a
hard-topology decoder.  It projects the exact facewise P1 Beltrami field of the
shared directed-Tutte authority target through the CPU/float64 Whitney-Hodge
reference, imposing the authority boundary exactly.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from time import perf_counter
from typing import Any

# Must precede NumPy/SciPy/Torch imports in a fresh Windows process.  The
# repository's direct target solver and sparse Whitney reference otherwise
# load incompatible MKL/OpenMP threading runtimes on the tested Conda build.
if platform.system() == "Windows":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import scipy
import scipy.sparse.linalg as sparse_linalg
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qcopt.beltrami import face_beltrami
from qcopt.forward.discrete_conjugacy import assemble_facewise_conductivity
from qcopt.forward.whitney_hodge import WhitneyHodge
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.benchmarks import synthetic_image, warp_image_backward
from qcopt.neural_bijection.metrics import compute_p1_map_metrics
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from qcopt.neural_bijection.tutte.instance_optimization import build_directed_target


PROTOCOL_KIND = "one_shot_fixed_boundary_structure_preserving_reference"
TOPOLOGY_SCOPE = "observed_P1_topology_for_this_target_only_no_general_guarantee"
ROUTE2_AUTHORITY_COMMIT = "74b452c"
IDENTITY_RTOL = 1.0e-12
IDENTITY_ATOL = 1.0e-12


@dataclass(frozen=True)
class BenchmarkConfig:
    task: str = "map"
    control_side: int = 25
    resolution: int = 256
    seed: int = 20260922
    target_strength: float = 0.25
    threads: int = 1
    fd_control_side: int = 5
    fd_resolution: int = 17
    fd_epsilon: float = 1.0e-6
    image_name: str = "medical_phantom"

    @property
    def target_height(self) -> float:
        return 0.9 if self.task == "map" else 1.0


def beltrami_to_tensor(mu: np.ndarray) -> np.ndarray:
    """Vectorized determinant-one ``A(mu)`` in the repository convention."""

    values = np.asarray(mu, dtype=np.complex128)
    squared = values.real * values.real + values.imag * values.imag
    if (
        not np.all(np.isfinite(values.real))
        or not np.all(np.isfinite(values.imag))
        or np.any(squared >= 1.0)
    ):
        raise ValueError("mu must be finite and lie strictly inside the unit disk")
    denominator = 1.0 - squared
    result = np.empty(values.shape + (2, 2), dtype=np.float64)
    result[..., 0, 0] = (1.0 - 2.0 * values.real + squared) / denominator
    result[..., 0, 1] = -2.0 * values.imag / denominator
    result[..., 1, 0] = result[..., 0, 1]
    result[..., 1, 1] = (1.0 + 2.0 * values.real + squared) / denominator
    return result


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    data = value.detach().to(dtype=torch.float64, device="cpu")
    flat = data.reshape(-1)
    return {
        "shape": list(data.shape),
        "sum": float(data.sum()),
        "l2": float(torch.linalg.vector_norm(flat)),
        "minimum": float(flat.min()) if flat.numel() else None,
        "maximum": float(flat.max()) if flat.numel() else None,
    }


def target_identity(target: Any) -> dict[str, Any]:
    """The Route-II numeric-identity contract, with no new hash layer."""

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


def _is_formal_common_input(config: BenchmarkConfig) -> bool:
    return (
        config.control_side == 25
        and config.resolution == 256
        and config.seed == 20260922
        and config.target_strength == 0.25
        and config.image_name == "medical_phantom"
    )


def validate_formal_target_identity(
    identity: dict[str, Any], config: BenchmarkConfig
) -> dict[str, Any]:
    """Require numeric identity with the clean Route-II authority at N25/R256."""

    compared_fields = ["shape", "sum", "l2", "minimum", "maximum"]
    if not _is_formal_common_input(config):
        return {
            "required": False,
            "matched": None,
            "reason": "nonformal tiny/custom configuration",
            "compared_fields": compared_fields,
            "rtol": IDENTITY_RTOL,
            "atol": IDENTITY_ATOL,
        }
    authority_path = (
        ROOT
        / "docs/research_phase5/raw_results"
        / f"route2_mvc_instance_formal_{config.task}_gpu_O1_ai_74b452c_clean.json"
    )
    if not authority_path.is_file():
        raise ValueError(f"missing clean Route-II target authority: {authority_path}")
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority_config = authority.get("configuration", {})
    expected_config = {
        "task": config.task,
        "control_side": config.control_side,
        "resolution": config.resolution,
        "seed": config.seed,
        "target_strength": config.target_strength,
        "image_name": config.image_name,
    }
    for name, expected in expected_config.items():
        if authority_config.get(name) != expected:
            raise ValueError(f"Route-II authority configuration mismatch for {name}")
    environment = authority.get("environment", {})
    if environment.get("dirty") is not False or not str(
        environment.get("commit", "")
    ).startswith(ROUTE2_AUTHORITY_COMMIT):
        raise ValueError("Route-II target authority is not the clean 74b452c receipt")
    expected_identity = authority.get("shared_target", {}).get("numeric_identity")
    if not isinstance(expected_identity, dict) or set(identity) != set(expected_identity):
        raise ValueError("target numeric identity tensor set does not match Route-II")
    for tensor_name, expected_summary in expected_identity.items():
        actual_summary = identity.get(tensor_name)
        if not isinstance(actual_summary, dict):
            raise ValueError(f"target numeric identity missing {tensor_name}")
        for field in compared_fields:
            actual = actual_summary.get(field)
            expected = expected_summary.get(field)
            if field == "shape":
                matched = actual == expected
            elif actual is None or expected is None:
                matched = actual is expected
            else:
                matched = bool(
                    np.isclose(
                        float(actual),
                        float(expected),
                        rtol=IDENTITY_RTOL,
                        atol=IDENTITY_ATOL,
                    )
                )
            if not matched:
                raise ValueError(
                    "target numeric identity mismatch: "
                    f"{tensor_name}.{field}: actual={actual!r}, expected={expected!r}"
                )
    return {
        "required": True,
        "matched": True,
        "authority_path": str(authority_path.relative_to(ROOT)).replace("\\", "/"),
        "authority_commit": environment["commit"],
        "compared_fields": compared_fields,
        "rtol": IDENTITY_RTOL,
        "atol": IDENTITY_ATOL,
    }


def _topology_dict(metrics: Any) -> dict[str, Any]:
    return {
        "flip_count": int(metrics.flip_count),
        "minimum_signed_area": float(metrics.minimum_signed_area),
        "minimum_area_ratio": float(metrics.minimum_area_ratio),
        "boundary_order_min_gap": float(metrics.boundary_order_min_gap),
        "global_injectivity_certificate": bool(metrics.global_injectivity_certificate),
    }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(ROOT), "status", "--porcelain"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _process_memory() -> dict[str, Any]:
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
    proc_status = Path("/proc/self/status")
    if proc_status.is_file():
        try:
            for line in proc_status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) * 1024
            source = "linux_proc_status"
        except (OSError, ValueError, IndexError):
            pass
    return {"rss_bytes": rss, "process_hwm_bytes": hwm, "source": source}


def _environment() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": "cpu",
        "dtype": "float64",
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
        **_git_state(),
    }


def _validate_config(config: BenchmarkConfig) -> None:
    if config.task not in ("map", "image"):
        raise ValueError("task must be 'map' or 'image'")
    if config.control_side < 3 or config.fd_control_side < 3:
        raise ValueError("control sides must be at least three")
    if config.resolution < 2 or config.fd_resolution < 2:
        raise ValueError("resolutions must be at least two")
    if config.threads < 1:
        raise ValueError("threads must be positive")
    if not np.isfinite(config.target_strength) or config.target_strength <= 0.0:
        raise ValueError("target_strength must be finite and positive")
    if not np.isfinite(config.fd_epsilon) or config.fd_epsilon <= 0.0:
        raise ValueError("fd_epsilon must be finite and positive")
    if config.image_name != "medical_phantom":
        raise ValueError("the common-input protocol fixes image_name=medical_phantom")


def _probe_loss(
    task: str,
    dense: torch.Tensor,
    warped: torch.Tensor,
    target_dense: torch.Tensor,
    fixed: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    """Nonzero deterministic diagnostic added to the named benchmark loss."""

    if task == "map":
        x = torch.linspace(-0.7, 0.9, dense.shape[1], dtype=dense.dtype)
        y = torch.linspace(0.6, -0.8, dense.shape[0], dtype=dense.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        weight = torch.stack((xx + 0.17 * yy, yy - 0.11 * xx), dim=-1)
        objective = torch.mean((dense - target_dense).square())
        return objective + 1.0e-3 * torch.mean(dense * weight), "map_mse_plus_weighted_dense_probe"
    x = torch.linspace(-0.8, 0.9, warped.shape[-1], dtype=warped.dtype)
    y = torch.linspace(0.7, -0.6, warped.shape[-2], dtype=warped.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    weight = (xx + 0.23 * yy)[None, None]
    objective = torch.mean((warped - fixed).square())
    return objective + 1.0e-3 * torch.mean(warped * weight), "image_mse_plus_weighted_warp_probe"


def _finite_difference_check(config: BenchmarkConfig) -> dict[str, Any]:
    start = perf_counter()
    mesh = structured_rectangle(config.fd_control_side - 1, config.fd_control_side - 1)
    target = build_directed_target(
        mesh,
        image_height=config.fd_resolution,
        image_width=config.fd_resolution,
        seed=config.seed,
        strength=config.target_strength,
        height=config.target_height,
    )
    target_control_tensor = target.control.detach().to(torch.float64).clone()
    target_control = target_control_tensor.numpy()
    tensor_np = beltrami_to_tensor(face_beltrami(mesh, target_control))
    system = WhitneyHodge(mesh.vertices, mesh.faces)
    boundary = np.array(mesh.boundary_loops[0], dtype=np.int64, copy=True)
    boundary_values = target_control_tensor[boundary].clone()
    table = StructuredDenseQueryTable.from_mesh(
        mesh, height=config.fd_resolution, width=config.fd_resolution
    )
    table.prepare(device="cpu", dtype=torch.float64)
    moving = synthetic_image(
        config.image_name,
        height=config.fd_resolution,
        width=config.fd_resolution,
        dtype=torch.float64,
    )
    target_dense = target.dense.detach().to(torch.float64).clone()
    fixed = warp_image_backward(moving, target_dense)

    tensor = torch.tensor(tensor_np, dtype=torch.float64, requires_grad=True)
    control = system.solve(tensor, boundary, boundary_values)
    dense = table.interpolate(control)
    warped = warp_image_backward(moving, dense)
    loss, loss_kind = _probe_loss(config.task, dense, warped, target_dense, fixed)
    (gradient,) = torch.autograd.grad(loss, (tensor,))

    generator = torch.Generator().manual_seed(config.seed + 17)
    direction = torch.randn(tensor.shape, dtype=torch.float64, generator=generator)
    direction = (direction + direction.transpose(-1, -2)) / 2.0
    direction = direction / torch.linalg.vector_norm(direction)

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        solved = system.solve(value, boundary, boundary_values)
        dense_value = table.interpolate(solved)
        warped_value = warp_image_backward(moving, dense_value)
        return _probe_loss(
            config.task, dense_value, warped_value, target_dense, fixed
        )[0]

    epsilon = config.fd_epsilon
    with torch.no_grad():
        finite_difference = (
            evaluate(tensor + epsilon * direction)
            - evaluate(tensor - epsilon * direction)
        ) / (2.0 * epsilon)
    vjp = torch.sum(gradient * direction)
    absolute = float(torch.abs(finite_difference - vjp))
    scale = max(float(torch.abs(finite_difference)), float(torch.abs(vjp)), 1.0e-12)
    return {
        "control_side": config.fd_control_side,
        "resolution": config.fd_resolution,
        "epsilon": epsilon,
        "loss_kind": loss_kind,
        "finite_difference": float(finite_difference),
        "analytic_vjp": float(vjp),
        "absolute_error": absolute,
        "relative_error": absolute / scale,
        "path": (
            "tensor_to_whitney_solve_to_dense_map"
            if config.task == "map"
            else "tensor_to_whitney_solve_to_dense_map_to_backward_image_warp"
        ),
        "elapsed_seconds": perf_counter() - start,
        "diagnostic_forward_scalar_rhs": 6,
        "diagnostic_backward_scalar_rhs": 2,
        "excluded_from_primary_solve_counts": True,
    }


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    _validate_config(config)
    torch.set_num_threads(config.threads)
    total_start = perf_counter()
    memory_before = _process_memory()

    target_start = perf_counter()
    mesh = structured_rectangle(config.control_side - 1, config.control_side - 1)
    target = build_directed_target(
        mesh,
        image_height=config.resolution,
        image_width=config.resolution,
        seed=config.seed,
        strength=config.target_strength,
        height=config.target_height,
    )
    target_seconds = perf_counter() - target_start

    setup_start = perf_counter()
    target_control = target.control.detach().to(torch.float64).clone()
    target_dense = target.dense.detach().to(torch.float64).clone()
    target_mu = face_beltrami(mesh, target_control.numpy())
    tensor_np = beltrami_to_tensor(target_mu)
    system = WhitneyHodge(mesh.vertices, mesh.faces)
    boundary = np.array(mesh.boundary_loops[0], dtype=np.int64, copy=True)
    interior = np.setdiff1d(np.arange(mesh.n_vertices), boundary)
    boundary_values = target_control[boundary].clone()
    table = StructuredDenseQueryTable.from_mesh(
        mesh, height=config.resolution, width=config.resolution
    )
    table.prepare(device="cpu", dtype=torch.float64)
    moving = synthetic_image(
        config.image_name,
        height=config.resolution,
        width=config.resolution,
        dtype=torch.float64,
    )
    fixed = warp_image_backward(moving, target_dense)
    tensor = torch.tensor(tensor_np, dtype=torch.float64, requires_grad=True)
    setup_seconds = perf_counter() - setup_start

    forward_start = perf_counter()
    control = system.solve(tensor, boundary, boundary_values)
    forward_seconds = perf_counter() - forward_start

    dense_start = perf_counter()
    dense = table.interpolate(control)
    dense_seconds = perf_counter() - dense_start

    warp_start = perf_counter()
    warped = warp_image_backward(moving, dense)
    image_mse_tensor = torch.mean((warped - fixed).square())
    warp_seconds = perf_counter() - warp_start

    probe_loss, backward_loss_kind = _probe_loss(
        config.task, dense, warped, target_dense, fixed
    )
    backward_start = perf_counter()
    probe_loss.backward()
    backward_seconds = perf_counter() - backward_start
    tensor_gradient_norm = float(torch.linalg.vector_norm(tensor.grad))

    values = control.detach().numpy()
    p1_metrics = compute_p1_map_metrics(mesh, values, target=target_control.numpy())
    measured_mu = face_beltrami(mesh, values)
    mu_error = np.abs(measured_mu - target_mu)

    # Independent direct P1 assembly/solve, separate from Whitney-Hodge's
    # local quadrature and custom autograd path.
    direct_operator = assemble_facewise_conductivity(mesh, tensor_np)
    direct_values = np.zeros_like(values)
    direct_values[boundary] = target_control.numpy()[boundary]
    direct_values[interior] = sparse_linalg.spsolve(
        direct_operator[interior][:, interior],
        -direct_operator[interior][:, boundary] @ direct_values[boundary],
    )
    independent_residual = direct_operator @ target_control.numpy()
    whitney_residual = system.apply(tensor.detach(), control.detach()).numpy()

    explicit_state_bytes = int(
        target_control.numel() * target_control.element_size()
        + target_dense.numel() * target_dense.element_size()
        + tensor.numel() * tensor.element_size()
        + boundary_values.numel() * boundary_values.element_size()
        + dense.numel() * dense.element_size()
        + moving.numel() * moving.element_size()
        + fixed.numel() * fixed.element_size()
        + warped.numel() * warped.element_size()
        + table._vertex_indices_cpu.numel() * table._vertex_indices_cpu.element_size()
        + table._barycentric_cpu.numel() * table._barycentric_cpu.element_size()
    )
    primary_done = perf_counter()
    memory_after_primary = _process_memory()
    vjp_check = _finite_difference_check(config)
    memory_after = _process_memory()

    image_mse = float(image_mse_tensor.detach())
    dense_map_mse = float(torch.mean((dense.detach() - target_dense).square()))
    numeric_identity = target_identity(target)
    route2_authority_match = validate_formal_target_identity(numeric_identity, config)
    receipt = {
        "schema": "phase5_route3_pref_common_benchmark_v1",
        "status": "ok",
        "utc": datetime.now(timezone.utc).isoformat(),
        "research_question": (
            "P-ref common-input exact-P1 projection through the full Whitney-Hodge reference"
        ),
        "task": config.task,
        "route_label": "P-ref",
        "protocol_kind": PROTOCOL_KIND,
        "topology_scope": TOPOLOGY_SCOPE,
        "configuration": {
            **asdict(config),
            "target_height": config.target_height,
            "warp_convention": "backward_map_fixed_to_moving",
            "dtype": "float64",
            "device": "cpu",
        },
        "environment": _environment(),
        "optimization": {
            "performed": False,
            "iterations": None,
            "stopping_threshold": None,
        },
        "primary_objective": {
            "kind": (
                "dense_map_coordinate_mse"
                if config.task == "map"
                else "backward_warp_image_mse"
            ),
            "value": dense_map_mse if config.task == "map" else image_mse,
            "optimized": False,
            "scope": "common-input evaluation objective; no parameter update was performed",
        },
        "shared_target": {
            "generated_once": True,
            "setup_primal_solves": 1,
            "authority_dtype": "float64",
            "authority_device": "cpu",
            "topology": _topology_dict(target.metrics),
            "numeric_identity": numeric_identity,
            "route2_authority_match": route2_authority_match,
        },
        "problem_size": {
            "control_side": config.control_side,
            "control_vertex_count": mesh.n_vertices,
            "face_count": mesh.n_faces,
            "image_resolution": config.resolution,
            "dense_query_count": config.resolution * config.resolution,
        },
        "solve_counts": {
            "target_setup_primal_global_solves": 1,
            "primary_forward_factorizations": 1,
            "primary_forward_global_solve_calls": 1,
            "primary_forward_scalar_rhs": 2,
            "primary_backward_reused_factorizations": 1,
            "primary_backward_global_adjoint_calls": 1,
            "primary_backward_scalar_rhs": 2,
            "primary_total_global_solve_calls_including_adjoint": 2,
            "finite_difference_diagnostic_excluded_from_primary": True,
        },
        "timings_seconds": {
            "target_setup": target_seconds,
            "reference_setup": setup_seconds,
            "forward_solve": forward_seconds,
            "dense_interpolation": dense_seconds,
            "dense_warp_and_image_objective": warp_seconds,
            "backward": backward_seconds,
            "primary_total": primary_done - total_start,
            "finite_difference_check": vjp_check["elapsed_seconds"],
            "total_with_diagnostics": perf_counter() - total_start,
        },
        "metrics": {
            **_topology_dict(p1_metrics),
            "map_rmse": float(p1_metrics.map_rmse),
            "maximum_map_error": float(p1_metrics.maximum_map_error),
            "mu_rmse": float(np.sqrt(np.mean(mu_error * mu_error))),
            "maximum_mu_error": float(np.max(mu_error)),
            "image_mse": image_mse,
        },
        "exact_p1_recovery": {
            "claim_scope": (
                "this target-derived facewise exact P1 mu, determinant-one A(mu), "
                "and imposed exact target boundary"
            ),
            "whitney_vs_target_rmse": float(
                np.sqrt(np.mean(np.sum((values - target_control.numpy()) ** 2, axis=1)))
            ),
            "whitney_vs_target_max_abs": float(
                np.max(np.abs(values - target_control.numpy()))
            ),
            "independent_p1_vs_target_max_abs": float(
                np.max(np.abs(direct_values - target_control.numpy()))
            ),
            "whitney_vs_independent_p1_max_abs": float(
                np.max(np.abs(values - direct_values))
            ),
            "independent_operator_residual_linf": float(
                np.max(np.abs(independent_residual[interior]))
            ),
            "whitney_solver_residual_linf": float(
                np.max(np.abs(whitney_residual[interior]))
            ),
            "boundary_max_abs": float(
                np.max(np.abs(values[boundary] - target_control.numpy()[boundary]))
            ),
            "target_mu_max_abs": float(np.max(np.abs(target_mu))),
            "tensor_determinant_max_abs_error": float(
                np.max(np.abs(np.linalg.det(tensor_np) - 1.0))
            ),
        },
        "backward": {
            "loss_kind": backward_loss_kind,
            "loss_value": float(probe_loss.detach()),
            "tensor_gradient_l2": tensor_gradient_norm,
            "boundary_is_fixed": True,
            "diagnostic_only": True,
            "excluded_from_common_primary_objective": True,
            "scope": (
                "first-order tensor VJP through solve and dense map/image path; "
                "the weighted probe exists only to avoid a zero gradient at exact recovery"
            ),
        },
        "vjp_check": vjp_check,
        "memory": {
            "before": memory_before,
            "after_primary": memory_after_primary,
            "after_diagnostics": memory_after,
            "primary_rss_delta_bytes": (
                None
                if memory_before["rss_bytes"] is None
                or memory_after_primary["rss_bytes"] is None
                else memory_after_primary["rss_bytes"] - memory_before["rss_bytes"]
            ),
            "explicit_tensor_and_query_state_bytes": explicit_state_bytes,
            "explicit_scope": (
                "listed dense/control/image/tensor/query tensors only; excludes Python, "
                "SciPy sparse factor internals, allocator overhead, and shared libraries"
            ),
        },
        "replayable_state": {
            "target_builder": {
                "function": "build_directed_target",
                "control_side": config.control_side,
                "image_height": config.resolution,
                "image_width": config.resolution,
                "seed": config.seed,
                "strength": config.target_strength,
                "height": config.target_height,
            },
            "reference_pipeline": (
                "face_beltrami(target.control) -> determinant-one A(mu) -> "
                "WhitneyHodge.solve(A, target boundary)"
            ),
            "authority_numeric_identity": numeric_identity,
            "note": "replay at the recorded Git commit and runtime; no optimized latent state exists",
        },
        "limitations": [
            "one-shot fixed-boundary teacher/reference, not instance optimization",
            "observed injectivity of this output is not a general topology guarantee",
            "CPU float64 sparse reference; no GPU, batch, or matrix-free claim",
        ],
    }
    return receipt


def strict_json_text(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(value, indent=indent, allow_nan=False, sort_keys=True) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("map", "image"), required=True)
    parser.add_argument("--N", "--control-side", dest="control_side", type=int, default=25)
    parser.add_argument("--R", "--resolution", dest="resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--strength", dest="target_strength", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fd-N", dest="fd_control_side", type=int, default=5)
    parser.add_argument("--fd-R", dest="fd_resolution", type=int, default=17)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = BenchmarkConfig(
        task=args.task,
        control_side=args.control_side,
        resolution=args.resolution,
        seed=args.seed,
        target_strength=args.target_strength,
        threads=args.threads,
        fd_control_side=args.fd_control_side,
        fd_resolution=args.fd_resolution,
        fd_epsilon=args.fd_epsilon,
    )
    receipt = run_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strict_json_text(receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "task": receipt["task"],
                "output": str(args.output),
                "map_rmse": receipt["metrics"]["map_rmse"],
                "image_mse": receipt["metrics"]["image_mse"],
                "vjp_relative_error": receipt["vjp_check"]["relative_error"],
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
