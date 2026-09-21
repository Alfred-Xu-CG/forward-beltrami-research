"""Compact M1/M2 differentiable MVC-layer benchmark.

The default bounded profile pairs control-side/batch sizes as
``11:8,25:4,49:1`` and runs both public floating dtypes.  A tiny independent
CPU smoke run is, for example::

    PYTHONPATH=src python experiments/phase5/route2_mvc_layer_benchmark.py \
      --profiles 3:1 --dtypes float64 --devices cpu --backend direct \
      --warmup 0 --repeats 1

CUDA is permitted only with the Route-I matrix-free directed solver.  Timing
uses synchronized wall clock on CUDA.  A row records failures with its stage
and partial receipt instead of aborting the remaining matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from qcopt.mesh import structured_rectangle  # noqa: E402
from qcopt.neural_bijection.metrics import compute_p1_map_metrics  # noqa: E402
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer  # noqa: E402
from qcopt.neural_bijection.tutte.iterative import MatrixFreeDirectedTutteLayer  # noqa: E402
from qcopt.neural_bijection.tutte.mvc import MVCCanonicalizationLayer  # noqa: E402
from qcopt.neural_bijection.tutte.mvc_retraction import CovarianceTutteRetraction  # noqa: E402


@dataclass(frozen=True)
class BenchmarkConfig:
    """One explicit benchmark row; ``control_side`` is vertices per side."""

    control_side: int = 11
    batch: int = 1
    dtype: str = "float64"
    device: str = "cpu"
    backend: str = "auto"
    warmup: int = 1
    repeats: int = 2
    seed: int = 2718
    strength: float = 0.12
    step_size: float = 0.05
    direction_scale: float = 0.02
    fd_epsilon: float | None = None
    solver_rtol: float | None = None
    threads: int = 1
    max_iterations: int = 2000


def _plain(value: Any) -> Any:
    """Make a strict-JSON snapshot; unknown/nonfinite measurements become null."""

    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _require_finite_measurements(value: Any, path: str = "report") -> None:
    """Reject a nominally successful receipt containing a nonfinite measurement."""

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"nonfinite measurement at {path}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating) and not bool(np.isfinite(value).all()):
            raise RuntimeError(f"nonfinite measurement at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_measurements(item, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_finite_measurements(item, f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise RuntimeError(f"nonfinite measurement at {path}")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(REPOSITORY), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().lower()
    if result.returncode != 0 or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        return None
    return value


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _environment(device: torch.device | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git_commit": _git_commit(),
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        result["cuda_device"] = torch.cuda.get_device_name(device)
        result["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def _resolve_backend(config: BenchmarkConfig, device: torch.device) -> str:
    if config.backend not in ("auto", "direct", "matrix_free_directed"):
        raise ValueError("backend must be auto, direct, or matrix_free_directed")
    backend = (
        "matrix_free_directed"
        if config.backend == "auto" and device.type == "cuda"
        else "direct" if config.backend == "auto" else config.backend
    )
    if device.type == "cuda" and backend != "matrix_free_directed":
        raise ValueError("CUDA benchmarking permits only the matrix_free_directed backend")
    if backend == "direct" and device.type != "cpu":
        raise ValueError("direct backend is CPU-only")
    return backend


def _validate_config(config: BenchmarkConfig) -> tuple[torch.device, torch.dtype, str, float]:
    if isinstance(config.control_side, bool) or config.control_side < 3:
        raise ValueError("control_side must be an integer at least three")
    if isinstance(config.batch, bool) or config.batch < 1:
        raise ValueError("batch must be a positive integer")
    if config.dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if config.device not in ("cpu", "cuda") and not config.device.startswith("cuda:"):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if config.warmup < 0 or config.repeats < 1 or config.threads < 1:
        raise ValueError("warmup must be nonnegative and repeats/threads positive")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    for name, value, strictly_positive in (
        ("strength", config.strength, True),
        ("step_size", config.step_size, True),
        ("direction_scale", config.direction_scale, True),
    ):
        if not math.isfinite(value) or (strictly_positive and value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if config.fd_epsilon is not None and (
        not math.isfinite(config.fd_epsilon) or config.fd_epsilon <= 0.0
    ):
        raise ValueError("fd_epsilon must be finite and positive when supplied")
    if config.solver_rtol is not None and (
        isinstance(config.solver_rtol, bool)
        or not math.isfinite(config.solver_rtol)
        or config.solver_rtol <= 0.0
    ):
        raise ValueError("solver_rtol must be finite and positive when supplied")
    device = torch.device(config.device)
    backend = _resolve_backend(config, device)
    if config.solver_rtol is not None and backend != "matrix_free_directed":
        raise ValueError("solver_rtol is supported only by matrix_free_directed")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    epsilon = config.fd_epsilon
    if epsilon is None:
        # Matrix-free float32 solve error dominates very small central steps.
        epsilon = 2.0e-2 if dtype == torch.float32 else 2.0e-4
    return device, dtype, backend, float(epsilon)


def _make_solver(
    mesh,
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
    max_iterations: int,
    solver_rtol: float | None,
):
    if backend == "direct":
        return DirectTutteLayer(mesh)
    relative_tolerance = (
        solver_rtol
        if solver_rtol is not None
        else 1.0e-5 if dtype == torch.float32 else 1.0e-11
    )
    return MatrixFreeDirectedTutteLayer(
        mesh,
        rtol=relative_tolerance,
        atol=0.0,
        max_iter=max_iterations,
    ).to(device)


def _inputs(mesh, system, config: BenchmarkConfig, dtype: torch.dtype, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    logits = config.strength * torch.randn(
        (config.batch, system.n_rows, system.max_degree),
        generator=generator,
        dtype=torch.float64,
    )
    boundary = torch.as_tensor(mesh.vertices[system.loop], dtype=torch.float64)
    boundary = boundary.unsqueeze(0).repeat(config.batch, 1, 1)
    source = torch.from_numpy(mesh.vertices.copy())
    x, y = source[:, 0], source[:, 1]
    direction = torch.stack(
        (
            torch.sin(torch.pi * x) * torch.sin(torch.pi * y),
            0.7 * torch.sin(2.0 * torch.pi * x) * torch.sin(torch.pi * y),
        ),
        dim=-1,
    )
    direction[torch.as_tensor(system.loop)] = 0.0
    batch_scale = 1.0 + 0.1 * torch.arange(config.batch, dtype=torch.float64)
    direction = config.direction_scale * batch_scale[:, None, None] * direction[None]
    return tuple(value.to(device=device, dtype=dtype) for value in (logits, boundary, direction))


def _cotangents(logits: torch.Tensor, control: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    latent = torch.arange(logits[0].numel(), dtype=logits.dtype, device=logits.device)
    latent = torch.cos(0.37 * latent).reshape((1,) + tuple(logits.shape[1:])).expand_as(logits)
    mapped = torch.arange(control[0].numel(), dtype=control.dtype, device=control.device)
    mapped = torch.sin(0.23 * mapped).reshape((1,) + tuple(control.shape[1:])).expand_as(control)
    return latent, mapped


def _gpu_measurement_start(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    _sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "baseline_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "baseline_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _gpu_measurement_finish(device: torch.device, baseline: dict[str, int] | None):
    if baseline is None:
        return None
    _sync(device)
    return {
        **baseline,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _summarize_measurement(
    samples: list[dict[str, Any]],
    *,
    warmup: int,
    gpu_memory: dict[str, int] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "warmup_iterations": warmup,
        "samples": samples,
        "gpu_memory": gpu_memory,
        "gradients_finite": all(sample["gradients_finite"] for sample in samples),
    }
    for field in ("forward_seconds", "loss_seconds", "backward_seconds", "end_to_end_seconds"):
        values = [float(sample[field]) for sample in samples]
        result["mean_" + field] = statistics.mean(values)
        result["median_" + field] = statistics.median(values)
    return result


def _measure_stage(
    iteration: Callable[[], dict[str, Any]],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    for _ in range(warmup):
        iteration()
    baseline = _gpu_measurement_start(device)
    samples = [iteration() for _ in range(repeats)]
    memory = _gpu_measurement_finish(device, baseline)
    return _summarize_measurement(samples, warmup=warmup, gpu_memory=memory)


def _timed_autograd(
    forward: Callable[[], tuple[Any, tuple[torch.Tensor, ...]]],
    loss: Callable[[Any], torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    _sync(device)
    started = perf_counter()
    output, inputs = forward()
    _sync(device)
    forward_done = perf_counter()
    scalar = loss(output)
    _sync(device)
    loss_done = perf_counter()
    gradients = torch.autograd.grad(scalar, inputs)
    _sync(device)
    backward_done = perf_counter()
    gradients_finite = all(bool(torch.isfinite(value).all()) for value in gradients)
    gradient_norms = [float(torch.linalg.vector_norm(value).detach().cpu()) for value in gradients]
    if not gradients_finite:
        raise RuntimeError("a measured backward produced a nonfinite gradient")
    return {
        "forward_seconds": forward_done - started,
        "loss_seconds": loss_done - forward_done,
        "backward_seconds": backward_done - loss_done,
        "end_to_end_seconds": backward_done - started,
        "gradients_finite": gradients_finite,
        "gradient_l2_norms": gradient_norms,
    }


def _map_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().double() - expected.detach().double()
    norm = torch.linalg.vector_norm(difference, dim=-1)
    return {
        "maximum_vertex_error": float(norm.max().cpu()),
        "map_rmse": float(torch.sqrt(torch.mean(norm.square())).cpu()),
        "maximum_coordinate_error": float(difference.abs().max().cpu()),
    }


def _topology_summary(mesh, mapped: torch.Tensor) -> dict[str, Any]:
    values = mapped.detach().double().cpu().numpy()
    if values.ndim == 2:
        values = values[None]
    rows = [compute_p1_map_metrics(mesh, sample) for sample in values]
    return {
        "sample_count": len(rows),
        "all_certified": all(row.global_injectivity_certificate for row in rows),
        "maximum_flip_count": max(row.flip_count for row in rows),
        "minimum_signed_area": min(row.minimum_signed_area for row in rows),
        "minimum_area_ratio": min(row.minimum_area_ratio for row in rows),
        "minimum_boundary_gap": min(row.boundary_order_min_gap for row in rows),
        "samples": [
            {
                "certified": row.global_injectivity_certificate,
                "flip_count": row.flip_count,
                "minimum_signed_area": row.minimum_signed_area,
                "minimum_area_ratio": row.minimum_area_ratio,
                "boundary_order_min_gap": row.boundary_order_min_gap,
            }
            for row in rows
        ],
    }


def _global_neighbors(system) -> np.ndarray:
    result = np.full_like(system.neighbors, -1)
    rows, slots = np.nonzero(system.valid_mask)
    represented = system.neighbors[rows, slots]
    boundary = system.neighbor_is_boundary[rows, slots]
    result[rows[boundary], slots[boundary]] = system.loop[represented[boundary]]
    result[rows[~boundary], slots[~boundary]] = system.interior[represented[~boundary]]
    return result


def _directed_residual(system, logits: torch.Tensor, mapped: torch.Tensor) -> dict[str, float]:
    z = logits.detach()
    y = mapped.detach()
    if z.ndim == 2:
        z = z.unsqueeze(0)
    if y.ndim == 2:
        y = y.unsqueeze(0)
    mask = torch.as_tensor(system.valid_mask, dtype=torch.bool, device=z.device)
    probability = torch.softmax(z.masked_fill(~mask, -torch.inf), dim=-1).double().cpu().numpy()
    coordinates = y.double().cpu().numpy()
    neighbors = _global_neighbors(system)
    residual = coordinates[:, system.interior].copy()
    weighted = np.zeros_like(residual)
    for row, slot in zip(*np.nonzero(system.valid_mask)):
        value = probability[:, row, slot, None] * coordinates[:, neighbors[row, slot]]
        residual[:, row] -= value
        weighted[:, row] += value
    absolute = np.linalg.norm(residual, axis=-1)
    scale = np.maximum(
        np.maximum(np.linalg.norm(coordinates[:, system.interior], axis=-1), np.linalg.norm(weighted, axis=-1)),
        np.finfo(np.float64).tiny,
    )
    relative = absolute / scale
    return {
        "maximum_absolute_residual": float(np.max(absolute)),
        "root_mean_square_absolute_residual": float(np.sqrt(np.mean(absolute**2))),
        "maximum_relative_residual": float(np.max(relative)),
        "minimum_supported_probability": float(np.min(probability[:, system.valid_mask])),
    }


def _maximum_row_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().double(), dim=-1).max().cpu())


def _solver_diagnostics(solver) -> dict[str, Any] | None:
    diagnostics = getattr(solver, "last_diagnostics", None)
    if diagnostics is None:
        return None
    return {
        "forward": _plain(diagnostics.forward),
        "adjoint": _plain(diagnostics.adjoint),
    }


def run_case(config: BenchmarkConfig) -> dict[str, Any]:
    """Run one row and always return a strict-JSON-compatible receipt."""

    started = perf_counter()
    report: dict[str, Any] = {
        "schema": "phase5_route2_mvc_layer_case_v1",
        "status": "failure",
        "stage": "validation",
        "research_question": (
            "What are the synchronized forward/backward cost, memory, and first-order accuracy "
            "of the differentiable MVC encoder, full E_MVC o D layer, and canonical covariance retraction?"
        ),
        "config": asdict(config),
        "timing": {},
        "audit": {},
    }
    old_threads = torch.get_num_threads()
    try:
        device, dtype, backend, fd_epsilon = _validate_config(config)
        report["resolved_backend"] = backend
        torch.set_num_threads(config.threads)
        report["environment"] = _environment(device)
        report["stage"] = "setup"
        mesh = structured_rectangle(config.control_side - 1, config.control_side - 1)
        solver = _make_solver(
            mesh,
            backend,
            dtype,
            device,
            config.max_iterations,
            config.solver_rtol,
        )
        m1 = MVCCanonicalizationLayer(mesh, solver).to(device)
        m2 = CovarianceTutteRetraction(solver, encoder=m1.encoder).to(device)
        raw_logits, boundary, direction = _inputs(mesh, solver.system, config, dtype, device)
        report["parameter_count"] = int(raw_logits.numel() + boundary.numel())
        report["shapes"] = {
            "logits": list(raw_logits.shape),
            "boundary": list(boundary.shape),
            "control": [config.batch, mesh.n_vertices, 2],
            "direction": list(direction.shape),
        }
        report["solver_settings"] = (
            {
                "algorithm": "SuperLU",
                "public_dtype": config.dtype,
                "internal_dtype": "float64",
                "batch_execution": "sequential CPU",
            }
            if backend == "direct"
            else {
                "algorithm": "matrix-free directed BiCGStab with documented CUDA-float32 fallback",
                "relative_tolerance": solver.rtol,
                "relative_tolerance_source": (
                    "explicit" if config.solver_rtol is not None else "dtype_default"
                ),
                "absolute_tolerance": solver.atol,
                "maximum_iterations": solver.max_iter,
                "public_dtype": config.dtype,
            }
        )

        report["stage"] = "base_decode"
        with torch.no_grad():
            base_control = solver(raw_logits, boundary)
        latent_cotangent, control_cotangent = _cotangents(raw_logits, base_control)

        def encoder_iteration() -> dict[str, Any]:
            def forward():
                current = base_control.detach().clone().requires_grad_()
                return m1.encoder(current), (current,)

            return _timed_autograd(
                forward,
                lambda result: torch.mean(result.logits * latent_cotangent),
                device,
            )

        def m1_iteration() -> dict[str, Any]:
            def forward():
                logits = raw_logits.detach().clone().requires_grad_()
                fixed_boundary = boundary.detach().clone().requires_grad_()
                return m1(logits, fixed_boundary), (logits, fixed_boundary)

            return _timed_autograd(
                forward,
                lambda result: torch.mean(result.logits * latent_cotangent)
                + 0.01 * torch.mean(result.control * control_cotangent),
                device,
            )

        def m2_iteration() -> dict[str, Any]:
            def forward():
                logits = raw_logits.detach().clone().requires_grad_()
                fixed_boundary = boundary.detach().clone().requires_grad_()
                tangent = direction.detach().clone().requires_grad_()
                alpha = torch.tensor(
                    config.step_size, dtype=dtype, device=device, requires_grad=True
                )
                return m2(logits, fixed_boundary, tangent, alpha), (
                    logits,
                    fixed_boundary,
                    tangent,
                    alpha,
                )

            return _timed_autograd(
                forward,
                lambda result: torch.mean(result.updated_control * control_cotangent)
                + 0.01 * torch.mean(result.updated_control.square()),
                device,
            )

        for name, iteration in (
            ("encoder_only", encoder_iteration),
            ("full_m1", m1_iteration),
            ("canonical_m2", m2_iteration),
        ):
            report["stage"] = "measurement_" + name
            report["timing"][name] = _measure_stage(
                iteration,
                warmup=config.warmup,
                repeats=config.repeats,
                device=device,
            )

        report["stage"] = "audit_canonical"
        with torch.no_grad():
            encoded = m1.encoder(base_control)
            canonical_redecode = solver(encoded.logits, encoded.boundary)
        canonical_solver_diagnostics = _solver_diagnostics(solver)
        report["audit"]["canonical_redecode"] = _map_error(canonical_redecode, base_control)
        mask = torch.as_tensor(solver.system.valid_mask, dtype=torch.bool, device=device)
        maximums = torch.where(mask, encoded.logits, -torch.inf).amax(dim=-1)
        minimums = torch.where(mask, encoded.logits, torch.inf).amin(dim=-1)
        report["audit"]["mvc_diagnostics"] = {
            "maximum_barycentric_residual": float(
                encoded.diagnostics.barycentric_residual.max().detach().cpu()
            ),
            "maximum_covariance_condition": float(
                encoded.diagnostics.covariance_condition.max().detach().cpu()
            ),
            "minimum_edge_length": float(encoded.diagnostics.minimum_edge_length.min().detach().cpu()),
            "minimum_angle_sine": float(encoded.diagnostics.minimum_angle_sine.min().detach().cpu()),
            "maximum_winding_error": float(encoded.diagnostics.winding_error.max().detach().cpu()),
            "minimum_supported_probability": float(encoded.probabilities[:, mask].min().detach().cpu()),
            "maximum_canonical_logit_spread": float((maximums - minimums).max().detach().cpu()),
        }

        report["stage"] = "audit_m2"
        with torch.no_grad():
            zero = m2(raw_logits, boundary, direction, 0.0)
            finite = m2(raw_logits, boundary, direction, config.step_size)
            independently_decoded = solver(finite.updated_logits, finite.updated_boundary)
        finite_solver_diagnostics = _solver_diagnostics(solver)
        report["audit"]["m2_zero_step"] = {
            **{
                key + "_vs_base": value
                for key, value in _map_error(zero.updated_control, base_control).items()
            },
            "canonical_logit_maximum_error": float(
                (zero.updated_logits - zero.canonical_logits).abs().max().detach().cpu()
            ),
        }
        report["audit"]["m2_finite_step"] = {
            "step_size": config.step_size,
            "independent_redecode_maximum_vertex_error": _map_error(
                independently_decoded, finite.updated_control
            )["maximum_vertex_error"],
            "maximum_vertex_displacement_from_base": _map_error(
                finite.updated_control, base_control
            )["maximum_vertex_error"],
        }
        report["audit"]["m2_lift_diagnostics"] = {
            "maximum_linearized_residual": _maximum_row_norm(finite.lift.linearized_residual),
            "maximum_equilibrium_residual": _maximum_row_norm(finite.lift.equilibrium_residual),
            "maximum_covariance_condition": float(
                finite.lift.condition_number.detach().double().max().cpu()
            ),
        }

        report["stage"] = "audit_first_variation"
        with torch.no_grad():
            plus = m2(raw_logits, boundary, direction, fd_epsilon).updated_control
            minus = m2(raw_logits, boundary, direction, -fd_epsilon).updated_control
        finite_difference = (plus - minus) / (2.0 * fd_epsilon)
        variation_error = finite_difference.detach().double() - direction.detach().double()
        direction_norm = torch.linalg.vector_norm(direction.detach().double())
        fd_error_norm = torch.linalg.vector_norm(variation_error)

        alpha = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
        projected_output = m2(raw_logits, boundary, direction, alpha).updated_control
        projection = torch.sum(projected_output * control_cotangent)
        projected_derivative = torch.autograd.grad(projection, alpha)[0]
        expected_projection = torch.sum(direction * control_cotangent)
        projected_error = torch.abs(projected_derivative - expected_projection)
        projected_scale = torch.clamp_min(torch.abs(expected_projection), torch.finfo(dtype).tiny)
        report["audit"]["first_variation"] = {
            "finite_difference_epsilon": fd_epsilon,
            "finite_difference_maximum_vertex_error": float(
                torch.linalg.vector_norm(variation_error, dim=-1).max().cpu()
            ),
            "finite_difference_relative_l2_error": float((fd_error_norm / direction_norm).cpu()),
            "projected_jvp_definition": (
                "reverse-mode scalar-alpha derivative of <cotangent,R(alpha)>; "
                "compared with <cotangent,d>"
            ),
            "projected_jvp_absolute_error": float(projected_error.detach().cpu()),
            "projected_jvp_relative_error": float((projected_error / projected_scale).detach().cpu()),
        }

        report["stage"] = "audit_topology_residual"
        topology = {
            "base": _topology_summary(mesh, base_control),
            "canonical_redecode": _topology_summary(mesh, canonical_redecode),
            "m2_zero": _topology_summary(mesh, zero.updated_control),
            "m2_finite": _topology_summary(mesh, finite.updated_control),
        }
        report["audit"]["topology"] = topology
        report["audit"]["solver_residuals"] = {
            "base": _directed_residual(solver.system, raw_logits, base_control),
            "canonical_redecode": _directed_residual(
                solver.system, encoded.logits, canonical_redecode
            ),
            "m2_finite": _directed_residual(
                solver.system, finite.updated_logits, finite.updated_control
            ),
        }
        report["audit"]["solver_diagnostics"] = {
            "canonical_redecode": canonical_solver_diagnostics,
            "finite_independent_redecode": finite_solver_diagnostics,
            "last_first_variation_backward": _solver_diagnostics(solver),
        }
        if not all(item["all_certified"] for item in topology.values()):
            raise RuntimeError("an independently audited returned P1 map failed topology certification")

        report["metric_definitions"] = {
            "control_side": "number of control vertices per square side; total vertices=control_side^2",
            "timing": (
                "synchronized wall time; input cloning excluded; forward, scalar loss, and first-order "
                "reverse-mode backward reported separately"
            ),
            "canonical_redecode_error": "Euclidean vertex error between D(E(Y)) and Y",
            "finite_difference_first_variation": (
                "central [R(+epsilon)-R(-epsilon)]/(2 epsilon) compared to requested full vertex tangent d"
            ),
            "projected_jvp": (
                "one deterministic dual projection of dR/dalpha computed by ordinary reverse-mode autograd; "
                "not a full forward-mode JVP"
            ),
            "solver_relative_residual": (
                "row Euclidean barycentric residual divided by max(norm(Y_i),norm(sum_j p_ij Y_j),tiny)"
            ),
            "covariance_condition": "largest/smallest eigenvalue of each 2x2 row covariance",
            "gpu_memory": (
                "absolute PyTorch allocator peaks after warmup reset; reserved baseline includes cached warmup blocks"
            ),
        }
        report["semantics"] = {
            "encoder_only": "E_MVC on a detached accepted decoder map",
            "full_m1": "E_MVC o D using the same fixed directed decoder",
            "canonical_m2": (
                "D raw -> E_MVC canonical gauge -> covariance lift -> D updated; finite-step timing"
            ),
            "topology": (
                "independent floating P1 audit on each sample; this is not an exact-predicate continuum proof"
            ),
            "cuda": (
                "only matrix-free directed is allowed; solver topology screens still synchronize/copy to CPU"
            ),
            "differentiation": "first-order only; no higher-order derivative claim",
            "failure_rows": "a failed configuration retains its stage, config, completed timing/audit fields, and error",
        }
        _require_finite_measurements({"timing": report["timing"], "audit": report["audit"]})
        report.update(status="ok", stage="complete")
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        if hasattr(error, "report"):
            report["error"]["solver_report"] = _plain(error.report)
    finally:
        torch.set_num_threads(old_threads)
        report["harness_wall_seconds"] = perf_counter() - started
    return _plain(report)


def parse_profiles(value: str) -> list[tuple[int, int]]:
    """Parse ``control:batch`` pairs without silently dropping duplicates."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("profiles must be a nonempty comma-separated control:batch list")
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        pieces = item.strip().split(":")
        if len(pieces) != 2:
            raise ValueError("each profile must have control:batch form")
        try:
            pair = (int(pieces[0]), int(pieces[1]))
        except ValueError as error:
            raise ValueError("profile control and batch must be integers") from error
        if pair[0] < 1 or pair[1] < 1:
            raise ValueError("profile control and batch must be positive")
        result.append(pair)
    return result


def run_suite(configurations: list[BenchmarkConfig]) -> dict[str, Any]:
    started = perf_counter()
    rows = [run_case(config) for config in configurations]
    successes = sum(row["status"] == "ok" for row in rows)
    return _plain(
        {
            "schema": "phase5_route2_mvc_layer_suite_v1",
            "status": "ok" if successes == len(rows) else "complete_with_failures",
            "research_question": (
                "How do M1/M2 forward, backward, memory, and first-variation behavior scale over "
                "the declared control-side/batch/dtype/device matrix?"
            ),
            "environment": _environment(),
            "row_count": len(rows),
            "success_count": successes,
            "failure_count": len(rows) - successes,
            "rows": rows,
            "suite_wall_seconds": perf_counter() - started,
        }
    )


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _comma_values(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("comma-separated option must be nonempty")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _Parser(description=__doc__)
        parser.add_argument("--profiles", default="11:8,25:4,49:1")
        parser.add_argument("--dtypes", default="float32,float64")
        parser.add_argument("--devices", default="cpu")
        parser.add_argument("--backend", default="auto")
        parser.add_argument("--warmup", type=int, default=1)
        parser.add_argument("--repeats", type=int, default=2)
        parser.add_argument("--seed", type=int, default=2718)
        parser.add_argument("--strength", type=float, default=0.12)
        parser.add_argument("--step-size", type=float, default=0.05)
        parser.add_argument("--direction-scale", type=float, default=0.02)
        parser.add_argument("--fd-epsilon", type=float)
        parser.add_argument("--solver-rtol", type=float)
        parser.add_argument("--threads", type=int, default=1)
        parser.add_argument("--max-iterations", type=int, default=2000)
        parser.add_argument("--output", type=Path)
        arguments = parser.parse_args(argv)
        profiles = parse_profiles(arguments.profiles)
        dtypes = _comma_values(arguments.dtypes)
        devices = _comma_values(arguments.devices)
        configurations = [
            BenchmarkConfig(
                control_side=control,
                batch=batch,
                dtype=dtype,
                device=device,
                backend=arguments.backend,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
                seed=arguments.seed,
                strength=arguments.strength,
                step_size=arguments.step_size,
                direction_scale=arguments.direction_scale,
                fd_epsilon=arguments.fd_epsilon,
                solver_rtol=arguments.solver_rtol,
                threads=arguments.threads,
                max_iterations=arguments.max_iterations,
            )
            for device in devices
            for dtype in dtypes
            for control, batch in profiles
        ]
        receipt = run_suite(configurations)
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        code = 0 if receipt["failure_count"] == 0 else 2
    except Exception as error:
        receipt = {
            "schema": "phase5_route2_mvc_layer_suite_v1",
            "status": "argument_failure",
            "row_count": 0,
            "success_count": 0,
            "failure_count": 1,
            "rows": [],
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        code = 2
    print(json.dumps(_plain(receipt), allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
