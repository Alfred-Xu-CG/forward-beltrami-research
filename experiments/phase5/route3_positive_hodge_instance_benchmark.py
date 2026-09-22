"""Route III-H--K fixed-boundary positive-Hodge/Tutte instance benchmark.

Every optimized P1 evaluation follows the declared chain

``w -> mu(w) -> A(mu) -> frozen learned positive local map -> shared edges
   -> symmetric CG Tutte solve -> fixed dense P1 warp -> loss``.

With ``steps=41`` the optimized methods use exactly 42 completed primal and
41 completed adjoint solves, hence 83 budgeted global solves.  Target setup and
the independent final re-decode are counted separately.  The one-shot QC
projection row is explicitly ineligible for equal-budget ranking.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np

# Safe Windows/Conda startup order; intentionally not KMP_DUPLICATE_LIB_OK.
_NUMPY_LINALG_BOOTSTRAP = float(np.linalg.cond(np.eye(1, dtype=np.float64)))

import scipy
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from qcopt.beltrami import face_beltrami  # noqa: E402
from qcopt.mesh import structured_rectangle  # noqa: E402
from qcopt.neural_bijection.benchmarks import synthetic_image, warp_image_backward  # noqa: E402
from qcopt.neural_bijection.metrics import compute_p1_map_metrics  # noqa: E402
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable  # noqa: E402
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer  # noqa: E402
from qcopt.neural_bijection.tutte.instance_optimization import build_directed_target  # noqa: E402
from qcopt.neural_bijection.tutte.iterative import MatrixFreeDirectedTutteLayer  # noqa: E402
from qcopt.neural_bijection.tutte.positive_hodge import (  # noqa: E402
    LearnedPositiveDirectionMap,
    PositiveHodgeTutteLayer,
    beltrami_tensor,
    build_standard_square_graph,
    inverse_softplus_conductances,
)


METHOD_NAMES = (
    "diagnostic_O2_row_softmax_directed",
    "P1_positive_uniform",
    "P1_positive_qc_initialized",
    "P1_positive_qc_projection_only",
)
MAXIMUM_MU_RADIUS = 0.95
P1_CHAIN = (
    "latent_w",
    "mu_in_unit_disk",
    "A(mu)_spd",
    "learned_softplus_positive_local_map",
    "shared_edge_average",
    "symmetric_spd_tutte_solve",
    "dense_p1_warp",
    "loss",
)


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ("git", "-C", str(REPOSITORY), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ("git", "-C", str(REPOSITORY), "status", "--porcelain"),
        capture_output=True,
        text=True,
        check=False,
    )
    value = commit.stdout.strip().lower()
    return {
        "commit": value if commit.returncode == 0 and len(value) == 40 else None,
        "dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def _process_memory() -> dict[str, int | None]:
    path = Path("/proc/self/status")
    result: dict[str, int | None] = {"rss_bytes": None, "hwm_bytes": None}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            result["rss_bytes"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            result["hwm_bytes"] = int(line.split()[1]) * 1024
    return result


def _environment(device: torch.device) -> dict[str, Any]:
    result = {
        **_git_state(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "cuda_runtime": torch.version.cuda,
                "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return result


def _mesh_summary(mesh) -> dict[str, Any]:
    return {
        "kind": "standard_square_fixed_sw_ne_diagonal",
        "vertices": mesh.n_vertices,
        "faces": mesh.n_faces,
        "vertex_shape": list(mesh.vertices.shape),
        "face_shape": list(mesh.faces.shape),
        "vertex_sum": float(np.sum(mesh.vertices)),
        "vertex_l2": float(np.linalg.norm(mesh.vertices)),
        "face_index_sum": int(np.sum(mesh.faces)),
        "boundary_loop": mesh.boundary_loops[0].tolist(),
    }


def _target_summary(target, boundary: torch.Tensor) -> dict[str, Any]:
    control = target.control.detach().to(dtype=torch.float64, device="cpu")
    dense = target.dense.detach().to(dtype=torch.float64, device="cpu")
    return {
        "control_shape": list(control.shape),
        "dense_shape": list(dense.shape),
        "control_sum": float(control.sum()),
        "control_l2": float(torch.linalg.vector_norm(control)),
        "control_min": float(control.min()),
        "control_max": float(control.max()),
        "dense_sum": float(dense.sum()),
        "dense_l2": float(torch.linalg.vector_norm(dense)),
        "dense_min": float(dense.min()),
        "dense_max": float(dense.max()),
        "boundary": boundary.detach().to(dtype=torch.float64, device="cpu").tolist(),
        # build_directed_target has no second target, so its four accuracy
        # fields are intentionally undefined (NaN in P1MapMetrics).  Omit them
        # explicitly rather than silently converting nonfinite values to null.
        "metrics": {
            "accuracy_fields_defined": False,
            "flip_count": target.metrics.flip_count,
            "minimum_signed_area": target.metrics.minimum_signed_area,
            "minimum_area_ratio": target.metrics.minimum_area_ratio,
            "boundary_order_min_gap": target.metrics.boundary_order_min_gap,
            "global_injectivity_certificate": target.metrics.global_injectivity_certificate,
        },
    }


def _audit_clean_target_authority(
    task: str, current: dict[str, Any], *, required: bool
) -> dict[str, Any]:
    """Fail closed against the committed clean Route-II O1 target identity."""

    if not required:
        return {
            "required": False,
            "matched": None,
            "reason": "only mandatory for the preregistered N25/R256 formal cohort",
        }
    path = REPOSITORY / (
        "docs/research_phase5/raw_results/"
        f"route2_mvc_instance_formal_{task}_gpu_O1_ai_74b452c_clean.json"
    )
    if not path.is_file():
        raise RuntimeError(f"clean Route-II target authority is missing: {path}")
    authority = json.loads(path.read_text(encoding="utf-8"))
    environment = authority.get("environment", {})
    if (
        authority.get("status") != "ok"
        or environment.get("commit") != "74b452c39c39601bf5d1204288307e9a02a2aaa5"
        or environment.get("dirty") is not False
    ):
        raise RuntimeError("clean Route-II target authority has invalid provenance")
    numeric = authority["shared_target"]["numeric_identity"]
    expected = {
        "control_shape": numeric["control"]["shape"],
        "control_sum": numeric["control"]["sum"],
        "control_l2": numeric["control"]["l2"],
        "control_min": numeric["control"]["minimum"],
        "control_max": numeric["control"]["maximum"],
        "dense_shape": numeric["dense"]["shape"],
        "dense_sum": numeric["dense"]["sum"],
        "dense_l2": numeric["dense"]["l2"],
        "dense_min": numeric["dense"]["minimum"],
        "dense_max": numeric["dense"]["maximum"],
    }
    comparisons: dict[str, Any] = {}
    matched = True
    for key, expected_value in expected.items():
        actual_value = current.get(key)
        if key.endswith("_shape"):
            field_match = list(actual_value) == list(expected_value)
            difference = None
            tolerance = None
        else:
            difference = abs(float(actual_value) - float(expected_value))
            tolerance = 1.0e-12 * max(1.0, abs(float(expected_value)))
            field_match = difference <= tolerance
        comparisons[key] = {
            "actual": actual_value,
            "expected": expected_value,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "matched": field_match,
        }
        matched = matched and field_match
    if not matched:
        failed = [key for key, row in comparisons.items() if not row["matched"]]
        raise RuntimeError(f"target authority mismatch in fields: {failed}")
    return {
        "required": True,
        "matched": True,
        "authority_path": str(path.relative_to(REPOSITORY)).replace("\\", "/"),
        "authority_commit": environment["commit"],
        "authority_dirty": environment["dirty"],
        "comparisons": comparisons,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _w_to_mu(raw_w: torch.Tensor) -> torch.Tensor:
    if raw_w.shape[-1] != 2:
        raise ValueError("raw w must have trailing dimension two")
    squared = raw_w.square().sum(dim=-1, keepdim=True)
    tiny = torch.finfo(raw_w.dtype).tiny
    radius = torch.sqrt(squared + tiny)
    scale = MAXIMUM_MU_RADIUS * torch.tanh(radius) / radius
    return scale * raw_w


def _mu_to_tensor_torch(mu_components: torch.Tensor) -> torch.Tensor:
    real = mu_components[..., 0]
    imaginary = mu_components[..., 1]
    squared = real.square() + imaginary.square()
    denominator = 1.0 - squared
    if bool(torch.any(denominator <= 0.0)):
        raise RuntimeError("latent mu left the unit disk")
    result = torch.empty(mu_components.shape[:-1] + (2, 2), dtype=mu_components.dtype, device=mu_components.device)
    result[..., 0, 0] = (1.0 - 2.0 * real + squared) / denominator
    result[..., 0, 1] = -2.0 * imaginary / denominator
    result[..., 1, 0] = result[..., 0, 1]
    result[..., 1, 1] = (1.0 + 2.0 * real + squared) / denominator
    return result


def _encode_mu_as_w(values: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mu = np.asarray(values, dtype=np.complex128)
    radius = np.abs(mu)
    if np.any(~np.isfinite(radius)) or np.any(radius >= MAXIMUM_MU_RADIUS):
        raise ValueError(
            f"target face mu must have modulus strictly below {MAXIMUM_MU_RADIUS}"
        )
    raw_radius = np.arctanh(radius / MAXIMUM_MU_RADIUS)
    unit = np.zeros(mu.shape + (2,), dtype=np.float64)
    nonzero = radius > 0.0
    unit[..., 0][nonzero] = mu.real[nonzero] / radius[nonzero]
    unit[..., 1][nonzero] = mu.imag[nonzero] / radius[nonzero]
    return torch.tensor(raw_radius[..., None] * unit, dtype=dtype, device=device)


def _train_projector(
    directions: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    hidden_features: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[LearnedPositiveDirectionMap, dict[str, Any]]:
    model = LearnedPositiveDirectionMap(
        directions,
        hidden_features=hidden_features,
        minimum_conductance=1.0e-6,
        seed=seed,
    ).to(device=device, dtype=dtype)
    radii = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)
    angles = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
    mus = np.asarray(
        [radius * np.exp(1j * angle) for radius in radii for angle in angles],
        dtype=np.complex128,
    )
    target = torch.tensor(beltrami_tensor(mus), dtype=dtype, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start = perf_counter()
    initial_loss = final_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        coefficients = model(target)
        fitted = model.fitted_tensors(coefficients)
        loss = (
            (fitted - target).square().sum(dim=(-2, -1))
            / target.square().sum(dim=(-2, -1))
        ).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("projector training produced a nonfinite loss")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    _synchronize(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "hidden_features": hidden_features,
        "training_samples": len(target),
        "training_radii": list(radii),
        "training_angles": 180,
        "training_angle_spacing_degrees": 2.0,
        "initial_relative_frobenius_squared_loss": initial_loss,
        "final_relative_frobenius_squared_loss": final_loss,
        "wall_seconds": perf_counter() - start,
        "state_dict": {
            name: value.detach().to(dtype=torch.float64, device="cpu").tolist()
            for name, value in model.state_dict().items()
        },
    }


def _loss(
    task: str,
    dense: torch.Tensor,
    target_dense: torch.Tensor,
    *,
    moving: torch.Tensor | None,
    fixed: torch.Tensor | None,
) -> torch.Tensor:
    if task == "map":
        return (dense - target_dense).square().mean()
    if moving is None or fixed is None:
        raise RuntimeError("image task requires moving and fixed images")
    tolerance = 64.0 * torch.finfo(dense.dtype).eps
    if bool(torch.any(dense < -tolerance)) or bool(torch.any(dense > 1.0 + tolerance)):
        raise RuntimeError("dense backward map left the unit image domain")
    return (warp_image_backward(moving, dense) - fixed).square().mean()


def _directed_krylov(report) -> dict[str, Any] | None:
    if report is None:
        return None
    iterations = report.iterations.detach().cpu()
    relative = report.relative_residual.detach().cpu()
    converged = report.converged.detach().cpu()
    return {
        "method": report.method,
        "iterations": iterations.tolist(),
        "maximum_iterations": int(iterations.max()) if iterations.numel() else 0,
        "relative_residual": relative.tolist(),
        "maximum_relative_residual": float(relative.max()) if relative.numel() else 0.0,
        "all_converged": bool(converged.all()),
        "primary_failure": report.primary_failure,
    }


def _symmetric_krylov(stats) -> dict[str, Any] | None:
    if stats is None:
        return None
    return {
        "method": "conjugate_gradient",
        "iterations": stats.iterations,
        "maximum_iterations": stats.iterations,
        "relative_residual": stats.relative_residual,
        "maximum_relative_residual": stats.relative_residual,
        "all_converged": stats.converged,
        "primary_failure": None,
    }


def _aggregate_krylov(trace: list[dict[str, Any]]) -> dict[str, Any]:
    forward = [row["forward_krylov"] for row in trace if row["forward_krylov"] is not None]
    adjoint = [row["adjoint_krylov"] for row in trace if row["adjoint_krylov"] is not None]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "solve_count": len(rows),
            "maximum_iterations": max((row["maximum_iterations"] for row in rows), default=0),
            "maximum_relative_residual": max(
                (row["maximum_relative_residual"] for row in rows), default=0.0
            ),
            "all_converged": all(row["all_converged"] for row in rows),
            "methods": sorted({row["method"] for row in rows}),
        }

    return {"forward": summarize(forward), "adjoint": summarize(adjoint)}


def _extract_protocol_slices(
    trace: list[dict[str, Any]], *, primary_global_solves: int = 83
) -> dict[str, Any]:
    exact = next(
        (
            row
            for row in trace
            if row["observation_completed_global_solves"] == primary_global_solves
        ),
        None,
    )

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        position = trace.index(row)
        observed_rows = trace[: position + 1]
        completed_update_rows = trace[:position]
        return {
            "iteration": row["iteration"],
            "objective": row["objective"],
            "map_rmse": row["map_rmse"],
            "mu_rmse": row["mu_rmse"],
            "maximum_mu_error": row["maximum_mu_error"],
            "minimum_area_ratio": row["minimum_area_ratio"],
            "topology_certified": row["topology_certified"],
            "completed_global_solves": row["observation_completed_global_solves"],
            "observation_wall_seconds": row["observation_wall_seconds"],
            "cumulative_timing_at_observation": {
                "forward_seconds": math.fsum(
                    item.get("forward_seconds", 0.0) for item in observed_rows
                ),
                "solver_seconds": math.fsum(
                    item.get("solver_seconds", 0.0) for item in observed_rows
                ),
                "dense_interpolation_seconds": math.fsum(
                    item.get("dense_interpolation_seconds", 0.0)
                    for item in observed_rows
                ),
                "task_objective_seconds": math.fsum(
                    item.get("task_objective_seconds", 0.0) for item in observed_rows
                ),
                "audit_seconds": math.fsum(
                    item.get("audit_seconds", 0.0) for item in observed_rows
                ),
                "backward_seconds": math.fsum(
                    item.get("backward_seconds", 0.0)
                    for item in completed_update_rows
                ),
                "optimizer_step_seconds": math.fsum(
                    item.get("optimizer_step_seconds", 0.0)
                    for item in completed_update_rows
                ),
            },
        }

    return {
        "primary_exact_global_solves": {
            "requested": primary_global_solves,
            "missing": exact is None,
            "row": compact(exact) if exact is not None else None,
        },
        "final_observation": compact(trace[-1]),
    }


def _independent_direct_redecode(
    *,
    name: str,
    mesh,
    graph,
    final_state: dict[str, Any],
    accepted_control: torch.Tensor,
    boundary: torch.Tensor,
    resolution: int,
    dtype: torch.dtype,
    target_control: np.ndarray,
) -> dict[str, Any]:
    """Reassemble and solve a CPU-float64 authority from serialized state."""

    boundary64 = boundary.detach().to(dtype=torch.float64, device="cpu")
    boundary_numpy = boundary64.numpy()
    if name == "diagnostic_O2_row_softmax_directed":
        logits = np.asarray(final_state["interior_logits"], dtype=np.float64)
        direct_layer = DirectTutteLayer(mesh)
        with torch.no_grad():
            control64 = direct_layer(torch.tensor(logits), boundary64)
        probabilities = direct_layer.system._probabilities(logits)
        rows = direct_layer.system.n_rows
        matrix = np.eye(rows, dtype=np.float64)
        rhs = np.zeros((rows, 2), dtype=np.float64)
        for row in range(rows):
            for slot in np.flatnonzero(direct_layer.system.valid_mask[row]):
                probability = probabilities[row, slot]
                neighbor = direct_layer.system.neighbors[row, slot]
                if direct_layer.system.neighbor_is_boundary[row, slot]:
                    rhs[row] += probability * boundary_numpy[neighbor]
                else:
                    matrix[row, neighbor] -= probability
        authority = "cpu_float64_superlu_directed_reassembly_from_final_logits"
    else:
        conductances = np.asarray(final_state["edge_conductances"], dtype=np.float64)
        if conductances.shape != (len(graph.active_edges),) or np.any(conductances <= 0.0):
            raise RuntimeError("serialized P1 edge conductances are invalid")
        interior = np.setdiff1d(
            np.arange(mesh.n_vertices, dtype=np.int64), mesh.boundary_loops[0]
        )
        interior_index = {int(vertex): row for row, vertex in enumerate(interior)}
        boundary_index = {
            int(vertex): row for row, vertex in enumerate(mesh.boundary_loops[0])
        }
        matrix = np.zeros((len(interior), len(interior)), dtype=np.float64)
        rhs = np.zeros((len(interior), 2), dtype=np.float64)
        for value, (first, second) in zip(conductances, graph.active_edges, strict=True):
            first_row = interior_index.get(int(first))
            second_row = interior_index.get(int(second))
            if first_row is not None and second_row is not None:
                matrix[first_row, first_row] += value
                matrix[second_row, second_row] += value
                matrix[first_row, second_row] -= value
                matrix[second_row, first_row] -= value
            else:
                row, boundary_vertex = (
                    (first_row, int(second))
                    if first_row is not None
                    else (second_row, int(first))
                )
                assert row is not None
                matrix[row, row] += value
                rhs[row] += value * boundary_numpy[boundary_index[boundary_vertex]]
        interior_values = np.linalg.solve(matrix, rhs)
        mapped = np.empty((mesh.n_vertices, 2), dtype=np.float64)
        mapped[mesh.boundary_loops[0]] = boundary_numpy
        mapped[interior] = interior_values
        control64 = torch.tensor(mapped, dtype=torch.float64)
        authority = "cpu_float64_dense_spd_reassembly_from_final_edge_conductances"

    accepted64 = accepted_control.detach().to(dtype=torch.float64, device="cpu")
    system = direct_layer.system if name == "diagnostic_O2_row_softmax_directed" else None
    interior_vertices = (
        system.interior
        if system is not None
        else np.setdiff1d(
            np.arange(mesh.n_vertices, dtype=np.int64), mesh.boundary_loops[0]
        )
    )
    accepted_interior = accepted64.numpy()[interior_vertices]
    residual = matrix @ accepted_interior - rhs
    rhs_norms = np.linalg.norm(rhs, axis=0)
    residual_norms = np.linalg.norm(residual, axis=0)
    relative_residuals = np.where(
        rhs_norms > 0.0, residual_norms / rhs_norms, residual_norms
    )
    condition = float(np.linalg.cond(matrix)) if len(matrix) else 1.0
    direct_interior = control64.numpy()[interior_vertices]
    direct_norms = np.linalg.norm(direct_interior, axis=0)
    component_bounds = condition * relative_residuals * np.maximum(direct_norms, 1.0)
    roundoff = 512.0 * torch.finfo(dtype).eps * max(
        1.0, float(torch.linalg.vector_norm(control64))
    )
    agreement_bound = float(np.max(component_bounds) + roundoff)
    maximum_error = float(
        torch.linalg.vector_norm(control64 - accepted64, dim=-1).max()
    )
    if maximum_error > agreement_bound:
        raise RuntimeError(
            "accepted iterative map disagrees with independent CPU direct authority: "
            f"error={maximum_error:.3e}, bound={agreement_bound:.3e}"
        )
    metrics = compute_p1_map_metrics(mesh, control64.numpy(), target=target_control)
    if not metrics.global_injectivity_certificate:
        raise RuntimeError("independent CPU direct authority failed topology certification")
    query64 = StructuredDenseQueryTable.from_mesh(
        mesh, height=resolution, width=resolution
    )
    query64.prepare(device="cpu", dtype=torch.float64)
    dense64 = query64.interpolate(control64)
    return {
        "authority": authority,
        "control": control64.tolist(),
        "dense_numeric_identity": {
            "shape": list(dense64.shape),
            "sum": float(dense64.sum()),
            "l2": float(torch.linalg.vector_norm(dense64)),
            "minimum": float(dense64.min()),
            "maximum": float(dense64.max()),
        },
        "maximum_vertex_error": maximum_error,
        "represented_relative_residual_by_coordinate": relative_residuals.tolist(),
        "interior_system_condition_2norm": condition,
        "forward_error_agreement_bound": agreement_bound,
        "agreement_passed": True,
        "metrics": asdict(metrics),
        "topology_certified": metrics.global_injectivity_certificate,
        "krylov": None,
    }


def _method_memory_start(device: torch.device) -> tuple[dict[str, int | None], int | None, int | None]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
    else:
        allocated = reserved = None
    return _process_memory(), allocated, reserved


def _method_memory_end(
    device: torch.device,
    before: dict[str, int | None],
    baseline_allocated: int | None,
    baseline_reserved: int | None,
) -> dict[str, Any]:
    _synchronize(device)
    after = _process_memory()
    return {
        "rss_before_bytes": before["rss_bytes"],
        "rss_after_bytes": after["rss_bytes"],
        "process_hwm_bytes": after["hwm_bytes"],
        "cuda_baseline_allocated_bytes": baseline_allocated,
        "cuda_baseline_reserved_bytes": baseline_reserved,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }


def _run_method(
    *,
    name: str,
    mesh,
    graph,
    target,
    boundary: torch.Tensor,
    query: StructuredDenseQueryTable,
    task: str,
    steps: int,
    learning_rate: float,
    objective_threshold: float,
    projector_setup_seconds: float,
    dtype: torch.dtype,
    device: torch.device,
    projector: LearnedPositiveDirectionMap | None,
    moving: torch.Tensor | None,
    fixed: torch.Tensor | None,
) -> dict[str, Any]:
    projection_only = name == "P1_positive_qc_projection_only"
    method_steps = 0 if projection_only else steps
    target_dense = target.dense.detach().to(device=device, dtype=dtype)
    target_control = target.control.detach().to(device="cpu", dtype=torch.float64).numpy()
    if name == "diagnostic_O2_row_softmax_directed":
        solver: Any = MatrixFreeDirectedTutteLayer(mesh).to(device=device)
        latent = torch.nn.Parameter(
            torch.zeros(
                (solver.system.n_rows, solver.system.max_degree),
                dtype=dtype,
                device=device,
            )
        )
        chain = (
            "directed_row_logits",
            "positive_row_softmax",
            "directed_bicgstab_tutte_solve",
            "dense_p1_warp",
            "loss",
        )

        def decode_control() -> tuple[torch.Tensor, dict[str, Any] | None]:
            control = solver(latent, boundary)
            diagnostics = solver.last_diagnostics
            return control, _directed_krylov(
                diagnostics.forward if diagnostics is not None else None
            )

        def adjoint_diagnostics() -> dict[str, Any] | None:
            diagnostics = solver.last_diagnostics
            return _directed_krylov(
                diagnostics.adjoint if diagnostics is not None else None
            )

        def state() -> dict[str, Any]:
            return {
                "interior_logits": latent.detach().to(dtype=torch.float64, device="cpu").tolist()
            }

    else:
        if projector is None:
            raise RuntimeError("P1 method requires the trained positive projector")
        solver = PositiveHodgeTutteLayer(
            graph,
            projector,
            minimum_conductance=projector.minimum_conductance,
            relative_tolerance=1.0e-11 if dtype == torch.float64 else 1.0e-5,
            max_iterations=1000,
        ).to(device=device)
        if name == "P1_positive_uniform":
            initial_w = torch.zeros((mesh.n_faces, 2), dtype=dtype, device=device)
        else:
            target_mu = face_beltrami(mesh, target_control)
            initial_w = _encode_mu_as_w(target_mu, dtype=dtype, device=device)
        latent = torch.nn.Parameter(initial_w)
        chain = P1_CHAIN

        def decode_control() -> tuple[torch.Tensor, dict[str, Any] | None]:
            mu = _w_to_mu(latent)
            tensors = _mu_to_tensor_torch(mu)
            control = solver(tensors, boundary)
            return control, _symmetric_krylov(solver.solver.last_forward_stats)

        def adjoint_diagnostics() -> dict[str, Any] | None:
            return _symmetric_krylov(solver.solver.last_adjoint_stats)

        def state() -> dict[str, Any]:
            with torch.no_grad():
                mu = _w_to_mu(latent)
                tensors = _mu_to_tensor_torch(mu)
                edge_values = solver.project_edge_conductances(tensors)
                edge_logits = inverse_softplus_conductances(
                    edge_values, solver.solver.minimum_conductance
                )
            return {
                "latent_w": latent.detach().to(dtype=torch.float64, device="cpu").tolist(),
                "face_mu_components": mu.detach().to(dtype=torch.float64, device="cpu").tolist(),
                "edge_conductances": edge_values.detach().to(dtype=torch.float64, device="cpu").tolist(),
                "edge_logits": edge_logits.detach().to(dtype=torch.float64, device="cpu").tolist(),
            }

    optimizer = torch.optim.Adam((latent,), lr=learning_rate) if method_steps else None
    memory_before, baseline_allocated, baseline_reserved = _method_memory_start(device)
    trace: list[dict[str, Any]] = []
    primal_attempts = adjoint_attempts = 0
    primal_completed = adjoint_completed = 0
    start = perf_counter()
    final_control: torch.Tensor | None = None
    final_dense: torch.Tensor | None = None
    final_metrics = None

    for iteration in range(method_steps + 1):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        _synchronize(device)
        forward_start = perf_counter()
        primal_attempts += 1
        solver_start = perf_counter()
        control, forward_krylov = decode_control()
        primal_completed += 1
        _synchronize(device)
        solver_seconds = perf_counter() - solver_start
        dense_start = perf_counter()
        dense = query.interpolate(control)
        _synchronize(device)
        dense_interpolation_seconds = perf_counter() - dense_start
        objective_start = perf_counter()
        objective = _loss(
            task,
            dense,
            target_dense,
            moving=moving,
            fixed=fixed,
        )
        _synchronize(device)
        task_objective_seconds = perf_counter() - objective_start
        forward_seconds = perf_counter() - forward_start
        audit_start = perf_counter()
        mapped = control.detach().to(dtype=torch.float64, device="cpu").numpy()
        metrics = compute_p1_map_metrics(mesh, mapped, target=target_control)
        audit_seconds = perf_counter() - audit_start
        if not metrics.global_injectivity_certificate:
            raise RuntimeError("an accepted decoder output failed the independent P1 certificate")
        backward_seconds = 0.0
        optimizer_step_seconds = 0.0
        adjoint_krylov = None
        observation_primal = primal_completed
        observation_adjoint = adjoint_completed
        observation_wall = perf_counter() - start
        if iteration < method_steps:
            adjoint_attempts += 1
            _synchronize(device)
            backward_start = perf_counter()
            objective.backward()
            _synchronize(device)
            backward_seconds = perf_counter() - backward_start
            adjoint_completed += 1
            adjoint_krylov = adjoint_diagnostics()
            if latent.grad is None or not bool(torch.isfinite(latent.grad).all()):
                raise RuntimeError("latent gradient is absent or nonfinite")
            _synchronize(device)
            optimizer_start = perf_counter()
            assert optimizer is not None
            optimizer.step()
            _synchronize(device)
            optimizer_step_seconds = perf_counter() - optimizer_start
        trace.append(
            {
                "iteration": iteration,
                "objective": float(objective.detach().cpu()),
                "map_rmse": metrics.map_rmse,
                "mu_rmse": metrics.mu_rmse,
                "maximum_mu_error": metrics.maximum_mu_error,
                "minimum_area_ratio": metrics.minimum_area_ratio,
                "topology_certified": metrics.global_injectivity_certificate,
                "completed_primal_solves": primal_completed,
                "completed_adjoint_solves": adjoint_completed,
                "completed_global_solves": primal_completed + adjoint_completed,
                "observation_completed_primal_solves": observation_primal,
                "observation_completed_adjoint_solves": observation_adjoint,
                "observation_completed_global_solves": observation_primal + observation_adjoint,
                "forward_seconds": forward_seconds,
                "solver_seconds": solver_seconds,
                "dense_interpolation_seconds": dense_interpolation_seconds,
                "task_objective_seconds": task_objective_seconds,
                "backward_seconds": backward_seconds,
                "audit_seconds": audit_seconds,
                "optimizer_step_seconds": optimizer_step_seconds,
                "wall_seconds": perf_counter() - start,
                "observation_wall_seconds": observation_wall,
                "forward_krylov": forward_krylov,
                "adjoint_krylov": adjoint_krylov,
                "accepted_from_decoder": True,
            }
        )
        final_control = control
        final_dense = dense
        final_metrics = metrics

    assert final_control is not None and final_dense is not None and final_metrics is not None
    budget_wall = perf_counter() - start
    final_state = state()
    final_state.update(
        {
            "boundary": boundary.detach().to(dtype=torch.float64, device="cpu").tolist(),
            "control": final_control.detach().to(dtype=torch.float64, device="cpu").tolist(),
        }
    )
    # A separately assembled CPU-float64 direct authority is intentionally
    # outside the optimization budget.  It consumes the serialized final state
    # rather than reusing the matrix-free decode closure.
    independent_redecode = _independent_direct_redecode(
        name=name,
        mesh=mesh,
        graph=graph,
        final_state=final_state,
        accepted_control=final_control,
        boundary=boundary,
        resolution=query.height,
        dtype=dtype,
        target_control=target_control,
    )
    memory = _method_memory_end(
        device, memory_before, baseline_allocated, baseline_reserved
    )
    threshold_row = next(
        (row for row in trace if row["objective"] <= objective_threshold), None
    )
    threshold_wall = (
        float(threshold_row["observation_wall_seconds"])
        if threshold_row is not None
        else None
    )
    threshold_total_wall = (
        projector_setup_seconds + threshold_wall if threshold_wall is not None else None
    )
    if projection_only:
        protocol_kind = "projection_only"
    elif name == "P1_positive_qc_initialized":
        protocol_kind = "oracle_initialized_optimized_upper_bound"
    elif name == "diagnostic_O2_row_softmax_directed":
        protocol_kind = "diagnostic_row_softmax_O2"
    else:
        protocol_kind = "optimized_instance"
    rank_eligible = name == "P1_positive_uniform"
    return {
        "name": name,
        "protocol_kind": protocol_kind,
        "status": "success",
        "equal_budget_rank_eligible": rank_eligible,
        "chain": list(chain),
        "initial_state": {
            "objective": trace[0]["objective"],
            "map_rmse": trace[0]["map_rmse"],
            "mu_rmse": trace[0]["mu_rmse"],
        },
        "final_state": final_state,
        "trace": trace,
        "attempted_primal_solves": primal_attempts,
        "attempted_adjoint_solves": adjoint_attempts,
        "completed_primal_solves": primal_completed,
        "completed_adjoint_solves": adjoint_completed,
        "completed_global_solves": primal_completed + adjoint_completed,
        "verification_primal_solves": 1,
        "total_actual_primal_solves_including_verification": primal_completed + 1,
        "total_actual_global_solves_including_verification": primal_completed + adjoint_completed + 1,
        "best_objective": min(row["objective"] for row in trace),
        "final_objective": trace[-1]["objective"],
        "objective_threshold": objective_threshold,
        "evaluations_to_threshold": (
            int(threshold_row["iteration"]) + 1 if threshold_row is not None else None
        ),
        "iterations_to_threshold": (
            int(threshold_row["iteration"]) if threshold_row is not None else None
        ),
        "global_solves_to_threshold": (
            int(threshold_row["observation_completed_global_solves"])
            if threshold_row is not None
            else None
        ),
        "wall_seconds_to_threshold": threshold_wall,
        "wall_seconds": budget_wall,
        "time_to_useful": {
            "projector_setup_seconds": projector_setup_seconds,
            "method_budget_wall_seconds": budget_wall,
            "projector_setup_plus_method_wall_seconds": projector_setup_seconds + budget_wall,
            "projector_setup_plus_threshold_wall_seconds": threshold_total_wall,
            "allocation_semantics": (
                "full deterministic projector setup assigned to every P1 row as if run in a fresh process"
                if name.startswith("P1_")
                else "no projector setup required"
            ),
        },
        "timing": {
            "forward_seconds_total": math.fsum(row["forward_seconds"] for row in trace),
            "solver_seconds_total": math.fsum(row["solver_seconds"] for row in trace),
            "dense_interpolation_seconds_total": math.fsum(
                row["dense_interpolation_seconds"] for row in trace
            ),
            "task_objective_seconds_total": math.fsum(
                row["task_objective_seconds"] for row in trace
            ),
            "backward_seconds_total": math.fsum(row["backward_seconds"] for row in trace),
            "audit_seconds_total": math.fsum(row["audit_seconds"] for row in trace),
            "optimizer_step_seconds_total": math.fsum(
                row["optimizer_step_seconds"] for row in trace
            ),
            "budget_wall_seconds": budget_wall,
        },
        "krylov": _aggregate_krylov(trace),
        "protocol_slices": _extract_protocol_slices(trace),
        "all_iterates_topology_certified": all(
            row["topology_certified"] for row in trace
        ),
        "final_metrics": asdict(final_metrics),
        "independent_redecode": independent_redecode,
        "memory": memory,
    }


def _failure_row(name: str, error: Exception) -> dict[str, Any]:
    if name == "P1_positive_qc_projection_only":
        protocol_kind = "projection_only"
    elif name == "P1_positive_qc_initialized":
        protocol_kind = "oracle_initialized_optimized_upper_bound"
    elif name == "diagnostic_O2_row_softmax_directed":
        protocol_kind = "diagnostic_row_softmax_O2"
    else:
        protocol_kind = "optimized_instance"
    return {
        "name": name,
        "status": "failure",
        "protocol_kind": protocol_kind,
        "equal_budget_rank_eligible": False,
        "failure_type": type(error).__name__,
        "failure_message": str(error),
    }


def run_benchmark(
    *,
    task: str,
    methods: Iterable[str],
    control_side: int,
    resolution: int,
    steps: int,
    learning_rate: float,
    seed: int,
    strength: float,
    image_name: str,
    projector_steps: int,
    projector_learning_rate: float,
    projector_hidden_features: int,
    projector_seed: int = 20260922,
    objective_threshold: float | None = None,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    selected = tuple(methods)
    if task not in ("map", "image"):
        raise ValueError("task must be map or image")
    if not selected or len(set(selected)) != len(selected) or any(name not in METHOD_NAMES for name in selected):
        raise ValueError(f"methods must be distinct names drawn from {METHOD_NAMES}")
    if control_side < 3 or resolution < 2 or steps < 1:
        raise ValueError("control_side>=3, resolution>=2 and steps>=1 are required")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(strength) or strength <= 0.0:
        raise ValueError("strength must be finite and positive")
    threshold = (1.0e-4 if task == "map" else 1.0e-3) if objective_threshold is None else float(objective_threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("objective_threshold must be finite and nonnegative")
    if projector_steps < 1 or projector_hidden_features < 1:
        raise ValueError("projector steps/features must be positive")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be float32 or float64")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        whole_cuda_baseline_allocated = torch.cuda.memory_allocated(device)
        whole_cuda_baseline_reserved = torch.cuda.memory_reserved(device)
    else:
        whole_cuda_baseline_allocated = whole_cuda_baseline_reserved = None
    torch.manual_seed(seed)
    np.random.seed(seed)
    mesh = structured_rectangle(control_side - 1, control_side - 1)
    graph = build_standard_square_graph(control_side - 1)
    if not np.array_equal(mesh.faces, graph.mesh.faces) or not np.array_equal(mesh.vertices, graph.mesh.vertices):
        raise RuntimeError("positive graph does not match the shared structured mesh")
    target_start = perf_counter()
    target = build_directed_target(
        mesh,
        image_height=resolution,
        image_width=resolution,
        seed=seed,
        strength=strength,
        height=0.9 if task == "map" else 1.0,
    )
    target_setup_seconds = perf_counter() - target_start
    loop = graph.mesh.boundary_loops[0]
    loop_tensor = torch.tensor(np.array(loop, copy=True), dtype=torch.int64)
    boundary = target.control.index_select(0, loop_tensor).detach().to(device=device, dtype=dtype)
    target_summary = _target_summary(target, boundary)
    formal_target_required = (
        control_side == 25
        and resolution == 256
        and seed == 20260922
        and math.isclose(strength, 0.25, rel_tol=0.0, abs_tol=0.0)
    )
    target_authority_audit = _audit_clean_target_authority(
        task, target_summary, required=formal_target_required
    )
    query = StructuredDenseQueryTable.from_mesh(mesh, height=resolution, width=resolution)
    query.prepare(device=device, dtype=dtype)
    moving = fixed = None
    if task == "image":
        moving = synthetic_image(
            image_name,
            height=resolution,
            width=resolution,
            dtype=dtype,
            device=device,
        )
        fixed = warp_image_backward(
            moving, target.dense.detach().to(device=device, dtype=dtype)
        )

    projector = None
    projector_training = None
    if any(name.startswith("P1_") for name in selected):
        projector, projector_training = _train_projector(
            graph.direction_angles,
            steps=projector_steps,
            learning_rate=projector_learning_rate,
            hidden_features=projector_hidden_features,
            dtype=dtype,
            device=device,
            seed=projector_seed,
        )

    if device.type == "cuda":
        _synchronize(device)
        pre_method_peak_allocated = torch.cuda.max_memory_allocated(device)
        pre_method_peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        pre_method_peak_allocated = pre_method_peak_reserved = None

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name in selected:
        try:
            row = _run_method(
                name=name,
                mesh=mesh,
                graph=graph,
                target=target,
                boundary=boundary,
                query=query,
                task=task,
                steps=steps,
                learning_rate=learning_rate,
                objective_threshold=threshold,
                projector_setup_seconds=(
                    float(projector_training["wall_seconds"])
                    if name.startswith("P1_") and projector_training is not None
                    else 0.0
                ),
                dtype=dtype,
                device=device,
                projector=projector,
                moving=moving,
                fixed=fixed,
            )
        except Exception as error:
            row = _failure_row(name, error)
            failures.append(
                {"method": name, "type": type(error).__name__, "message": str(error)}
            )
        rows.append(row)

    receipt = {
        "schema": "phase5_route3_positive_hodge_instance_v1",
        "status": "ok" if not failures else "complete_with_failures",
        "protocol_kind": "optimized_instance_with_projection_only_baseline",
        "research_question": (
            "Can a facewise Beltrami latent pass through a learned strictly-positive "
            "fixed-planar projector and symmetric Tutte solve as a differentiable, "
            "hard-certified N-by-N control layer under an exact solve budget?"
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "task": task,
            "methods": list(selected),
            "control_side": control_side,
            "control_vertices": mesh.n_vertices,
            "faces": mesh.n_faces,
            "resolution": resolution,
            "steps": steps,
            "learning_rate": learning_rate,
            "requested_global_solves": 1 + 2 * steps,
            "formal_primary_global_solve_slice": 83,
            "formal_final_global_solve_censor": 163,
            "seed": seed,
            "strength": strength,
            "target_height": 0.9 if task == "map" else 1.0,
            "image_name": image_name if task == "image" else None,
            "dtype": str(dtype).removeprefix("torch."),
            "device": str(device),
            "fixed_target_boundary": True,
            "maximum_mu_radius": MAXIMUM_MU_RADIUS,
            "warp_convention": "fixed-to-moving backward coordinates; no inverse computed",
            "projector_steps": projector_steps,
            "projector_learning_rate": projector_learning_rate,
            "projector_hidden_features": projector_hidden_features,
            "projector_seed": projector_seed,
            "projector_seed_is_logically_independent_of_target_seed": True,
            "projector_training_radii": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
            "projector_training_angles": 180,
            "objective_threshold": threshold,
        },
        "environment": _environment(device),
        "structured_mesh": _mesh_summary(mesh),
        "shared_target": {
            "generated_once_cpu_float64": True,
            "setup_primal_solves": 1,
            "setup_seconds": target_setup_seconds,
            **target_summary,
        },
        "target_authority_audit": target_authority_audit,
        "projector_training": projector_training,
        "memory_scope": {
            "per_method_reset_after_projector_training": True,
            "per_method_peak_excludes_projector_training": True,
            "per_method_baseline_includes_resident_target_query_and_projector": True,
            "formal_interpretation": "one selected method per fresh process",
        },
        "whole_benchmark_memory": {
            "cuda_baseline_allocated_bytes": whole_cuda_baseline_allocated,
            "cuda_baseline_reserved_bytes": whole_cuda_baseline_reserved,
            "whole_benchmark_cuda_peak_allocated_bytes": (
                max(
                    [
                        int(pre_method_peak_allocated or 0),
                        *[
                            int(row.get("memory", {}).get("cuda_peak_allocated_bytes") or 0)
                            for row in rows
                        ],
                    ]
                )
                if device.type == "cuda"
                else None
            ),
            "whole_benchmark_cuda_peak_reserved_bytes": (
                max(
                    [
                        int(pre_method_peak_reserved or 0),
                        *[
                            int(row.get("memory", {}).get("cuda_peak_reserved_bytes") or 0)
                            for row in rows
                        ],
                    ]
                )
                if device.type == "cuda"
                else None
            ),
            "scope": (
                "maximum of absolute CUDA allocator peaks across entry-to-method setup and each "
                "post-reset method segment; includes target/query/projector training and optimization"
            ),
        },
        "methods": rows,
        "failures": failures,
    }
    _validate_finite_json(receipt)
    return receipt


def _validate_finite_json(value: Any, path: str = "receipt") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise RuntimeError(f"nonfinite JSON value at {path}")


def _parse_methods(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("methods must be nonempty")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("map", "image"), required=True)
    parser.add_argument("--methods", type=_parse_methods, required=True)
    parser.add_argument("--N", dest="control_side", type=int, required=True)
    parser.add_argument("--R", dest="resolution", type=int, required=True)
    parser.add_argument("--steps", type=int, default=81)
    parser.add_argument("--lr", dest="learning_rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--image", dest="image_name", default="medical_phantom")
    parser.add_argument("--projector-steps", type=int, default=1500)
    parser.add_argument("--projector-learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--projector-hidden-features", type=int, default=32)
    parser.add_argument("--projector-seed", type=int, default=20260922)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unknown = sorted(set(args.methods) - set(METHOD_NAMES))
    if unknown:
        parser.error(f"unknown methods: {unknown}; choices are {METHOD_NAMES}")
    if args.control_side < 3 or args.resolution < 2 or args.steps < 1:
        parser.error("N>=3, R>=2 and steps>=1 are required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    receipt = run_benchmark(
        task=args.task,
        methods=args.methods,
        control_side=args.control_side,
        resolution=args.resolution,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        strength=args.strength,
        image_name=args.image_name,
        projector_steps=args.projector_steps,
        projector_learning_rate=args.projector_learning_rate,
        projector_hidden_features=args.projector_hidden_features,
        projector_seed=args.projector_seed,
        objective_threshold=args.threshold,
        dtype=dtype,
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if receipt["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
