"""Route III-E--G fixed-planar positive direction-moment coverage study.

The receipt reports three deliberately separate levels:

1. local ``sum c d d^T`` tensor approximation;
2. the fixed-boundary rows of a globally assembled shared-edge operator; and
3. the decoded P1 map and its facewise Beltrami coefficient.

No equality between these levels is assumed.  The full anisotropic P1 operator
is used only as a numerical teacher; it has no automatic topology guarantee.
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
from typing import Any, Callable, Iterable

import numpy as np

# Initialize NumPy LAPACK before Torch on the supported Windows/Conda host.
_NUMPY_LINALG_BOOTSTRAP = float(np.linalg.cond(np.eye(1, dtype=np.float64)))

import scipy
from scipy.optimize import nnls
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from qcopt.neural_bijection.metrics import compute_p1_map_metrics  # noqa: E402
from qcopt.neural_bijection.tutte.positive_hodge import (  # noqa: E402
    LearnedPositiveDirectionMap,
    PositivePlanarGraph,
    audit_positive_planar_graph,
    beltrami_tensor,
    build_center_split_square_graph,
    build_standard_square_graph,
    build_stellar_square_graph,
    fit_direction_tensor_nnls,
)


DEFAULT_RADII = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)
MINIMUM_CONDUCTANCE = 1.0e-6
MINIMUM_EXCESS = 1.0e-10


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
    status = Path("/proc/self/status")
    result: dict[str, int | None] = {"rss_bytes": None, "hwm_bytes": None}
    if not status.exists():
        return result
    for line in status.read_text(encoding="utf-8").splitlines():
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


def _unique_interior(graph: PositivePlanarGraph) -> tuple[np.ndarray, np.ndarray]:
    boundary = graph.mesh.boundary_loops[0]
    interior = np.setdiff1d(np.arange(graph.mesh.n_vertices, dtype=np.int64), boundary)
    return interior, boundary


def _fem_stiffness(graph: PositivePlanarGraph, tensor: np.ndarray) -> np.ndarray:
    mesh = graph.mesh
    local = mesh.areas[:, None, None] * np.einsum(
        "tvi,ij,twj->tvw", mesh.gradients, tensor, mesh.gradients
    )
    result = np.zeros((mesh.n_vertices, mesh.n_vertices), dtype=np.float64)
    for face, element in zip(mesh.faces, local, strict=True):
        result[np.ix_(face, face)] += element
    return result


def _graph_operator_bases(graph: PositivePlanarGraph) -> np.ndarray:
    count = len(graph.direction_angles)
    bases = np.zeros((count, graph.mesh.n_vertices, graph.mesh.n_vertices), dtype=np.float64)
    for (first, second), direction in zip(
        graph.active_edges, graph.active_edge_direction_indices, strict=True
    ):
        basis = bases[int(direction)]
        basis[first, first] += 1.0
        basis[second, second] += 1.0
        basis[first, second] -= 1.0
        basis[second, first] -= 1.0
    return bases


def _monotone_rectangle_boundary(graph: PositivePlanarGraph) -> np.ndarray:
    _, boundary = _unique_interior(graph)
    source = graph.mesh.vertices[boundary]
    # phi'(t)=1+0.25 cos(2 pi t)>0, so side order and corners are exact.
    def phi(values: np.ndarray) -> np.ndarray:
        return values + 0.25 * np.sin(2.0 * math.pi * values) / (2.0 * math.pi)

    return np.column_stack((phi(source[:, 0]), phi(source[:, 1])))


def _solve_dirichlet(
    matrix: np.ndarray,
    interior: np.ndarray,
    boundary: np.ndarray,
    boundary_values: np.ndarray,
) -> np.ndarray:
    result = np.empty((matrix.shape[0], boundary_values.shape[1]), dtype=np.float64)
    result[boundary] = boundary_values
    result[interior] = np.linalg.solve(
        matrix[np.ix_(interior, interior)],
        -matrix[np.ix_(interior, boundary)] @ boundary_values,
    )
    return result


def _operator_diagnostics(
    target: np.ndarray,
    bases: np.ndarray,
    coefficients: np.ndarray,
    interior: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    candidate = np.einsum("m,mij->ij", coefficients, bases)
    target_rows = target[interior]
    candidate_rows = candidate[interior]
    denominator = float(np.sum(candidate_rows * candidate_rows))
    scale = float(np.sum(candidate_rows * target_rows) / denominator)
    scale = max(scale, np.finfo(np.float64).tiny)
    error = float(
        np.linalg.norm(scale * candidate_rows - target_rows, ord="fro")
        / np.linalg.norm(target_rows, ord="fro")
    )
    design = np.column_stack([basis[interior].ravel() for basis in bases])
    lower = np.full(bases.shape[0], MINIMUM_CONDUCTANCE + MINIMUM_EXCESS)
    free, _ = nnls(design, target_rows.ravel() - design @ lower)
    best = lower + free
    best_error = float(
        np.linalg.norm(design @ best - target_rows.ravel())
        / np.linalg.norm(target_rows.ravel())
    )
    return scale * candidate, error, best_error


def _evaluate_coefficients(
    graph: PositivePlanarGraph,
    tensor: np.ndarray,
    coefficients: np.ndarray,
    fitted_tensor: np.ndarray,
    bases: np.ndarray,
    boundary_values: np.ndarray,
) -> dict[str, float | bool]:
    interior, boundary = _unique_interior(graph)
    teacher_operator = _fem_stiffness(graph, tensor)
    student_operator, assembled_error, global_lower_bound = _operator_diagnostics(
        teacher_operator, bases, coefficients, interior
    )
    teacher_map = _solve_dirichlet(
        teacher_operator, interior, boundary, boundary_values
    )
    student_map = _solve_dirichlet(
        student_operator, interior, boundary, boundary_values
    )
    student_metrics = compute_p1_map_metrics(
        graph.mesh, student_map, target=teacher_map
    )
    teacher_metrics = compute_p1_map_metrics(graph.mesh, teacher_map)
    difference = fitted_tensor - tensor
    return {
        "tensor_frobenius_relative_error": float(
            np.linalg.norm(difference, ord="fro") / np.linalg.norm(tensor, ord="fro")
        ),
        "local_operator_symbol_relative_error": float(
            np.linalg.norm(difference, ord=2) / np.linalg.norm(tensor, ord=2)
        ),
        "assembled_fixed_boundary_operator_relative_error": assembled_error,
        "best_global_class_operator_nnls_relative_error": global_lower_bound,
        "decoded_map_rmse": student_metrics.map_rmse,
        "decoded_mu_rmse": student_metrics.mu_rmse,
        "decoded_maximum_mu_error": student_metrics.maximum_mu_error,
        "student_topology_certified": student_metrics.global_injectivity_certificate,
        "student_minimum_area_ratio": student_metrics.minimum_area_ratio,
        "teacher_topology_certified": teacher_metrics.global_injectivity_certificate,
        "teacher_minimum_area_ratio": teacher_metrics.minimum_area_ratio,
    }


def _training_tensors(radii: tuple[float, ...], angles: np.ndarray) -> np.ndarray:
    values = np.asarray(
        [radius * np.exp(1j * angle) for radius in radii for angle in angles],
        dtype=np.complex128,
    )
    return beltrami_tensor(values)


def _train_local_map(
    graph: PositivePlanarGraph,
    *,
    radii: tuple[float, ...],
    train_angles: np.ndarray,
    train_steps: int,
    learning_rate: float,
    hidden_features: int,
    device: torch.device,
    seed: int,
) -> tuple[LearnedPositiveDirectionMap, dict[str, Any]]:
    model = LearnedPositiveDirectionMap(
        graph.direction_angles,
        hidden_features=hidden_features,
        minimum_conductance=MINIMUM_CONDUCTANCE,
        seed=seed,
    ).to(device=device, dtype=torch.float64)
    target = torch.tensor(
        _training_tensors(radii, train_angles), dtype=torch.float64, device=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start = perf_counter()
    initial_loss = None
    final_loss = None
    for step in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        coefficients = model(target)
        fitted = model.fitted_tensors(coefficients)
        numerator = (fitted - target).square().sum(dim=(-2, -1))
        denominator = target.square().sum(dim=(-2, -1))
        loss = (numerator / denominator).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("learned local direction fit produced a nonfinite loss")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, {
        "seed": seed,
        "hidden_features": hidden_features,
        "steps": train_steps,
        "learning_rate": learning_rate,
        "training_samples": len(target),
        "initial_relative_frobenius_squared_loss": initial_loss,
        "final_relative_frobenius_squared_loss": final_loss,
        "wall_seconds": perf_counter() - start,
    }


def _metric_summary(values: list[float], phases: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    worst = int(np.argmax(array))
    return {
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "maximum": float(array[worst]),
        "worst_mu_argument_radians": float(phases[worst]),
        "worst_mu_argument_degrees": float(math.degrees(phases[worst]) % 360.0),
    }


def _summarize_samples(
    *,
    graph: PositivePlanarGraph,
    radius: float,
    method: str,
    split: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = (
        "tensor_frobenius_relative_error",
        "local_operator_symbol_relative_error",
        "assembled_fixed_boundary_operator_relative_error",
        "best_global_class_operator_nnls_relative_error",
        "decoded_map_rmse",
        "decoded_mu_rmse",
        "decoded_maximum_mu_error",
        "student_minimum_area_ratio",
        "teacher_minimum_area_ratio",
    )
    phases = [float(sample["phase"]) for sample in samples]
    result: dict[str, Any] = {
        "graph": graph.name,
        "radius": radius,
        "method": method,
        "split": split,
        "sample_count": len(samples),
        "strict_minimum_conductance": float(
            min(sample["minimum_coefficient"] for sample in samples)
        ),
        "student_topology_failure_count": sum(
            not sample["student_topology_certified"] for sample in samples
        ),
        "teacher_topology_failure_count": sum(
            not sample["teacher_topology_certified"] for sample in samples
        ),
    }
    for name in metric_names:
        result[name] = _metric_summary(
            [float(sample[name]) for sample in samples], phases
        )
    return result


def _projective_maximum_gap(angles: np.ndarray) -> float:
    values = np.sort(np.mod(angles, math.pi))
    gaps = np.diff(np.r_[values, values[0] + math.pi])
    return float(np.max(gaps))


def _graph_record(graph: PositivePlanarGraph) -> dict[str, Any]:
    audit = audit_positive_planar_graph(graph)
    gap = _projective_maximum_gap(graph.direction_angles)
    # The exact-positive threshold is a worst-orientation local cone result,
    # not a global shared-edge or map theorem.
    uniform_radius = 0.0 if gap > math.pi / 4.0 + 1.0e-13 else math.sqrt(2.0) - 1.0
    return {
        "name": graph.name,
        "direction_degrees": np.rad2deg(graph.direction_angles).tolist(),
        "maximum_projective_gap_degrees": math.degrees(gap),
        "local_cone_uniform_exact_positive_radius_open_bound": uniform_radius,
        "strict_floor_can_add_nonzero_error_at_the_cone_boundary": True,
        "audit": asdict(audit),
    }


def _evaluate_method(
    graph: PositivePlanarGraph,
    *,
    radius: float,
    phases: np.ndarray,
    bases: np.ndarray,
    boundary_values: np.ndarray,
    coefficient_function: Any,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for phase in phases:
        tensor = beltrami_tensor(radius * np.exp(1j * phase))
        coefficients, fitted = coefficient_function(tensor)
        diagnostics = _evaluate_coefficients(
            graph, tensor, coefficients, fitted, bases, boundary_values
        )
        samples.append(
            {
                "phase": float(phase),
                "minimum_coefficient": float(np.min(coefficients)),
                **diagnostics,
            }
        )
    return samples


def run_coverage(
    *,
    cells_per_side: int = 4,
    angles_per_split: int = 180,
    radii: Iterable[float] = DEFAULT_RADII,
    train_steps: int = 1500,
    train_learning_rate: float = 5.0e-3,
    hidden_features: int = 32,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    if cells_per_side < 2:
        raise ValueError("cells_per_side must be at least two")
    if angles_per_split < 4 or angles_per_split % 2:
        raise ValueError("angles_per_split must be an even integer at least four")
    radii_tuple = tuple(float(value) for value in radii)
    if not radii_tuple or any(not math.isfinite(value) or value < 0.0 or value >= 1.0 for value in radii_tuple):
        raise ValueError("radii must be finite and lie in [0,1)")
    if train_steps < 1 or hidden_features < 1:
        raise ValueError("train_steps and hidden_features must be positive")
    if not math.isfinite(train_learning_rate) or train_learning_rate <= 0.0:
        raise ValueError("train_learning_rate must be finite and positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        cuda_baseline_allocated = torch.cuda.memory_allocated(device)
        cuda_baseline_reserved = torch.cuda.memory_reserved(device)
    else:
        cuda_baseline_allocated = cuda_baseline_reserved = None
    memory_before = _process_memory()
    start = perf_counter()
    train_angles = np.arange(angles_per_split, dtype=np.float64) * (
        2.0 * math.pi / angles_per_split
    )
    heldout_angles = np.mod(
        train_angles + math.pi / angles_per_split, 2.0 * math.pi
    )
    builders: tuple[Callable[[int], PositivePlanarGraph], ...] = (
        build_standard_square_graph,
        build_center_split_square_graph,
        build_stellar_square_graph,
    )
    graph_records: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for graph_index, builder in enumerate(builders):
        graph = builder(cells_per_side)
        graph_records.append(_graph_record(graph))
        bases = _graph_operator_bases(graph)
        boundary_values = _monotone_rectangle_boundary(graph)
        try:
            learned, training = _train_local_map(
                graph,
                radii=radii_tuple,
                train_angles=train_angles,
                train_steps=train_steps,
                learning_rate=train_learning_rate,
                hidden_features=hidden_features,
                device=device,
                seed=20260922 + graph_index,
            )
            training_records.append({"graph": graph.name, **training})
        except Exception as error:
            failures.append(
                {
                    "graph": graph.name,
                    "phase": "learned_local_map_training",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue

        def nnls_coefficients(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            fit = fit_direction_tensor_nnls(
                tensor,
                graph.direction_angles,
                minimum_conductance=MINIMUM_CONDUCTANCE,
                minimum_excess=MINIMUM_EXCESS,
            )
            return fit.conductances, fit.fitted_tensor

        def learned_coefficients(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            with torch.no_grad():
                target = torch.tensor(tensor, dtype=torch.float64, device=device)
                coefficients_torch = learned(target)
                fitted_torch = learned.fitted_tensors(coefficients_torch)
            return (
                coefficients_torch.detach().cpu().numpy(),
                fitted_torch.detach().cpu().numpy(),
            )

        for radius in radii_tuple:
            for method, coefficient_function in (
                ("deterministic_lower_bound_nnls", nnls_coefficients),
                ("learned_softplus_local_map", learned_coefficients),
            ):
                try:
                    samples = _evaluate_method(
                        graph,
                        radius=radius,
                        phases=heldout_angles,
                        bases=bases,
                        boundary_values=boundary_values,
                        coefficient_function=coefficient_function,
                    )
                    summaries.append(
                        _summarize_samples(
                            graph=graph,
                            radius=radius,
                            method=method,
                            split="held_out_half_step_angle_grid",
                            samples=samples,
                        )
                    )
                except Exception as error:
                    failures.append(
                        {
                            "graph": graph.name,
                            "radius": radius,
                            "method": method,
                            "phase": "held_out_evaluation",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    memory_after = _process_memory()
    receipt = {
        "schema": "phase5_route3_positive_hodge_coverage_v1",
        "status": "ok" if not failures else "complete_with_failures",
        "research_question": (
            "What local tensor, assembled fixed-boundary operator, and decoded-map "
            "accuracy frontier is attainable with strictly positive coefficients "
            "on three fixed planar direction sets?"
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "local_direction_moment_is_not_global_operator": True,
            "assembled_operator_is_not_schur_effective_tensor": True,
            "finite_directions_exact_arbitrary_spd": False,
            "teacher_topology_guarantee": False,
            "student_topology_claim": (
                "independently certified represented P1 map for each successful positive solve"
            ),
        },
        "configuration": {
            "cells_per_side": cells_per_side,
            "radii": list(radii_tuple),
            "angles_per_split": angles_per_split,
            "train_angle_spacing_degrees": 360.0 / angles_per_split,
            "heldout_angle_offset_degrees": 180.0 / angles_per_split,
            "minimum_conductance": MINIMUM_CONDUCTANCE,
            "minimum_excess": MINIMUM_EXCESS,
            "train_steps": train_steps,
            "train_learning_rate": train_learning_rate,
            "hidden_features": hidden_features,
            "tensor_error": "relative Frobenius norm of local direction moment",
            "local_operator_symbol_error": "relative spectral norm of the same 2x2 local symbol",
            "assembled_operator_error": (
                "relative Frobenius norm of interior rows after one best positive global scale"
            ),
            "map_error": "student versus full-anisotropic-P1 teacher on one monotone rectangle boundary",
        },
        "environment": _environment(device),
        "graphs": graph_records,
        "learned_training": training_records,
        "summaries": summaries,
        "failures": failures,
        "wall_seconds": perf_counter() - start,
        "memory": {
            "rss_before_bytes": memory_before["rss_bytes"],
            "rss_after_bytes": memory_after["rss_bytes"],
            "process_hwm_bytes": memory_after["hwm_bytes"],
            "cuda_baseline_allocated_bytes": cuda_baseline_allocated,
            "cuda_baseline_reserved_bytes": cuda_baseline_reserved,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            ),
        },
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


def _parse_radii(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("radii must be comma-separated numbers") from error
    if not result:
        raise argparse.ArgumentTypeError("radii must be nonempty")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--angles-per-split", type=int, default=180)
    parser.add_argument("--radii", type=_parse_radii, default=DEFAULT_RADII)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--train-learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--hidden-features", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.angles_per_split < 4 or args.angles_per_split % 2:
        parser.error("angles-per-split must be an even integer at least four")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    receipt = run_coverage(
        cells_per_side=args.cells,
        angles_per_split=args.angles_per_split,
        radii=args.radii,
        train_steps=args.train_steps,
        train_learning_rate=args.train_learning_rate,
        hidden_features=args.hidden_features,
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
