"""Fair fixed-boundary instance optimization for Route-II MVC geometry.

The four methods in this module deliberately share one target, target boundary,
uniform-row initialization, dense query table, loss, step count, learning rate,
and Adam hyperparameters.  They differ only in their interior coordinates and
update rule:

``O1_sigmoid_positive``
    unconstrained raw rows represented as ``log(sigmoid(raw))`` before the
    decoder's supported row softmax;
``O2_row_softmax``
    ordinary supported row-softmax logits;
``O3_mvc_adam``
    projected logit Adam followed by candidate decode, MVC re-encoding, and a
    second canonical decode after every step.  Adam moments are retained with
    an explicitly declared identity transport across that nonlinear projection;
``O4_covariance_retraction``
    vertex-space Adam on interior vertices, MVC canonicalization, the
    covariance logit lift, and a decoder-mediated retraction.  It never sends
    the loss through a decoder adjoint and never accepts the raw Euler point.

The fixed target boundary is intentional: it makes the zero-boundary tangent
semantics of O4 exact.  This benchmark therefore does not evaluate learnable
boundary or modulus parameters.  Completed numerical solves and attempted
calls are separate fields; O3's candidate and canonical re-decodes are never
hidden.  Local reverse-mode work in O4 is recorded separately and is not
mislabeled as a global adjoint solve.

Equal outer steps and equal completed global solves are different protocols.
The primary prescribed slice uses exactly 83 completed global solves
(``41/41/27/81`` accepted steps for O1/O2/O3/O4); the secondary slice uses 40
outer steps and therefore ``81/81/122/42`` solves.  Extraction helpers keep
these cohorts distinct.  A shared numeric learning rate is only a robustness
cohort because logits, raw weights, and vertex coordinates have different
units; tuned-rate evidence must be reported separately.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as torch_functional

from ...mesh import TriMesh, structured_rectangle
from ..benchmarks import synthetic_image, warp_image_backward
from ..metrics import P1MapMetrics, compute_p1_map_metrics
from .dense_warp import StructuredDenseQueryTable
from .direct import DirectTutteLayer
from .instance_optimization import (
    DirectedTarget,
    _validate_prebuilt_target,
    build_directed_target,
)
from .iterative import MatrixFreeDirectedTutteLayer
from .mvc import MeanValueCoordinateEncoder
from .mvc_retraction import CovarianceLogitLift


METHOD_NAMES = (
    "O1_sigmoid_positive",
    "O2_row_softmax",
    "O3_mvc_adam",
    "O4_covariance_retraction",
)


@dataclass(frozen=True)
class OptimizationFailure:
    """The first failed operation; the last accepted state remains in the result."""

    phase: str
    attempted_step: int
    exception_type: str
    message: str


@dataclass(frozen=True)
class MVCOptimizationTraceRow:
    """One accepted decoder state and disjoint measured wall-time components.

    ``primal_seconds`` is the decoder solve only; ``dense_loss_seconds`` adds
    fixed-query interpolation, optional image warp, and objective evaluation.
    MVC covariance conditioning always uses canonical MVC probabilities and is
    named accordingly, rather than being presented as O1/O2 latent conditioning.
    """

    iteration: int
    objective: float
    primal_solves: int
    adjoint_solves: int
    global_solves: int
    local_backwards: int
    primal_attempts: int
    adjoint_attempts: int
    local_backward_attempts: int
    primal_krylov_rhs_iterations: int
    adjoint_krylov_rhs_iterations: int
    maximum_primal_krylov_iterations: int
    maximum_adjoint_krylov_iterations: int
    maximum_primal_krylov_relative_residual: float | None
    maximum_adjoint_krylov_relative_residual: float | None
    primal_seconds: float
    adjoint_seconds: float
    local_backward_seconds: float
    dense_loss_seconds: float
    optimizer_seconds: float
    canonicalization_seconds: float
    lift_seconds: float
    audit_seconds: float
    observation_wall_seconds: float
    raw_euler_gap_rmse: float | None
    flip_count: int
    minimum_signed_area: float
    minimum_area_ratio: float
    boundary_order_min_gap: float
    topology_certified: bool
    map_rmse: float
    maximum_map_error: float
    mu_rmse: float
    maximum_mu_error: float
    logit_spread: float
    maximum_mvc_canonical_covariance_condition: float
    system_condition_number: float | None
    system_condition_method: str
    canonicality_error: float


@dataclass(frozen=True)
class MVCMethodResult:
    """Complete accounting for one O1--O4 optimization run."""

    method: str
    task: str
    backend: str
    boundary_semantics: str
    optimizer_semantics: str
    moment_policy: str
    learning_rate: float
    steps_requested: int
    completed_steps: int
    initial_objective: float
    final_objective: float
    best_objective: float
    primal_solves: int
    adjoint_solves: int
    global_solves: int
    local_backwards: int
    primal_attempts: int
    adjoint_attempts: int
    local_backward_attempts: int
    primal_krylov_rhs_iterations: int
    adjoint_krylov_rhs_iterations: int
    maximum_primal_krylov_iterations: int
    maximum_adjoint_krylov_iterations: int
    maximum_primal_krylov_relative_residual: float | None
    maximum_adjoint_krylov_relative_residual: float | None
    objective_threshold: float | None
    accepted_steps_to_threshold: int | None
    global_solves_to_threshold: int | None
    wall_seconds_to_threshold: float | None
    wall_seconds: float
    algorithm_wall_seconds: float
    audit_wall_seconds: float
    unattributed_wall_seconds: float
    all_iterates_certified: bool
    final_metrics: P1MapMetrics
    final_logit_spread: float
    maximum_mvc_canonical_covariance_condition: float
    final_system_condition_number: float | None
    system_condition_method: str
    final_canonicality_error: float
    failure: OptimizationFailure | None
    initial_control: torch.Tensor
    final_control: torch.Tensor
    final_logits: torch.Tensor
    fixed_boundary: torch.Tensor
    trace: tuple[MVCOptimizationTraceRow, ...]


@dataclass(frozen=True)
class FixedBoundaryComparison:
    """A shared problem and ordered O1--O4 results."""

    task: str
    backend: str
    control_vertices: int
    image_resolution: int
    steps: int
    learning_rate: float
    seed: int
    target_strength: float
    image_name: str | None
    warp_convention: str
    boundary_semantics: str
    target_setup_primal_solves: int
    target_setup_seconds: float
    target_control: torch.Tensor
    target_dense: torch.Tensor
    fixed_target_boundary: torch.Tensor
    uniform_initial_control: torch.Tensor
    results: dict[str, MVCMethodResult]
    primary_global_solve_budget: int
    primary_expected_accepted_steps: dict[str, int]
    secondary_outer_step: int
    secondary_expected_global_solves: dict[str, int]
    shared_learning_rate_semantics: str
    pilot_recommendation: str


@dataclass(frozen=True)
class MVCComparisonSlice:
    """A predeclared cross-method slice without silently dropping failures."""

    protocol: str
    budget: int
    rows: dict[str, MVCOptimizationTraceRow]
    missing: dict[str, str]

    @property
    def complete(self) -> bool:
        return not self.missing


@dataclass
class _Counters:
    primal: int = 0
    adjoint: int = 0
    local_backward: int = 0
    primal_attempts: int = 0
    adjoint_attempts: int = 0
    local_backward_attempts: int = 0
    primal_krylov_rhs_iterations: int = 0
    adjoint_krylov_rhs_iterations: int = 0
    maximum_primal_krylov_iterations: int = 0
    maximum_adjoint_krylov_iterations: int = 0
    maximum_primal_krylov_relative_residual: float | None = None
    maximum_adjoint_krylov_relative_residual: float | None = None


@dataclass(frozen=True)
class _Audit:
    metrics: P1MapMetrics
    logit_spread: float
    maximum_mvc_canonical_covariance_condition: float
    system_condition_number: float | None
    system_condition_method: str
    canonicality_error: float


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _record_krylov(
    counters: _Counters,
    solver: torch.nn.Module,
    *,
    phase: str,
) -> None:
    """Snapshot one successful matrix-free report; direct solves have none."""

    diagnostics = getattr(solver, "last_diagnostics", None)
    report = getattr(diagnostics, "forward" if phase == "primal" else "adjoint", None)
    if report is None:
        return
    iterations = report.iterations.detach().to(dtype=torch.int64, device="cpu")
    residual = report.relative_residual.detach().to(dtype=torch.float64, device="cpu")
    if not bool(report.converged.all()):
        raise RuntimeError(
            f"successful {phase} call exposed an unconverged Krylov report"
        )
    iteration_sum = int(iterations.sum())
    iteration_max = int(iterations.max()) if iterations.numel() else 0
    residual_max = float(residual.max()) if residual.numel() else 0.0
    if phase == "primal":
        counters.primal_krylov_rhs_iterations += iteration_sum
        counters.maximum_primal_krylov_iterations = max(
            counters.maximum_primal_krylov_iterations, iteration_max
        )
        old = counters.maximum_primal_krylov_relative_residual
        counters.maximum_primal_krylov_relative_residual = (
            residual_max if old is None else max(old, residual_max)
        )
    else:
        counters.adjoint_krylov_rhs_iterations += iteration_sum
        counters.maximum_adjoint_krylov_iterations = max(
            counters.maximum_adjoint_krylov_iterations, iteration_max
        )
        old = counters.maximum_adjoint_krylov_relative_residual
        counters.maximum_adjoint_krylov_relative_residual = (
            residual_max if old is None else max(old, residual_max)
        )


def _make_solver(
    mesh: TriMesh,
    *,
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    if backend == "direct":
        if device.type != "cpu":
            raise ValueError("direct backend is CPU-only")
        solver: torch.nn.Module = DirectTutteLayer(mesh)
    elif backend == "directed_iterative":
        solver = MatrixFreeDirectedTutteLayer(mesh)
    else:
        raise ValueError("backend must be 'direct' or 'directed_iterative'")
    return solver.to(device=device, dtype=dtype)


def _decode(
    solver: torch.nn.Module,
    logits: torch.Tensor,
    boundary: torch.Tensor,
    counters: _Counters,
) -> tuple[torch.Tensor, float]:
    counters.primal_attempts += 1
    _synchronize(logits.device)
    start = perf_counter()
    control = solver(logits, boundary)
    _synchronize(logits.device)
    elapsed = perf_counter() - start
    counters.primal += 1
    _record_krylov(counters, solver, phase="primal")
    return control, elapsed


def _dense_loss(
    task: str,
    query_table: StructuredDenseQueryTable,
    control: torch.Tensor,
    target_dense: torch.Tensor,
    *,
    moving: torch.Tensor | None,
    fixed: torch.Tensor | None,
) -> torch.Tensor:
    dense = query_table.interpolate(control)
    if task == "supervised_map":
        return (dense - target_dense).square().mean()
    if moving is None or fixed is None:
        raise RuntimeError("image registration requires moving and fixed images")
    tolerance = 64.0 * torch.finfo(control.dtype).eps
    if bool(torch.any(dense < -tolerance)) or bool(torch.any(dense > 1.0 + tolerance)):
        raise RuntimeError("decoded backward map left the unit image domain")
    return (warp_image_backward(moving, dense) - fixed).square().mean()


def _timed_dense_loss(
    task: str,
    query_table: StructuredDenseQueryTable,
    control: torch.Tensor,
    target_dense: torch.Tensor,
    *,
    moving: torch.Tensor | None,
    fixed: torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    _synchronize(control.device)
    start = perf_counter()
    loss = _dense_loss(
        task,
        query_table,
        control,
        target_dense,
        moving=moving,
        fixed=fixed,
    )
    _synchronize(control.device)
    return loss, perf_counter() - start


def _supported_probabilities(
    logits: torch.Tensor, valid_mask: np.ndarray
) -> torch.Tensor:
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=logits.device)
    return torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)


def _center_supported_logits(
    logits: torch.Tensor, valid_mask: np.ndarray
) -> torch.Tensor:
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=logits.device)
    degree = mask.sum(dim=-1, keepdim=True)
    mean = (
        torch.where(mask, logits, torch.zeros_like(logits)).sum(dim=-1, keepdim=True)
        / degree
    )
    return torch.where(mask, logits - mean, torch.zeros_like(logits))


def _logit_spread(logits: torch.Tensor, valid_mask: np.ndarray) -> float:
    if logits.numel() == 0:
        return 0.0
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=logits.device)
    maximum = logits.masked_fill(~mask, -torch.inf).amax(dim=-1)
    minimum = logits.masked_fill(~mask, torch.inf).amin(dim=-1)
    return float((maximum - minimum).amax().detach().cpu())


def _system_condition(
    system,
    probabilities: torch.Tensor,
    *,
    dense_limit: int,
) -> tuple[float | None, str]:
    """Exact dense 2-norm condition only below an explicit diagnostic limit.

    No hidden inverse solve is performed above the limit.  This diagnostic is
    algebra on a detached matrix and is not counted as a primal/adjoint solve.
    """

    rows = system.n_rows
    if rows == 0:
        return 1.0, "exact_dense_2norm"
    if rows > dense_limit:
        return None, f"not_computed_n_rows_gt_{dense_limit}"
    values = probabilities.detach().to(dtype=torch.float64, device="cpu").numpy()
    matrix = np.eye(rows, dtype=np.float64)
    row, slot = np.nonzero(system.valid_mask & ~system.neighbor_is_boundary)
    matrix[row, system.neighbors[row, slot]] -= values[row, slot]
    return float(np.linalg.cond(matrix, p=2)), "exact_dense_2norm"


def _audit(
    mesh: TriMesh,
    encoder: MeanValueCoordinateEncoder,
    control: torch.Tensor,
    represented_logits: torch.Tensor,
    target_control: torch.Tensor,
    *,
    system_condition_dense_limit: int,
) -> _Audit:
    with torch.no_grad():
        detached = control.detach()
        canonical = encoder(detached)
        centered = _center_supported_logits(
            represented_logits.detach(), encoder.valid_mask
        )
        canonicality = float(torch.max(torch.abs(centered - canonical.logits)).cpu())
        probabilities = _supported_probabilities(
            represented_logits.detach(), encoder.valid_mask
        )
        condition, method = _system_condition(
            encoder.system,
            probabilities,
            dense_limit=system_condition_dense_limit,
        )
        covariance = (
            float(canonical.diagnostics.covariance_condition.max().cpu())
            if canonical.diagnostics.covariance_condition.numel()
            else 1.0
        )
    metrics = compute_p1_map_metrics(
        mesh,
        control.detach().to(dtype=torch.float64, device="cpu").numpy(),
        target=target_control.detach().to(dtype=torch.float64, device="cpu").numpy(),
    )
    return _Audit(
        metrics=metrics,
        logit_spread=_logit_spread(represented_logits.detach(), encoder.valid_mask),
        maximum_mvc_canonical_covariance_condition=covariance,
        system_condition_number=condition,
        system_condition_method=method,
        canonicality_error=canonicality,
    )


def _trace_row(
    *,
    iteration: int,
    loss: torch.Tensor,
    counters: _Counters,
    timings: dict[str, float],
    wall_start: float,
    audit: _Audit,
    raw_euler_gap_rmse: float | None = None,
) -> MVCOptimizationTraceRow:
    metrics = audit.metrics
    return MVCOptimizationTraceRow(
        iteration=iteration,
        objective=float(loss.detach().cpu()),
        primal_solves=counters.primal,
        adjoint_solves=counters.adjoint,
        global_solves=counters.primal + counters.adjoint,
        local_backwards=counters.local_backward,
        primal_attempts=counters.primal_attempts,
        adjoint_attempts=counters.adjoint_attempts,
        local_backward_attempts=counters.local_backward_attempts,
        primal_krylov_rhs_iterations=counters.primal_krylov_rhs_iterations,
        adjoint_krylov_rhs_iterations=counters.adjoint_krylov_rhs_iterations,
        maximum_primal_krylov_iterations=counters.maximum_primal_krylov_iterations,
        maximum_adjoint_krylov_iterations=counters.maximum_adjoint_krylov_iterations,
        maximum_primal_krylov_relative_residual=(
            counters.maximum_primal_krylov_relative_residual
        ),
        maximum_adjoint_krylov_relative_residual=(
            counters.maximum_adjoint_krylov_relative_residual
        ),
        primal_seconds=timings.get("primal", 0.0),
        adjoint_seconds=timings.get("adjoint", 0.0),
        local_backward_seconds=timings.get("local_backward", 0.0),
        dense_loss_seconds=timings.get("dense_loss", 0.0),
        optimizer_seconds=timings.get("optimizer", 0.0),
        canonicalization_seconds=timings.get("canonicalization", 0.0),
        lift_seconds=timings.get("lift", 0.0),
        audit_seconds=timings.get("audit", 0.0),
        observation_wall_seconds=perf_counter() - wall_start,
        raw_euler_gap_rmse=raw_euler_gap_rmse,
        flip_count=metrics.flip_count,
        minimum_signed_area=metrics.minimum_signed_area,
        minimum_area_ratio=metrics.minimum_area_ratio,
        boundary_order_min_gap=metrics.boundary_order_min_gap,
        topology_certified=metrics.global_injectivity_certificate,
        map_rmse=metrics.map_rmse,
        maximum_map_error=metrics.maximum_map_error,
        mu_rmse=metrics.mu_rmse,
        maximum_mu_error=metrics.maximum_mu_error,
        logit_spread=audit.logit_spread,
        maximum_mvc_canonical_covariance_condition=(
            audit.maximum_mvc_canonical_covariance_condition
        ),
        system_condition_number=audit.system_condition_number,
        system_condition_method=audit.system_condition_method,
        canonicality_error=audit.canonicality_error,
    )


def _failure(phase: str, step: int, error: Exception) -> OptimizationFailure:
    return OptimizationFailure(phase, step, type(error).__name__, str(error))


def _threshold(
    trace: list[MVCOptimizationTraceRow], threshold: float | None
) -> tuple[int | None, int | None, float | None]:
    if threshold is None:
        return None, None, None
    for row in trace:
        if row.objective <= threshold:
            return row.iteration, row.global_solves, row.observation_wall_seconds
    return None, None, None


def _finish(
    *,
    method: str,
    task: str,
    backend: str,
    learning_rate: float,
    steps: int,
    counters: _Counters,
    threshold: float | None,
    wall_start: float,
    failure: OptimizationFailure | None,
    initial_control: torch.Tensor,
    final_control: torch.Tensor,
    final_logits: torch.Tensor,
    fixed_boundary: torch.Tensor,
    trace: list[MVCOptimizationTraceRow],
) -> MVCMethodResult:
    if not trace:
        raise RuntimeError("an optimizer result requires at least one accepted state")
    accepted_steps, threshold_solves, threshold_wall = _threshold(trace, threshold)
    last = trace[-1]
    total_wall = perf_counter() - wall_start
    algorithm_wall = sum(
        row.primal_seconds
        + row.adjoint_seconds
        + row.local_backward_seconds
        + row.dense_loss_seconds
        + row.optimizer_seconds
        + row.canonicalization_seconds
        + row.lift_seconds
        for row in trace
    )
    audit_wall = sum(row.audit_seconds for row in trace)
    optimizer_semantics = {
        "O1_sigmoid_positive": "Torch Adam in bounded sigmoid-positive raw-weight coordinates",
        "O2_row_softmax": "Torch Adam in directed row-softmax logit coordinates",
        "O3_mvc_adam": (
            "projected Torch Adam: latent step, decoder candidate, MVC projection, "
            "then canonical re-decode"
        ),
        "O4_covariance_retraction": (
            "manual Adam in fixed-boundary vertex coordinates followed by MVC "
            "covariance lift and decoder retraction"
        ),
    }[method]
    moment_policy = {
        "O1_sigmoid_positive": "native Torch Adam moments",
        "O2_row_softmax": "native Torch Adam moments",
        "O3_mvc_adam": (
            "retain moments with identity transport across each nonlinear MVC projection"
        ),
        "O4_covariance_retraction": (
            "retain moments in fixed global vertex coordinates; boundary direction is zeroed"
        ),
    }[method]
    return MVCMethodResult(
        method=method,
        task=task,
        backend=backend,
        boundary_semantics="fixed_to_target_boundary; O4 boundary tangent is exactly zero",
        optimizer_semantics=optimizer_semantics,
        moment_policy=moment_policy,
        learning_rate=learning_rate,
        steps_requested=steps,
        completed_steps=last.iteration,
        initial_objective=trace[0].objective,
        final_objective=last.objective,
        best_objective=min(row.objective for row in trace),
        primal_solves=counters.primal,
        adjoint_solves=counters.adjoint,
        global_solves=counters.primal + counters.adjoint,
        local_backwards=counters.local_backward,
        primal_attempts=counters.primal_attempts,
        adjoint_attempts=counters.adjoint_attempts,
        local_backward_attempts=counters.local_backward_attempts,
        primal_krylov_rhs_iterations=counters.primal_krylov_rhs_iterations,
        adjoint_krylov_rhs_iterations=counters.adjoint_krylov_rhs_iterations,
        maximum_primal_krylov_iterations=counters.maximum_primal_krylov_iterations,
        maximum_adjoint_krylov_iterations=counters.maximum_adjoint_krylov_iterations,
        maximum_primal_krylov_relative_residual=(
            counters.maximum_primal_krylov_relative_residual
        ),
        maximum_adjoint_krylov_relative_residual=(
            counters.maximum_adjoint_krylov_relative_residual
        ),
        objective_threshold=threshold,
        accepted_steps_to_threshold=accepted_steps,
        global_solves_to_threshold=threshold_solves,
        wall_seconds_to_threshold=threshold_wall,
        wall_seconds=total_wall,
        algorithm_wall_seconds=algorithm_wall,
        audit_wall_seconds=audit_wall,
        unattributed_wall_seconds=max(0.0, total_wall - algorithm_wall - audit_wall),
        all_iterates_certified=all(row.topology_certified for row in trace),
        final_metrics=last_to_metrics(last),
        final_logit_spread=last.logit_spread,
        maximum_mvc_canonical_covariance_condition=max(
            row.maximum_mvc_canonical_covariance_condition for row in trace
        ),
        final_system_condition_number=last.system_condition_number,
        system_condition_method=last.system_condition_method,
        final_canonicality_error=last.canonicality_error,
        failure=failure,
        initial_control=initial_control.detach().cpu().clone(),
        final_control=final_control.detach().cpu().clone(),
        final_logits=final_logits.detach().cpu().clone(),
        fixed_boundary=fixed_boundary.detach().cpu().clone(),
        trace=tuple(trace),
    )


def last_to_metrics(row: MVCOptimizationTraceRow) -> P1MapMetrics:
    """Reconstruct the common metric record from a trace row."""

    return P1MapMetrics(
        flip_count=row.flip_count,
        minimum_signed_area=row.minimum_signed_area,
        minimum_area_ratio=row.minimum_area_ratio,
        boundary_order_min_gap=row.boundary_order_min_gap,
        global_injectivity_certificate=row.topology_certified,
        map_rmse=row.map_rmse,
        maximum_map_error=row.maximum_map_error,
        mu_rmse=row.mu_rmse,
        maximum_mu_error=row.maximum_mu_error,
    )


def _run_latent_method(
    *,
    method: str,
    task: str,
    mesh: TriMesh,
    query_table: StructuredDenseQueryTable,
    target_control: torch.Tensor,
    target_dense: torch.Tensor,
    fixed_boundary: torch.Tensor,
    moving: torch.Tensor | None,
    fixed: torch.Tensor | None,
    backend: str,
    steps: int,
    learning_rate: float,
    dtype: torch.dtype,
    device: torch.device,
    objective_threshold: float | None,
    system_condition_dense_limit: int,
) -> MVCMethodResult:
    solver = _make_solver(mesh, backend=backend, dtype=dtype, device=device)
    encoder = MeanValueCoordinateEncoder(mesh).to(device=device, dtype=dtype)
    system = solver.system
    counters = _Counters()
    trace: list[MVCOptimizationTraceRow] = []
    wall_start = perf_counter()
    uniform = torch.zeros(
        (system.n_rows, system.max_degree), dtype=dtype, device=device
    )
    initialization_canonicalization = 0.0
    initialization_primal = 0.0

    if method == "O1_sigmoid_positive":
        latent = torch.nn.Parameter(torch.zeros_like(uniform))
        transform = torch_functional.logsigmoid
        current_logits = transform(latent)
        initial_control, elapsed = _decode(
            solver, current_logits, fixed_boundary, counters
        )
        initialization_primal += elapsed
        current_control = initial_control
    elif method in ("O2_row_softmax", "O3_mvc_adam"):
        if method == "O3_mvc_adam":
            initial_control, elapsed = _decode(
                solver, uniform, fixed_boundary, counters
            )
            initialization_primal += elapsed
            _synchronize(device)
            start = perf_counter()
            with torch.no_grad():
                canonical = encoder(initial_control.detach())
            _synchronize(device)
            initialization_canonicalization = perf_counter() - start
            latent = torch.nn.Parameter(canonical.logits.detach().clone())
            current_logits = latent
            current_control, elapsed = _decode(
                solver, current_logits, fixed_boundary, counters
            )
            initialization_primal += elapsed
        else:
            latent = torch.nn.Parameter(torch.zeros_like(uniform))
            current_logits = latent
            initial_control, elapsed = _decode(
                solver, current_logits, fixed_boundary, counters
            )
            initialization_primal += elapsed
            current_control = initial_control
        transform = lambda value: value
    else:
        raise ValueError(f"unsupported latent method: {method}")

    optimizer = torch.optim.Adam([latent], lr=learning_rate)
    current_logits = transform(latent)
    loss, initial_dense_loss = _timed_dense_loss(
        task, query_table, current_control, target_dense, moving=moving, fixed=fixed
    )
    _synchronize(device)
    audit_start = perf_counter()
    audit = _audit(
        mesh,
        encoder,
        current_control,
        current_logits,
        target_control,
        system_condition_dense_limit=system_condition_dense_limit,
    )
    _synchronize(device)
    audit_seconds = perf_counter() - audit_start
    trace.append(
        _trace_row(
            iteration=0,
            loss=loss,
            counters=counters,
            timings={
                "primal": initialization_primal,
                "dense_loss": initial_dense_loss,
                "canonicalization": initialization_canonicalization,
                "audit": audit_seconds,
            },
            wall_start=wall_start,
            audit=audit,
        )
    )
    accepted_control = current_control.detach().clone()
    accepted_logits = current_logits.detach().clone()
    failure: OptimizationFailure | None = None

    for step in range(1, steps + 1):
        timings: dict[str, float] = {}
        active_phase = "loss_and_decoder_adjoint_backward"
        try:
            optimizer.zero_grad(set_to_none=True)
            counters.adjoint_attempts += 1
            _synchronize(device)
            start = perf_counter()
            loss.backward()
            _synchronize(device)
            timings["adjoint"] = perf_counter() - start
            counters.adjoint += 1
            _record_krylov(counters, solver, phase="adjoint")
            active_phase = "latent_gradient_validation"
            if latent.grad is None:
                raise RuntimeError("loss backward did not produce a latent gradient")
            if not bool(torch.isfinite(latent.grad).all()):
                raise ValueError("latent gradient must remain finite")

            active_phase = "adam_step"
            _synchronize(device)
            start = perf_counter()
            optimizer.step()
            _synchronize(device)
            timings["optimizer"] = perf_counter() - start

            if method == "O3_mvc_adam":
                candidate_logits = latent
                active_phase = "candidate_decode"
                with torch.no_grad():
                    candidate, elapsed = _decode(
                        solver, candidate_logits, fixed_boundary, counters
                    )
                timings["primal"] = elapsed
                active_phase = "mvc_reencode"
                _synchronize(device)
                start = perf_counter()
                with torch.no_grad():
                    canonical = encoder(candidate)
                    latent.copy_(canonical.logits)
                _synchronize(device)
                timings["canonicalization"] = perf_counter() - start
                current_logits = latent
                active_phase = "canonical_redecode"
                current_control, elapsed = _decode(
                    solver, current_logits, fixed_boundary, counters
                )
                timings["primal"] += elapsed
            else:
                current_logits = transform(latent)
                active_phase = "accepted_decode"
                current_control, elapsed = _decode(
                    solver, current_logits, fixed_boundary, counters
                )
                timings["primal"] = elapsed

            active_phase = "dense_loss"
            loss, timings["dense_loss"] = _timed_dense_loss(
                task,
                query_table,
                current_control,
                target_dense,
                moving=moving,
                fixed=fixed,
            )
            active_phase = "independent_audit"
            _synchronize(device)
            start = perf_counter()
            audit = _audit(
                mesh,
                encoder,
                current_control,
                current_logits,
                target_control,
                system_condition_dense_limit=system_condition_dense_limit,
            )
            _synchronize(device)
            timings["audit"] = perf_counter() - start
            trace.append(
                _trace_row(
                    iteration=step,
                    loss=loss,
                    counters=counters,
                    timings=timings,
                    wall_start=wall_start,
                    audit=audit,
                )
            )
            accepted_control = current_control.detach().clone()
            accepted_logits = current_logits.detach().clone()
        except Exception as error:  # preserve the shared-rate failure as data
            failure = _failure(active_phase, step, error)
            break

    return _finish(
        method=method,
        task=task,
        backend=backend,
        learning_rate=learning_rate,
        steps=steps,
        counters=counters,
        threshold=objective_threshold,
        wall_start=wall_start,
        failure=failure,
        initial_control=initial_control,
        final_control=accepted_control,
        final_logits=accepted_logits,
        fixed_boundary=fixed_boundary,
        trace=trace,
    )


def _run_covariance_method(
    *,
    task: str,
    mesh: TriMesh,
    query_table: StructuredDenseQueryTable,
    target_control: torch.Tensor,
    target_dense: torch.Tensor,
    fixed_boundary: torch.Tensor,
    moving: torch.Tensor | None,
    fixed: torch.Tensor | None,
    backend: str,
    steps: int,
    learning_rate: float,
    dtype: torch.dtype,
    device: torch.device,
    objective_threshold: float | None,
    system_condition_dense_limit: int,
    max_covariance_condition: float,
) -> MVCMethodResult:
    method = "O4_covariance_retraction"
    solver = _make_solver(mesh, backend=backend, dtype=dtype, device=device)
    encoder = MeanValueCoordinateEncoder(mesh).to(device=device, dtype=dtype)
    lift = CovarianceLogitLift(
        solver.system, max_covariance_condition=max_covariance_condition
    ).to(device=device, dtype=dtype)
    counters = _Counters()
    trace: list[MVCOptimizationTraceRow] = []
    wall_start = perf_counter()
    uniform = torch.zeros(
        (solver.system.n_rows, solver.system.max_degree), dtype=dtype, device=device
    )
    initial_control, initial_primal = _decode(solver, uniform, fixed_boundary, counters)
    _synchronize(device)
    start = perf_counter()
    with torch.no_grad():
        canonical = encoder(initial_control.detach())
    _synchronize(device)
    initial_canonicalization = perf_counter() - start
    current_logits = canonical.logits.detach().clone()
    current_control, elapsed = _decode(solver, current_logits, fixed_boundary, counters)
    initial_primal += elapsed
    current_control = current_control.detach().requires_grad_(True)
    _synchronize(device)
    start = perf_counter()
    with torch.no_grad():
        current_canonical = encoder(current_control.detach())
    _synchronize(device)
    initial_canonicalization += perf_counter() - start
    current_logits = current_canonical.logits.detach().clone()
    loss, initial_dense_loss = _timed_dense_loss(
        task, query_table, current_control, target_dense, moving=moving, fixed=fixed
    )
    _synchronize(device)
    start = perf_counter()
    audit = _audit(
        mesh,
        encoder,
        current_control,
        current_logits,
        target_control,
        system_condition_dense_limit=system_condition_dense_limit,
    )
    _synchronize(device)
    audit_seconds = perf_counter() - start
    trace.append(
        _trace_row(
            iteration=0,
            loss=loss,
            counters=counters,
            timings={
                "primal": initial_primal,
                "dense_loss": initial_dense_loss,
                "canonicalization": initial_canonicalization,
                "audit": audit_seconds,
            },
            wall_start=wall_start,
            audit=audit,
        )
    )
    accepted_control = current_control.detach().clone()
    accepted_logits = current_logits.detach().clone()

    first_moment = torch.zeros_like(current_control)
    second_moment = torch.zeros_like(current_control)
    boundary_index = torch.as_tensor(
        solver.system.loop, dtype=torch.int64, device=device
    )
    beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8
    failure: OptimizationFailure | None = None

    for step in range(1, steps + 1):
        timings: dict[str, float] = {}
        active_phase = "local_loss_backward"
        try:
            counters.local_backward_attempts += 1
            _synchronize(device)
            start = perf_counter()
            loss.backward()
            _synchronize(device)
            timings["local_backward"] = perf_counter() - start
            counters.local_backward += 1
            if current_control.grad is None:
                raise RuntimeError(
                    "vertex-space loss did not produce a control gradient"
                )

            active_phase = "vertex_adam_step"
            _synchronize(device)
            start = perf_counter()
            gradient = current_control.grad.detach()
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient.square()
            corrected_first = first_moment / (1.0 - beta1**step)
            corrected_second = second_moment / (1.0 - beta2**step)
            direction = (
                -learning_rate
                * corrected_first
                / (torch.sqrt(corrected_second) + epsilon)
            )
            direction = direction.clone()
            direction.index_fill_(0, boundary_index, 0.0)
            _synchronize(device)
            timings["optimizer"] = perf_counter() - start

            active_phase = "covariance_lift"
            _synchronize(device)
            start = perf_counter()
            with torch.no_grad():
                lifted = lift(
                    current_control.detach(),
                    current_canonical.probabilities,
                    direction,
                )
                updated_logits = current_canonical.logits + lifted.delta_logits
            _synchronize(device)
            timings["lift"] = perf_counter() - start

            raw_euler = current_control.detach() + direction
            active_phase = "retraction_decode"
            updated_control, elapsed = _decode(
                solver, updated_logits, fixed_boundary, counters
            )
            timings["primal"] = elapsed
            gap = torch.sqrt(
                torch.mean(
                    torch.sum((updated_control.detach() - raw_euler).square(), dim=-1)
                )
            )
            raw_euler_gap_rmse = float(gap.cpu())

            current_control = updated_control.detach().requires_grad_(True)
            active_phase = "mvc_reencode"
            _synchronize(device)
            start = perf_counter()
            with torch.no_grad():
                current_canonical = encoder(current_control.detach())
                current_logits = current_canonical.logits.detach().clone()
            _synchronize(device)
            timings["canonicalization"] = perf_counter() - start
            active_phase = "dense_loss"
            loss, timings["dense_loss"] = _timed_dense_loss(
                task,
                query_table,
                current_control,
                target_dense,
                moving=moving,
                fixed=fixed,
            )
            active_phase = "independent_audit"
            _synchronize(device)
            start = perf_counter()
            audit = _audit(
                mesh,
                encoder,
                current_control,
                current_logits,
                target_control,
                system_condition_dense_limit=system_condition_dense_limit,
            )
            _synchronize(device)
            timings["audit"] = perf_counter() - start
            trace.append(
                _trace_row(
                    iteration=step,
                    loss=loss,
                    counters=counters,
                    timings=timings,
                    wall_start=wall_start,
                    audit=audit,
                    raw_euler_gap_rmse=raw_euler_gap_rmse,
                )
            )
            accepted_control = current_control.detach().clone()
            accepted_logits = current_logits.detach().clone()
        except Exception as error:  # preserve the shared-rate failure as data
            failure = _failure(active_phase, step, error)
            break

    return _finish(
        method=method,
        task=task,
        backend=backend,
        learning_rate=learning_rate,
        steps=steps,
        counters=counters,
        threshold=objective_threshold,
        wall_start=wall_start,
        failure=failure,
        initial_control=initial_control,
        final_control=accepted_control,
        final_logits=accepted_logits,
        fixed_boundary=fixed_boundary,
        trace=trace,
    )


def extract_outer_step_slice(
    comparison: FixedBoundaryComparison,
    outer_step: int,
) -> MVCComparisonSlice:
    """Select the same accepted outer step without calling it equal-solve evidence."""

    if (
        isinstance(outer_step, bool)
        or not isinstance(outer_step, int)
        or outer_step < 0
    ):
        raise ValueError("outer_step must be a nonnegative integer")
    rows: dict[str, MVCOptimizationTraceRow] = {}
    missing: dict[str, str] = {}
    for method, result in comparison.results.items():
        match = next((row for row in result.trace if row.iteration == outer_step), None)
        if match is None:
            suffix = f"; first failure={result.failure.phase}" if result.failure else ""
            missing[method] = f"no accepted outer step {outer_step}{suffix}"
        else:
            rows[method] = match
    return MVCComparisonSlice("common_outer_step_secondary", outer_step, rows, missing)


def extract_exact_global_solve_slice(
    comparison: FixedBoundaryComparison,
    global_solves: int,
) -> MVCComparisonSlice:
    """Select accepted states at exactly one completed global-solve budget."""

    if (
        isinstance(global_solves, bool)
        or not isinstance(global_solves, int)
        or global_solves < 1
    ):
        raise ValueError("global_solves must be a positive integer")
    rows: dict[str, MVCOptimizationTraceRow] = {}
    missing: dict[str, str] = {}
    for method, result in comparison.results.items():
        match = next(
            (row for row in result.trace if row.global_solves == global_solves), None
        )
        if match is None:
            available = [row.global_solves for row in result.trace]
            suffix = f"; first failure={result.failure.phase}" if result.failure else ""
            missing[method] = (
                f"no accepted state at {global_solves} completed global solves; "
                f"available={available}{suffix}"
            )
        else:
            rows[method] = match
    return MVCComparisonSlice(
        "exact_global_solve_primary", global_solves, rows, missing
    )


def run_fixed_boundary_comparison(
    *,
    task: str,
    control_vertices: int,
    image_resolution: int,
    steps: int,
    backend: str,
    dtype: torch.dtype,
    seed: int,
    target_strength: float,
    learning_rate: float,
    image_name: str = "smooth_blobs",
    device: torch.device | str = "cpu",
    methods: Iterable[str] = METHOD_NAMES,
    target: DirectedTarget | None = None,
    objective_threshold: float | None = None,
    system_condition_dense_limit: int = 256,
    max_covariance_condition: float = 1.0e8,
) -> FixedBoundaryComparison:
    """Run a controlled O1--O4 comparison on one fixed-boundary problem.

    Equal learning rates are part of the requested primary comparison.  The
    function does not retune a failed method.  A caller may run a separately
    labeled pilot sweep, as recommended in ``pilot_recommendation``.
    """

    if task not in ("supervised_map", "image_registration"):
        raise ValueError("task must be 'supervised_map' or 'image_registration'")
    if (
        isinstance(control_vertices, bool)
        or not isinstance(control_vertices, int)
        or control_vertices < 3
    ):
        raise ValueError("control_vertices must be an integer at least three")
    if (
        isinstance(image_resolution, bool)
        or not isinstance(image_resolution, int)
        or image_resolution < 2
    ):
        raise ValueError("image_resolution must be an integer at least two")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    if not np.isfinite(target_strength) or target_strength <= 0.0:
        raise ValueError("target_strength must be finite and positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if objective_threshold is not None and (
        not np.isfinite(objective_threshold) or objective_threshold < 0.0
    ):
        raise ValueError("objective_threshold must be finite and nonnegative")
    if (
        isinstance(system_condition_dense_limit, bool)
        or not isinstance(system_condition_dense_limit, int)
        or system_condition_dense_limit < 0
    ):
        raise ValueError("system_condition_dense_limit must be a nonnegative integer")
    selected = tuple(methods)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("methods must be a nonempty sequence without duplicates")
    unknown = set(selected) - set(METHOD_NAMES)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")

    device = torch.device(device)
    mesh = structured_rectangle(control_vertices - 1, control_vertices - 1)
    target_setup_start = perf_counter()
    if target is None:
        target = build_directed_target(
            mesh,
            image_height=image_resolution,
            image_width=image_resolution,
            seed=seed,
            strength=target_strength,
            height=1.0 if task == "image_registration" else 0.9,
        )
        target_setup_primal_solves = 1
    else:
        _validate_prebuilt_target(target, mesh, image_resolution=image_resolution)
        target_setup_primal_solves = 0
    target_setup_seconds = perf_counter() - target_setup_start

    target_control = target.control.detach().to(device=device, dtype=dtype)
    target_dense = target.dense.detach().to(device=device, dtype=dtype)
    loop = torch.as_tensor(
        mesh.boundary_loops[0].copy(), dtype=torch.int64, device=device
    )
    fixed_boundary = target_control.index_select(0, loop).detach().clone()
    query_table = StructuredDenseQueryTable.from_mesh(
        mesh, height=image_resolution, width=image_resolution
    )
    query_table.prepare(device=device, dtype=dtype)

    moving = fixed = None
    if task == "image_registration":
        moving = synthetic_image(
            image_name,
            height=image_resolution,
            width=image_resolution,
            dtype=dtype,
            device=device,
        )
        fixed = warp_image_backward(moving, target_dense)

    results: OrderedDict[str, MVCMethodResult] = OrderedDict()
    common = dict(
        task=task,
        mesh=mesh,
        query_table=query_table,
        target_control=target_control,
        target_dense=target_dense,
        fixed_boundary=fixed_boundary,
        moving=moving,
        fixed=fixed,
        backend=backend,
        steps=steps,
        learning_rate=float(learning_rate),
        dtype=dtype,
        device=device,
        objective_threshold=objective_threshold,
        system_condition_dense_limit=system_condition_dense_limit,
    )
    for method in selected:
        if method == "O4_covariance_retraction":
            result = _run_covariance_method(
                **common,
                max_covariance_condition=max_covariance_condition,
            )
        else:
            result = _run_latent_method(method=method, **common)
        results[method] = result

    uniform_initial_control = next(iter(results.values())).initial_control
    for result in results.values():
        if not torch.allclose(
            result.initial_control,
            uniform_initial_control,
            atol=256.0 * torch.finfo(dtype).eps,
            rtol=256.0 * torch.finfo(dtype).eps,
        ):
            raise RuntimeError(
                "methods did not start from the same uniform-row decoded map"
            )

    return FixedBoundaryComparison(
        task=task,
        backend=backend,
        control_vertices=control_vertices,
        image_resolution=image_resolution,
        steps=steps,
        learning_rate=float(learning_rate),
        seed=seed,
        target_strength=float(target_strength),
        image_name=image_name if task == "image_registration" else None,
        warp_convention="backward_map_fixed_to_moving",
        boundary_semantics="all methods fix the identical target boundary; O4 uses zero boundary tangent",
        target_setup_primal_solves=target_setup_primal_solves,
        target_setup_seconds=target_setup_seconds,
        target_control=target_control.detach().cpu().clone(),
        target_dense=target_dense.detach().cpu().clone(),
        fixed_target_boundary=fixed_boundary.detach().cpu().clone(),
        uniform_initial_control=uniform_initial_control.detach().cpu().clone(),
        results=dict(results),
        primary_global_solve_budget=83,
        primary_expected_accepted_steps={
            "O1_sigmoid_positive": 41,
            "O2_row_softmax": 41,
            "O3_mvc_adam": 27,
            "O4_covariance_retraction": 81,
        },
        secondary_outer_step=40,
        secondary_expected_global_solves={
            "O1_sigmoid_positive": 81,
            "O2_row_softmax": 81,
            "O3_mvc_adam": 122,
            "O4_covariance_retraction": 42,
        },
        shared_learning_rate_semantics=(
            "robustness cohort only: equal numeric learning rates have different units in "
            "raw-weight, logit, and vertex-coordinate parameterizations"
        ),
        pilot_recommendation=(
            "Primary evidence must keep this shared learning rate unchanged and preserve failures. "
            "A separately labeled logarithmic pilot sweep may choose one stable rate per method "
            "before a second, explicitly non-equal-rate efficiency comparison."
        ),
    )


__all__ = [
    "FixedBoundaryComparison",
    "METHOD_NAMES",
    "MVCComparisonSlice",
    "MVCMethodResult",
    "MVCOptimizationTraceRow",
    "OptimizationFailure",
    "extract_exact_global_solve_slice",
    "extract_outer_step_slice",
    "last_to_metrics",
    "run_fixed_boundary_comparison",
]
