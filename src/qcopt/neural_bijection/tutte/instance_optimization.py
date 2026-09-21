"""No-network instance optimization for the Phase V Tutte decoder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import torch

from ...mesh import TriMesh, structured_rectangle
from ..benchmarks import synthetic_image, warp_image_backward
from ..metrics import P1MapMetrics, compute_p1_map_metrics
from .decoder import TutteRectangleDecoder
from .dense_warp import StructuredDenseQueryTable
from .direct import DirectTutteLayer
from .iterative import MatrixFreeDirectedTutteLayer
from .symmetric import MatrixFreeSymmetricTutteLayer


@dataclass(frozen=True)
class DirectedTarget:
    interior_logits: torch.Tensor
    boundary_logits: torch.Tensor
    raw_modulus: torch.Tensor
    control: torch.Tensor
    dense: torch.Tensor
    metrics: P1MapMetrics


@dataclass(frozen=True)
class InstanceTraceRow:
    iteration: int
    outer_step: int
    objective: float
    map_rmse: float
    forward_seconds: float
    backward_seconds: float
    audit_seconds: float
    optimizer_step_seconds: float
    observation_wall_seconds: float
    primal_solves: int
    adjoint_solves: int
    global_solves: int
    flip_count: int
    minimum_area_ratio: float
    topology_certified: bool


@dataclass(frozen=True)
class InstanceOptimizationResult:
    task: str
    backend: str
    optimizer_name: str
    control_vertices_per_side: int
    control_vertex_count: int
    image_resolution: int
    initial_objective: float
    final_objective: float
    best_objective: float
    primal_solves: int
    adjoint_solves: int
    global_solves: int
    target_setup_solves: int
    target_setup_seconds: float
    wall_seconds: float
    objective_threshold: float | None
    evaluations_to_threshold: int | None
    global_solves_to_threshold: int | None
    wall_seconds_to_threshold: float | None
    all_iterates_certified: bool
    final_metrics: P1MapMetrics
    trace: tuple[InstanceTraceRow, ...]


def build_directed_target(
    mesh: TriMesh,
    *,
    image_height: int,
    image_width: int,
    seed: int,
    strength: float,
    height: float,
) -> DirectedTarget:
    """Generate one deterministic positive-directed-Tutte target.

    The target belongs to the directed Tutte model family up to the numerical
    accuracy of the CPU float64 reference solve.  It need not belong to the
    smaller symmetric-conductance family.
    """
    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError("strength must be finite and positive")
    solver = DirectTutteLayer(mesh)
    decoder = TutteRectangleDecoder(
        mesh,
        solver,
        image_height=image_height,
        image_width=image_width,
        minimum_height=0.05,
    )
    generator = torch.Generator().manual_seed(seed)
    interior_logits = strength * torch.randn(
        (solver.system.n_rows, solver.system.max_degree),
        generator=generator,
        dtype=torch.float64,
    )
    boundary_logits = strength * torch.randn(
        decoder.boundary.n_segments,
        generator=generator,
        dtype=torch.float64,
    )
    raw_modulus = torch.tensor(
        decoder.boundary.raw_modulus_for_height(height), dtype=torch.float64
    )
    with torch.no_grad():
        decoded = decoder(interior_logits, boundary_logits, raw_modulus)
    metrics = compute_p1_map_metrics(mesh, decoded.control.numpy())
    if not metrics.global_injectivity_certificate:
        raise RuntimeError("generated positive-Tutte target failed its independent certificate")
    return DirectedTarget(
        interior_logits=interior_logits,
        boundary_logits=boundary_logits,
        raw_modulus=raw_modulus,
        control=decoded.control.detach().clone(),
        dense=decoded.dense.detach().clone(),
        metrics=metrics,
    )


def _training_decoder(
    mesh: TriMesh,
    *,
    backend: str,
    image_resolution: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[TutteRectangleDecoder, tuple[int, ...]]:
    if backend == "direct":
        if device.type != "cpu":
            raise ValueError("direct backend is CPU-only")
        solver = DirectTutteLayer(mesh)
        latent_shape = (solver.system.n_rows, solver.system.max_degree)
    elif backend == "directed_iterative":
        solver = MatrixFreeDirectedTutteLayer(mesh)
        latent_shape = (solver.system.n_rows, solver.system.max_degree)
    elif backend == "symmetric":
        solver = MatrixFreeSymmetricTutteLayer(mesh)
        latent_shape = (solver.n_conductances,)
    else:
        raise ValueError(f"unknown Tutte backend: {backend}")
    decoder = TutteRectangleDecoder(
        mesh,
        solver,
        image_height=image_resolution,
        image_width=image_resolution,
        minimum_height=0.05,
    ).to(device=device)
    decoder.prepare(device=device, dtype=dtype)
    return decoder, latent_shape


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _optimization_variables(
    decoder: TutteRectangleDecoder,
    latent_shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    learn_modulus: bool,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.Tensor, list[torch.Tensor]]:
    latent = torch.nn.Parameter(torch.zeros(latent_shape, dtype=dtype, device=device))
    boundary_logits = torch.nn.Parameter(
        torch.zeros(decoder.boundary.n_segments, dtype=dtype, device=device)
    )
    raw_value = decoder.boundary.raw_modulus_for_height(1.0)
    if learn_modulus:
        raw_modulus: torch.Tensor = torch.nn.Parameter(
            torch.tensor(raw_value, dtype=dtype, device=device)
        )
        parameters: list[torch.Tensor] = [latent, boundary_logits, raw_modulus]
    else:
        raw_modulus = torch.tensor(raw_value, dtype=dtype, device=device)
        parameters = [latent, boundary_logits]
    return latent, boundary_logits, raw_modulus, parameters


def _instance_loss(
    task: str,
    dense: torch.Tensor,
    target_dense: torch.Tensor,
    *,
    moving: torch.Tensor | None,
    fixed: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if task == "supervised_map":
        return (dense - target_dense).square().mean()
    if moving is None or fixed is None:
        raise RuntimeError("image task requires moving and fixed images")
    tolerance = 64.0 * torch.finfo(dtype).eps
    if bool(torch.any(dense < -tolerance)) or bool(torch.any(dense > 1.0 + tolerance)):
        raise RuntimeError("image optimization produced an out-of-domain backward map")
    return (warp_image_backward(moving, dense) - fixed).square().mean()


def _threshold_fields(
    trace: list[InstanceTraceRow], objective_threshold: float | None
) -> tuple[int | None, int | None, float | None]:
    if objective_threshold is None:
        return None, None, None
    for row in trace:
        if row.objective <= objective_threshold:
            return row.iteration + 1, row.global_solves, row.observation_wall_seconds
    return None, None, None


def _make_result(
    *,
    task: str,
    decoder: TutteRectangleDecoder,
    mesh: TriMesh,
    optimizer_name: str,
    trace: list[InstanceTraceRow],
    primal_solves: int,
    adjoint_solves: int,
    target_setup_solves: int,
    target_setup_seconds: float,
    objective_threshold: float | None,
    wall_seconds: float,
    final_metrics: P1MapMetrics,
) -> InstanceOptimizationResult:
    evaluations, solves, threshold_wall = _threshold_fields(trace, objective_threshold)
    side = int(round(np.sqrt(mesh.n_vertices)))
    if side * side != mesh.n_vertices:
        raise RuntimeError("instance runner currently requires a square structured control mesh")
    return InstanceOptimizationResult(
        task=task,
        backend=decoder.solver.__class__.__name__,
        optimizer_name=optimizer_name,
        control_vertices_per_side=side,
        control_vertex_count=mesh.n_vertices,
        image_resolution=decoder.image_height,
        initial_objective=trace[0].objective,
        final_objective=trace[-1].objective,
        best_objective=min(row.objective for row in trace),
        primal_solves=primal_solves,
        adjoint_solves=adjoint_solves,
        global_solves=primal_solves + adjoint_solves,
        target_setup_solves=target_setup_solves,
        target_setup_seconds=target_setup_seconds,
        wall_seconds=wall_seconds,
        objective_threshold=objective_threshold,
        evaluations_to_threshold=evaluations,
        global_solves_to_threshold=solves,
        wall_seconds_to_threshold=threshold_wall,
        all_iterates_certified=all(row.topology_certified for row in trace),
        final_metrics=final_metrics,
        trace=tuple(trace),
    )


def _run_adam(
    *,
    task: str,
    mesh: TriMesh,
    decoder: TutteRectangleDecoder,
    latent_shape: tuple[int, ...],
    target: DirectedTarget,
    steps: int,
    learning_rate: float,
    dtype: torch.dtype,
    device: torch.device,
    moving: torch.Tensor | None,
    fixed: torch.Tensor | None,
    learn_modulus: bool,
    target_setup_solves: int,
    target_setup_seconds: float,
    objective_threshold: float | None,
) -> InstanceOptimizationResult:
    latent, boundary_logits, raw_modulus, parameters = _optimization_variables(
        decoder, latent_shape, dtype=dtype, device=device, learn_modulus=learn_modulus
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    target_dense = target.dense.detach().to(device=device, dtype=dtype)
    target_control = target.control.detach().cpu().numpy().copy()
    trace: list[InstanceTraceRow] = []
    start = perf_counter()
    primal_solves = 0
    adjoint_solves = 0

    for iteration in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _synchronize(device)
        forward_start = perf_counter()
        decoded = decoder(latent, boundary_logits, raw_modulus)
        primal_solves += 1
        loss = _instance_loss(
            task,
            decoded.dense,
            target_dense,
            moving=moving,
            fixed=fixed,
            dtype=dtype,
        )
        _synchronize(device)
        forward_seconds = perf_counter() - forward_start

        audit_start = perf_counter()
        mapped = decoded.control.detach().to(device="cpu", dtype=torch.float64).numpy()
        metrics = compute_p1_map_metrics(mesh, mapped, target=target_control)
        audit_seconds = perf_counter() - audit_start
        observation_wall_seconds = perf_counter() - start
        observed_primal = primal_solves
        observed_adjoint = adjoint_solves

        backward_seconds = 0.0
        optimizer_seconds = 0.0
        if iteration < steps:
            _synchronize(device)
            backward_start = perf_counter()
            loss.backward()
            _synchronize(device)
            backward_seconds = perf_counter() - backward_start
            adjoint_solves += 1
            _synchronize(device)
            optimizer_start = perf_counter()
            optimizer.step()
            _synchronize(device)
            optimizer_seconds = perf_counter() - optimizer_start

        trace.append(
            InstanceTraceRow(
                iteration=iteration,
                outer_step=iteration,
                objective=float(loss.detach().cpu()),
                map_rmse=metrics.map_rmse,
                forward_seconds=forward_seconds,
                backward_seconds=backward_seconds,
                audit_seconds=audit_seconds,
                optimizer_step_seconds=optimizer_seconds,
                observation_wall_seconds=observation_wall_seconds,
                primal_solves=observed_primal,
                adjoint_solves=observed_adjoint,
                global_solves=observed_primal + observed_adjoint,
                flip_count=metrics.flip_count,
                minimum_area_ratio=metrics.minimum_area_ratio,
                topology_certified=metrics.global_injectivity_certificate,
            )
        )

    return _make_result(
        task=task,
        decoder=decoder,
        mesh=mesh,
        optimizer_name="adam",
        trace=trace,
        primal_solves=primal_solves,
        adjoint_solves=adjoint_solves,
        target_setup_solves=target_setup_solves,
        target_setup_seconds=target_setup_seconds,
        objective_threshold=objective_threshold,
        wall_seconds=perf_counter() - start,
        final_metrics=metrics,
    )


def _run_lbfgs(
    *,
    task: str,
    mesh: TriMesh,
    decoder: TutteRectangleDecoder,
    latent_shape: tuple[int, ...],
    target: DirectedTarget,
    steps: int,
    learning_rate: float,
    dtype: torch.dtype,
    device: torch.device,
    moving: torch.Tensor | None,
    fixed: torch.Tensor | None,
    learn_modulus: bool,
    target_setup_solves: int,
    target_setup_seconds: float,
    objective_threshold: float | None,
) -> InstanceOptimizationResult:
    latent, boundary_logits, raw_modulus, parameters = _optimization_variables(
        decoder, latent_shape, dtype=dtype, device=device, learn_modulus=learn_modulus
    )
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=learning_rate,
        max_iter=5,
        history_size=10,
        line_search_fn="strong_wolfe",
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )
    target_dense = target.dense.detach().to(device=device, dtype=dtype)
    target_control = target.control.detach().cpu().numpy().copy()
    trace: list[InstanceTraceRow] = []
    start = perf_counter()
    primal_solves = 0
    adjoint_solves = 0
    current_outer_step = 0

    def closure() -> torch.Tensor:
        nonlocal primal_solves, adjoint_solves
        optimizer.zero_grad(set_to_none=True)
        _synchronize(device)
        forward_start = perf_counter()
        decoded = decoder(latent, boundary_logits, raw_modulus)
        primal_solves += 1
        loss = _instance_loss(
            task,
            decoded.dense,
            target_dense,
            moving=moving,
            fixed=fixed,
            dtype=dtype,
        )
        _synchronize(device)
        forward_seconds = perf_counter() - forward_start

        audit_start = perf_counter()
        mapped = decoded.control.detach().to(device="cpu", dtype=torch.float64).numpy()
        metrics = compute_p1_map_metrics(mesh, mapped, target=target_control)
        audit_seconds = perf_counter() - audit_start
        observation_wall_seconds = perf_counter() - start
        observed_primal = primal_solves
        observed_adjoint = adjoint_solves

        _synchronize(device)
        backward_start = perf_counter()
        loss.backward()
        _synchronize(device)
        backward_seconds = perf_counter() - backward_start
        adjoint_solves += 1
        trace.append(
            InstanceTraceRow(
                iteration=len(trace),
                outer_step=current_outer_step,
                objective=float(loss.detach().cpu()),
                map_rmse=metrics.map_rmse,
                forward_seconds=forward_seconds,
                backward_seconds=backward_seconds,
                audit_seconds=audit_seconds,
                optimizer_step_seconds=0.0,
                observation_wall_seconds=observation_wall_seconds,
                primal_solves=observed_primal,
                adjoint_solves=observed_adjoint,
                global_solves=observed_primal + observed_adjoint,
                flip_count=metrics.flip_count,
                minimum_area_ratio=metrics.minimum_area_ratio,
                topology_certified=metrics.global_injectivity_certificate,
            )
        )
        return loss

    for outer_step in range(steps):
        current_outer_step = outer_step
        before = len(trace)
        _synchronize(device)
        step_start = perf_counter()
        optimizer.step(closure)
        _synchronize(device)
        elapsed = perf_counter() - step_start
        if len(trace) == before:
            raise RuntimeError("LBFGS optimizer did not evaluate its closure")
        closure_seconds = sum(
            row.forward_seconds + row.backward_seconds + row.audit_seconds
            for row in trace[before:]
        )
        trace[-1] = replace(
            trace[-1], optimizer_step_seconds=max(0.0, elapsed - closure_seconds)
        )

    optimizer.zero_grad(set_to_none=True)
    _synchronize(device)
    forward_start = perf_counter()
    decoded = decoder(latent, boundary_logits, raw_modulus)
    primal_solves += 1
    final_loss = _instance_loss(
        task,
        decoded.dense,
        target_dense,
        moving=moving,
        fixed=fixed,
        dtype=dtype,
    )
    _synchronize(device)
    forward_seconds = perf_counter() - forward_start
    audit_start = perf_counter()
    metrics = compute_p1_map_metrics(
        mesh,
        decoded.control.detach().to(device="cpu", dtype=torch.float64).numpy(),
        target=target_control,
    )
    audit_seconds = perf_counter() - audit_start
    trace.append(
        InstanceTraceRow(
            iteration=len(trace),
            outer_step=steps,
            objective=float(final_loss.detach().cpu()),
            map_rmse=metrics.map_rmse,
            forward_seconds=forward_seconds,
            backward_seconds=0.0,
            audit_seconds=audit_seconds,
            optimizer_step_seconds=0.0,
            observation_wall_seconds=perf_counter() - start,
            primal_solves=primal_solves,
            adjoint_solves=adjoint_solves,
            global_solves=primal_solves + adjoint_solves,
            flip_count=metrics.flip_count,
            minimum_area_ratio=metrics.minimum_area_ratio,
            topology_certified=metrics.global_injectivity_certificate,
        )
    )
    return _make_result(
        task=task,
        decoder=decoder,
        mesh=mesh,
        optimizer_name="lbfgs",
        trace=trace,
        primal_solves=primal_solves,
        adjoint_solves=adjoint_solves,
        target_setup_solves=target_setup_solves,
        target_setup_seconds=target_setup_seconds,
        objective_threshold=objective_threshold,
        wall_seconds=perf_counter() - start,
        final_metrics=metrics,
    )


def _validate_prebuilt_target(
    target: DirectedTarget,
    mesh: TriMesh,
    *,
    image_resolution: int,
) -> None:
    if target.control.shape != (mesh.n_vertices, 2):
        raise ValueError("prebuilt target control shape does not match the control mesh")
    if target.dense.shape != (image_resolution, image_resolution, 2):
        raise ValueError("prebuilt target dense shape does not match image_resolution")
    if target.control.dtype not in (torch.float32, torch.float64) or target.dense.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("prebuilt target tensors must be float32 or float64")
    if not bool(torch.isfinite(target.control).all()) or not bool(
        torch.isfinite(target.dense).all()
    ):
        raise ValueError("prebuilt target tensors must be finite")
    dense64 = target.dense.detach().to(dtype=torch.float64, device="cpu")
    tolerance = 64.0 * torch.finfo(torch.float64).eps
    if bool(torch.any(dense64 < -tolerance)) or bool(torch.any(dense64 > 1.0 + tolerance)):
        raise ValueError("prebuilt target dense map lies outside the unit image domain")
    control64 = target.control.detach().to(dtype=torch.float64, device="cpu")
    reconstructed = StructuredDenseQueryTable.from_mesh(
        mesh, height=image_resolution, width=image_resolution
    ).interpolate(control64)
    source_dtype = torch.promote_types(target.control.dtype, target.dense.dtype)
    consistency_tolerance = 512.0 * torch.finfo(source_dtype).eps
    if not bool(
        torch.allclose(
            reconstructed,
            dense64,
            atol=consistency_tolerance,
            rtol=consistency_tolerance,
        )
    ):
        raise ValueError("prebuilt target dense map is inconsistent with P1 control interpolation")
    metrics = compute_p1_map_metrics(
        mesh,
        control64.numpy(),
    )
    if not metrics.global_injectivity_certificate:
        raise ValueError("prebuilt target failed the independent P1 topology certificate")


def _target_for_run(
    mesh: TriMesh,
    *,
    target: DirectedTarget | None,
    image_resolution: int,
    seed: int,
    target_strength: float,
    height: float,
) -> tuple[DirectedTarget, int, float]:
    if target is not None:
        _validate_prebuilt_target(target, mesh, image_resolution=image_resolution)
        return target, 0, 0.0
    setup_start = perf_counter()
    generated = build_directed_target(
        mesh,
        image_height=image_resolution,
        image_width=image_resolution,
        seed=seed,
        strength=target_strength,
        height=height,
    )
    return generated, 1, perf_counter() - setup_start


def run_supervised_instance(
    *,
    control_vertices: int,
    image_resolution: int,
    steps: int,
    backend: str,
    optimizer_name: str,
    dtype: torch.dtype,
    seed: int,
    target_strength: float,
    learning_rate: float,
    device: torch.device | str = "cpu",
    target: DirectedTarget | None = None,
    objective_threshold: float | None = None,
) -> InstanceOptimizationResult:
    if optimizer_name not in ("adam", "lbfgs"):
        raise ValueError("optimizer_name must be 'adam' or 'lbfgs'")
    if control_vertices < 3 or steps < 1:
        raise ValueError("control_vertices must be >=3 and steps must be positive")
    device = torch.device(device)
    mesh = structured_rectangle(control_vertices - 1, control_vertices - 1)
    if objective_threshold is not None and (
        not np.isfinite(objective_threshold) or objective_threshold < 0.0
    ):
        raise ValueError("objective_threshold must be finite and nonnegative")
    target, target_setup_solves, target_setup_seconds = _target_for_run(
        mesh,
        target=target,
        image_resolution=image_resolution,
        seed=seed,
        target_strength=target_strength,
        height=0.9,
    )
    decoder, latent_shape = _training_decoder(
        mesh,
        backend=backend,
        image_resolution=image_resolution,
        dtype=dtype,
        device=device,
    )
    runner = _run_adam if optimizer_name == "adam" else _run_lbfgs
    return runner(
        task="supervised_map",
        mesh=mesh,
        decoder=decoder,
        latent_shape=latent_shape,
        target=target,
        steps=steps,
        learning_rate=learning_rate,
        dtype=dtype,
        device=device,
        moving=None,
        fixed=None,
        learn_modulus=True,
        target_setup_solves=target_setup_solves,
        target_setup_seconds=target_setup_seconds,
        objective_threshold=objective_threshold,
    )


def run_image_instance(
    *,
    control_vertices: int,
    image_resolution: int,
    steps: int,
    backend: str,
    optimizer_name: str,
    dtype: torch.dtype,
    seed: int,
    target_strength: float,
    learning_rate: float,
    image_name: str,
    device: torch.device | str = "cpu",
    target: DirectedTarget | None = None,
    objective_threshold: float | None = None,
) -> InstanceOptimizationResult:
    if optimizer_name not in ("adam", "lbfgs"):
        raise ValueError("optimizer_name must be 'adam' or 'lbfgs'")
    if control_vertices < 3 or steps < 1:
        raise ValueError("control_vertices must be >=3 and steps must be positive")
    device = torch.device(device)
    mesh = structured_rectangle(control_vertices - 1, control_vertices - 1)
    if objective_threshold is not None and (
        not np.isfinite(objective_threshold) or objective_threshold < 0.0
    ):
        raise ValueError("objective_threshold must be finite and nonnegative")
    target, target_setup_solves, target_setup_seconds = _target_for_run(
        mesh,
        target=target,
        image_resolution=image_resolution,
        seed=seed,
        target_strength=target_strength,
        height=1.0,
    )
    decoder, latent_shape = _training_decoder(
        mesh,
        backend=backend,
        image_resolution=image_resolution,
        dtype=dtype,
        device=device,
    )
    moving = synthetic_image(
        image_name,
        height=image_resolution,
        width=image_resolution,
        dtype=dtype,
        device=device,
    )
    target_dense = target.dense.detach().to(device=device, dtype=dtype)
    fixed = warp_image_backward(moving, target_dense)
    runner = _run_adam if optimizer_name == "adam" else _run_lbfgs
    return runner(
        task="image_registration",
        mesh=mesh,
        decoder=decoder,
        latent_shape=latent_shape,
        target=target,
        steps=steps,
        learning_rate=learning_rate,
        dtype=dtype,
        device=device,
        moving=moving,
        fixed=fixed,
        learn_modulus=False,
        target_setup_solves=target_setup_solves,
        target_setup_seconds=target_setup_seconds,
        objective_threshold=objective_threshold,
    )
