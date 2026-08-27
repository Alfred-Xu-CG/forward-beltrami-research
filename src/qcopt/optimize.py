"""Controlled μ-space and direct-map optimization loops."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from .energies import (
    amips_style_energy,
    lim_style_energy,
    symmetric_dirichlet_energy,
    torch_face_jacobians,
)
from .injectivity import audit_injectivity
from .mesh import TriMesh


@dataclass
class OptimizationResult:
    uv: np.ndarray
    history: list[dict[str, float | int | bool]]
    elapsed_seconds: float
    raw_mu: np.ndarray | None = None


def certified_interpolate_raw(
    old: np.ndarray,
    proposed: np.ndarray,
    map_from_raw: Callable[[np.ndarray], np.ndarray],
    mesh: TriMesh,
    *,
    rectangle: bool,
    max_backtracks: int = 20,
) -> tuple[np.ndarray, np.ndarray, int]:
    old = np.asarray(old, dtype=np.float64)
    proposed = np.asarray(proposed, dtype=np.float64)
    if old.shape != proposed.shape:
        raise ValueError("old and proposed raw parameters must have equal shape")
    for backtracks in range(max_backtracks + 1):
        step = 0.5**backtracks
        candidate = old + step * (proposed - old)
        uv = np.asarray(map_from_raw(candidate), dtype=np.float64)
        if audit_injectivity(mesh, uv, rectangle=rectangle).certified:
            return candidate, uv, backtracks
    old_uv = np.asarray(map_from_raw(old), dtype=np.float64)
    if not audit_injectivity(mesh, old_uv, rectangle=rectangle).certified:
        raise RuntimeError("neither proposal nor previous raw parameters are certified")
    return old.copy(), old_uv, max_backtracks + 1


def run_mu_optimization(
    mesh: TriMesh,
    raw_initial: torch.Tensor,
    map_from_raw: Callable[[torch.Tensor], torch.Tensor],
    loss_function: Callable[[torch.Tensor], torch.Tensor],
    *,
    iterations: int,
    learning_rate: float,
    rectangle: bool,
    max_backtracks: int = 20,
) -> OptimizationResult:
    raw = raw_initial.detach().clone().to(dtype=torch.double).requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    with torch.no_grad():
        initial_uv = map_from_raw(raw).detach().cpu().numpy()
    if not audit_injectivity(mesh, initial_uv, rectangle=rectangle).certified:
        raise ValueError("initial μ reconstruction must be certified")
    history: list[dict[str, float | int | bool]] = []
    start = perf_counter()
    accepted_uv = initial_uv
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        uv = map_from_raw(raw)
        loss = loss_function(uv)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("μ-space loss became non-finite")
        loss.backward()
        old = raw.detach().clone()
        optimizer.step()
        proposed = raw.detach().clone()
        accepted = False
        used_backtracks = max_backtracks + 1
        for backtracks in range(max_backtracks + 1):
            alpha = 0.5**backtracks
            candidate = old + alpha * (proposed - old)
            with torch.no_grad():
                raw.copy_(candidate)
                candidate_uv = map_from_raw(raw).detach().cpu().numpy()
            report = audit_injectivity(mesh, candidate_uv, rectangle=rectangle)
            if report.certified:
                accepted = True
                used_backtracks = backtracks
                accepted_uv = candidate_uv
                break
        if not accepted:
            with torch.no_grad():
                raw.copy_(old)
                accepted_uv = map_from_raw(raw).detach().cpu().numpy()
            report = audit_injectivity(mesh, accepted_uv, rectangle=rectangle)
        history.append(
            {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "accepted": accepted,
                "backtracks": used_backtracks,
                "certified": report.certified,
                "minimum_signed_area_ratio": report.minimum_signed_area_ratio,
            }
        )
    return OptimizationResult(
        uv=accepted_uv,
        history=history,
        elapsed_seconds=perf_counter() - start,
        raw_mu=raw.detach().cpu().numpy().copy(),
    )


def run_direct_optimization(
    mesh: TriMesh,
    initial_uv: np.ndarray,
    loss_function: Callable[[torch.Tensor], torch.Tensor],
    *,
    method: str,
    iterations: int = 100,
    learning_rate: float = 0.05,
    barrier_weight: float = 0.01,
    rectangle: bool = False,
    max_backtracks: int = 20,
) -> OptimizationResult:
    energy_functions = {
        "lim_style": lim_style_energy,
        "slim_style": symmetric_dirichlet_energy,
        "amips_style": amips_style_energy,
    }
    if method != "none" and method not in energy_functions:
        raise ValueError(f"unknown method: {method}")
    writable_initial = np.array(initial_uv, dtype=np.float64, copy=True)
    uv = torch.as_tensor(writable_initial, dtype=torch.double).clone().requires_grad_(True)
    if rectangle:
        _project_rectangle_sliding(mesh, uv)
    optimizer = torch.optim.Adam([uv], lr=learning_rate)
    history: list[dict[str, float | int | bool]] = []
    start = perf_counter()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        data_loss = loss_function(uv)
        objective = data_loss
        if method != "none":
            jacobians = torch_face_jacobians(mesh, uv)
            objective = objective + barrier_weight * energy_functions[method](jacobians)
        if not bool(torch.isfinite(objective).item()):
            raise FloatingPointError("direct-map objective became non-finite")
        objective.backward()
        old = uv.detach().clone()
        optimizer.step()
        if rectangle:
            _project_rectangle_sliding(mesh, uv)
        proposal = uv.detach().clone()
        used_backtracks = 0
        accepted = True
        if method != "none":
            accepted = False
            used_backtracks = max_backtracks + 1
            for backtracks in range(max_backtracks + 1):
                alpha = 0.5**backtracks
                with torch.no_grad():
                    uv.copy_(old + alpha * (proposal - old))
                    if rectangle:
                        _project_rectangle_sliding(mesh, uv)
                candidate = uv.detach().cpu().numpy()
                if audit_injectivity(mesh, candidate, rectangle=rectangle).certified:
                    accepted = True
                    used_backtracks = backtracks
                    break
            if not accepted:
                with torch.no_grad():
                    uv.copy_(old)
        report = audit_injectivity(
            mesh, uv.detach().cpu().numpy(), rectangle=rectangle
        )
        history.append(
            {
                "iteration": iteration,
                "loss": float(data_loss.detach()),
                "objective": float(objective.detach()),
                "accepted": accepted,
                "backtracks": used_backtracks,
                "certified": report.certified,
                "minimum_signed_area_ratio": report.minimum_signed_area_ratio,
            }
        )
    return OptimizationResult(
        uv=uv.detach().cpu().numpy().copy(),
        history=history,
        elapsed_seconds=perf_counter() - start,
    )


def _project_rectangle_sliding(mesh: TriMesh, uv: torch.Tensor) -> None:
    source = mesh.vertices
    x_min, y_min = source.min(axis=0)
    x_max, y_max = source.max(axis=0)
    tolerance = 1e-12 * max(x_max - x_min, y_max - y_min, 1.0)
    with torch.no_grad():
        for index, (x, y) in enumerate(source):
            if abs(x - x_min) <= tolerance:
                uv[index, 0] = float(x_min)
            elif abs(x - x_max) <= tolerance:
                uv[index, 0] = float(x_max)
            if abs(y - y_min) <= tolerance:
                uv[index, 1] = float(y_min)
            elif abs(y - y_max) <= tolerance:
                uv[index, 1] = float(y_max)
