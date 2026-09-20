"""Finite-element-safe explicit PL flow prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..injectivity import InjectivityReport, audit_injectivity
from ..mesh import TriMesh
from .safe_step import certified_residual_step


@dataclass(frozen=True)
class SafePLFlowResult:
    """Maps, active scalar steps, and independent topology reports."""

    maps: tuple[np.ndarray, ...]
    steps: tuple[float, ...]
    reports: tuple[InjectivityReport, ...]


def safe_pl_flow(
    mesh: TriMesh,
    initial: np.ndarray,
    velocities: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    min_det_margin: float = 1e-10,
    safety: float = 0.99,
) -> SafePLFlowResult:
    """Compose P1 Euler residuals while requiring every step to audit certified."""

    current = _validate(mesh, initial, "initial")
    maps = [current.copy()]
    steps: list[float] = []
    reports: list[InjectivityReport] = []
    for velocity in velocities:
        direction = _validate(mesh, velocity, "velocity")
        updated, step = certified_residual_step(
            mesh,
            current,
            direction,
            min_det_margin=min_det_margin,
            safety=safety,
        )
        report = audit_injectivity(mesh, updated)
        if not report.certified:
            raise RuntimeError("safe determinant step failed the global injectivity audit")
        current = updated
        maps.append(current.copy())
        steps.append(step)
        reports.append(report)
    return SafePLFlowResult(tuple(maps), tuple(steps), tuple(reports))


def _validate(mesh: TriMesh, values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (mesh.n_vertices, 2) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite with shape (n_vertices, 2)")
    return values
