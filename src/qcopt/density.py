"""Prescribed face-area targets and differentiable density losses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .energies import torch_face_jacobians
from .mesh import TriMesh


@dataclass(frozen=True)
class AreaTarget:
    name: str
    factors: np.ndarray
    feasibility_status: str
    known_map: np.ndarray | None = None


def normalize_area_factors(mesh: TriMesh, factors: np.ndarray) -> np.ndarray:
    values = np.asarray(factors, dtype=np.float64).reshape(-1)
    if values.shape != (mesh.n_faces,) or np.any(values <= 0.0):
        raise ValueError("area factors must be positive with one value per face")
    current = float(np.sum(mesh.areas * values))
    total = float(np.sum(mesh.areas))
    return values * (total / current)


def manufactured_area_target(mesh: TriMesh, amplitude: float = 0.10) -> AreaTarget:
    if not 0.0 <= amplitude <= 0.15:
        raise ValueError("manufactured amplitude must lie in [0, 0.15]")
    x, y = mesh.vertices[:, 0], mesh.vertices[:, 1]
    uv = np.column_stack(
        (
            x + amplitude * np.sin(np.pi * x) * np.sin(2.0 * np.pi * y),
            y + amplitude * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y),
        )
    )
    jacobians = np.einsum("fki,fkj->fij", uv[mesh.faces], mesh.gradients)
    factors = np.linalg.det(jacobians)
    if np.any(factors <= 0.0):
        raise ValueError("amplitude produced a non-bijective manufactured map")
    return AreaTarget(
        name="manufactured",
        factors=factors,
        feasibility_status="manufactured_feasible",
        known_map=uv,
    )


def stress_area_target(
    mesh: TriMesh, kind: str, *, seed: int = 20260828
) -> AreaTarget:
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    x, y = centroids[:, 0], centroids[:, 1]
    if kind == "checkerboard":
        parity = (np.floor(6.0 * x).astype(int) + np.floor(6.0 * y).astype(int)) % 2
        raw = np.exp(np.where(parity == 0, 1.4, -1.4))
    elif kind == "spike":
        radius_squared = (x - 0.68) ** 2 + (y - 0.34) ** 2
        raw = np.exp(3.2 * np.exp(-radius_squared / 0.012))
    elif kind == "random":
        rng = np.random.default_rng(seed)
        field = np.zeros(mesh.n_faces)
        for frequency in range(1, 5):
            phase_x, phase_y = rng.uniform(0.0, 2.0 * np.pi, size=2)
            coefficient = rng.normal(scale=0.5 / frequency)
            field += coefficient * np.sin(
                2.0 * np.pi * frequency * x + phase_x
            ) * np.cos(2.0 * np.pi * frequency * y + phase_y)
        raw = np.exp(1.8 * field / max(np.std(field), 1e-12))
    else:
        raise ValueError(f"unknown stress target: {kind}")
    return AreaTarget(
        name=kind,
        factors=normalize_area_factors(mesh, raw),
        feasibility_status="sum_normalized_feasibility_unproven",
    )


def torch_area_ratios(mesh: TriMesh, uv: torch.Tensor) -> torch.Tensor:
    return torch.linalg.det(torch_face_jacobians(mesh, uv))


def area_loss(
    ratios: torch.Tensor,
    targets: torch.Tensor,
    determinant_floor: float = 1e-8,
    inversion_penalty: float = 100.0,
) -> torch.Tensor:
    if ratios.shape != targets.shape:
        raise ValueError("ratios and targets must have matching shapes")
    safe_ratios = torch.clamp(ratios, min=determinant_floor)
    log_mismatch = (torch.log(safe_ratios) - torch.log(targets)).square().mean()
    inverted = torch.relu(determinant_floor - ratios).square().mean()
    return log_mismatch + inversion_penalty * inverted
