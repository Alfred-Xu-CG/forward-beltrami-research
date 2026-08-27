"""Differentiable direct-map distortion and orientation barriers."""

from __future__ import annotations

import torch

from .mesh import TriMesh


def torch_face_jacobians(mesh: TriMesh, uv: torch.Tensor) -> torch.Tensor:
    if uv.shape != (mesh.n_vertices, 2):
        raise ValueError("uv must have shape (mesh.n_vertices, 2)")
    faces = torch.as_tensor(mesh.faces.copy(), dtype=torch.long, device=uv.device)
    gradients = torch.as_tensor(
        mesh.gradients.copy(), dtype=uv.dtype, device=uv.device
    )
    return torch.einsum("fki,fkj->fij", uv[faces], gradients)


def _positive_determinants(
    jacobians: torch.Tensor, determinant_floor: float
) -> torch.Tensor | None:
    determinants = torch.linalg.det(jacobians)
    if bool(torch.any(determinants <= determinant_floor).item()):
        return None
    return determinants


def _infinity(jacobians: torch.Tensor) -> torch.Tensor:
    return torch.full((), torch.inf, dtype=jacobians.dtype, device=jacobians.device)


def log_det_barrier(
    jacobians: torch.Tensor, determinant_floor: float = 1e-12
) -> torch.Tensor:
    determinants = _positive_determinants(jacobians, determinant_floor)
    if determinants is None:
        return _infinity(jacobians)
    return -torch.log(determinants).mean()


def lim_style_energy(
    jacobians: torch.Tensor,
    barrier_weight: float = 1.0,
    determinant_floor: float = 1e-12,
) -> torch.Tensor:
    """Identity-centered elastic energy with a log-determinant barrier."""

    determinants = _positive_determinants(jacobians, determinant_floor)
    if determinants is None:
        return _infinity(jacobians)
    identity = torch.eye(2, dtype=jacobians.dtype, device=jacobians.device)
    elastic = (jacobians - identity).square().sum(dim=(-2, -1))
    return (elastic - barrier_weight * torch.log(determinants)).mean()


def symmetric_dirichlet_energy(
    jacobians: torch.Tensor, determinant_floor: float = 1e-12
) -> torch.Tensor:
    determinants = _positive_determinants(jacobians, determinant_floor)
    if determinants is None:
        return _infinity(jacobians)
    inverse = torch.linalg.inv(jacobians)
    return (
        jacobians.square().sum(dim=(-2, -1))
        + inverse.square().sum(dim=(-2, -1))
    ).mean()


def amips_style_energy(
    jacobians: torch.Tensor, determinant_floor: float = 1e-12
) -> torch.Tensor:
    """Scale-invariant 2D MIPS distortion, normalized to one at similarities."""

    determinants = _positive_determinants(jacobians, determinant_floor)
    if determinants is None:
        return _infinity(jacobians)
    frobenius_squared = jacobians.square().sum(dim=(-2, -1))
    return (0.5 * frobenius_squared / determinants).mean()
