"""PyTorch custom autograd wrappers around exact sparse QC solves."""

from __future__ import annotations

import numpy as np
import torch

from .adjoint import lbs_mu_vjp, lsqc_fast_mu_vjp, lsqc_mu_vjp
from .beltrami import radial_squash
from .constraints import LinearConstraints
from .lbs import solve_lbs
from .lsqc import solve_lsqc, solve_lsqc_fast
from .mesh import TriMesh


def _tensor_to_mu(mu_components: torch.Tensor, n_faces: int) -> np.ndarray:
    if mu_components.device.type != "cpu":
        raise ValueError("the SciPy reference layer supports CPU tensors only")
    if mu_components.shape != (n_faces, 2):
        raise ValueError("mu_components must have shape (mesh.n_faces, 2)")
    values = mu_components.detach().numpy().astype(np.float64, copy=False)
    return values[:, 0] + 1j * values[:, 1]


class _LBSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, mu_components: torch.Tensor, mesh: TriMesh, constraints: LinearConstraints
    ) -> torch.Tensor:
        mu = _tensor_to_mu(mu_components, mesh.n_faces)
        result = solve_lbs(mesh, mu, constraints)
        ctx.mesh = mesh
        ctx.mu = mu
        ctx.result = result
        return torch.as_tensor(result.uv.copy(), dtype=mu_components.dtype)

    @staticmethod
    def backward(ctx, grad_uv: torch.Tensor):
        gradient = lbs_mu_vjp(
            ctx.mesh,
            ctx.mu,
            ctx.result,
            grad_uv.detach().numpy().astype(np.float64, copy=False),
        ).gradient
        return torch.as_tensor(gradient, dtype=grad_uv.dtype), None, None


class _LSQCFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mu_components: torch.Tensor,
        mesh: TriMesh,
        constraints: LinearConstraints,
        weighted: bool,
    ) -> torch.Tensor:
        mu = _tensor_to_mu(mu_components, mesh.n_faces)
        result = solve_lsqc(mesh, mu, constraints, weighted=bool(weighted))
        ctx.mesh = mesh
        ctx.mu = mu
        ctx.result = result
        ctx.weighted = bool(weighted)
        return torch.as_tensor(result.uv.copy(), dtype=mu_components.dtype)

    @staticmethod
    def backward(ctx, grad_uv: torch.Tensor):
        gradient = lsqc_mu_vjp(
            ctx.mesh,
            ctx.mu,
            ctx.result,
            grad_uv.detach().numpy().astype(np.float64, copy=False),
            weighted=ctx.weighted,
        ).gradient
        return torch.as_tensor(gradient, dtype=grad_uv.dtype), None, None, None


class _FastLSQCFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mu_components: torch.Tensor,
        mesh: TriMesh,
        constraints: LinearConstraints,
    ) -> torch.Tensor:
        mu = _tensor_to_mu(mu_components, mesh.n_faces)
        result = solve_lsqc_fast(mesh, mu, constraints)
        ctx.mesh = mesh
        ctx.mu = mu
        ctx.result = result
        return torch.as_tensor(result.uv.copy(), dtype=mu_components.dtype)

    @staticmethod
    def backward(ctx, grad_uv: torch.Tensor):
        gradient = lsqc_fast_mu_vjp(
            ctx.mesh,
            ctx.mu,
            ctx.result,
            grad_uv.detach().numpy().astype(np.float64, copy=False),
        ).gradient
        return torch.as_tensor(gradient, dtype=grad_uv.dtype), None, None


def lbs_layer(
    mu_components: torch.Tensor, mesh: TriMesh, constraints: LinearConstraints
) -> torch.Tensor:
    """Reconstruct a constrained LBS map from bounded facewise μ components."""

    return _LBSFunction.apply(mu_components, mesh, constraints)


def lsqc_layer(
    mu_components: torch.Tensor,
    mesh: TriMesh,
    constraints: LinearConstraints,
    weighted: bool = False,
) -> torch.Tensor:
    """Reconstruct a two-pin free-boundary LSQC map from facewise μ."""

    return _LSQCFunction.apply(mu_components, mesh, constraints, weighted)


def lsqc_fast_layer(
    mu_components: torch.Tensor,
    mesh: TriMesh,
    constraints: LinearConstraints,
) -> torch.Tensor:
    """Fast weighted LSQC layer for exact coordinate-selector constraints."""

    return _FastLSQCFunction.apply(mu_components, mesh, constraints)


def lbs_from_raw(
    raw_mu: torch.Tensor,
    mesh: TriMesh,
    constraints: LinearConstraints,
    k_max: float = 0.95,
) -> torch.Tensor:
    return lbs_layer(radial_squash(raw_mu, k_max), mesh, constraints)


def lsqc_from_raw(
    raw_mu: torch.Tensor,
    mesh: TriMesh,
    constraints: LinearConstraints,
    k_max: float = 0.95,
    weighted: bool = False,
) -> torch.Tensor:
    return lsqc_layer(
        radial_squash(raw_mu, k_max), mesh, constraints, weighted=weighted
    )


def lsqc_fast_from_raw(
    raw_mu: torch.Tensor,
    mesh: TriMesh,
    constraints: LinearConstraints,
    k_max: float = 0.95,
) -> torch.Tensor:
    """Radially bound facewise mu, then apply the fast weighted LSQC layer."""

    return lsqc_fast_layer(radial_squash(raw_mu, k_max), mesh, constraints)
