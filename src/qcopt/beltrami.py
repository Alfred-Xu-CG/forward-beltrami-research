"""Piecewise-linear Jacobian and Beltrami calculations."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from .mesh import TriMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def face_jacobians(mesh: TriMesh, uv: FloatArray) -> FloatArray:
    """Return map Jacobians ``[[u_x,u_y],[v_x,v_y]]`` on every face."""

    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape != (mesh.n_vertices, 2):
        raise ValueError("uv must have shape (mesh.n_vertices, 2)")
    local = uv[mesh.faces]
    coordinate_gradients = np.einsum("fki,fkj->fij", local, mesh.gradients)
    return np.ascontiguousarray(coordinate_gradients)


def face_beltrami(mesh: TriMesh, uv: FloatArray) -> ComplexArray:
    """Compute the exact facewise Beltrami coefficient of a PL map."""

    jac = face_jacobians(mesh, uv)
    ux, uy = jac[:, 0, 0], jac[:, 0, 1]
    vx, vy = jac[:, 1, 0], jac[:, 1, 1]
    fz = 0.5 * ((ux + vy) + 1j * (vx - uy))
    fbar = 0.5 * ((ux - vy) + 1j * (vx + uy))
    threshold = 64.0 * np.finfo(np.float64).eps
    if np.any(np.abs(fz) <= threshold):
        raise ValueError("map has a face with vanishing complex derivative f_z")
    return np.asarray(fbar / fz, dtype=np.complex128)


def qc_dilation(mu: ComplexArray) -> FloatArray:
    """Return ``K=(1+|mu|)/(1-|mu|)`` and infinity outside the QC disk."""

    magnitude = np.abs(np.asarray(mu, dtype=np.complex128))
    with np.errstate(divide="ignore", invalid="ignore"):
        dilation = (1.0 + magnitude) / (1.0 - magnitude)
    return np.where(magnitude < 1.0, dilation, np.inf)


def radial_squash(raw: torch.Tensor, k_max: float = 0.95) -> torch.Tensor:
    """Smoothly map real two-vectors into ``|mu| < k_max``."""

    if raw.shape[-1] != 2:
        raise ValueError("raw must have final dimension 2")
    if not 0.0 < k_max < 1.0:
        raise ValueError("k_max must lie strictly between zero and one")
    radius = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    small = radius <= torch.finfo(raw.dtype).eps ** 0.25
    safe_radius = torch.where(small, torch.ones_like(radius), radius)
    regular_ratio = torch.tanh(radius) / safe_radius
    series_ratio = 1.0 - radius.square() / 3.0 + 2.0 * radius.pow(4) / 15.0
    ratio = torch.where(small, series_ratio, regular_ratio)
    return k_max * ratio * raw
