"""Differentiable small/medium-mesh Tutte boundary layer."""

from __future__ import annotations

import torch

from ..mesh import TriMesh
from .tutte import tutte_weights


def tutte_embedding_torch(mesh: TriMesh, target_boundary: torch.Tensor) -> torch.Tensor:
    """Apply precomputed dense Tutte weights to boundary coordinates.

    This explicit-weight implementation is intended for gradient checks and
    moderate meshes. High-resolution use should replace the dense matrix by an
    implicit sparse solve/JVP without changing the mathematical map.
    """

    if target_boundary.ndim != 2 or target_boundary.shape[1] != 2:
        raise ValueError("target_boundary must have shape (boundary_vertices, 2)")
    if not torch.is_floating_point(target_boundary):
        raise ValueError("target_boundary must be floating point")
    weights = torch.as_tensor(
        tutte_weights(mesh), dtype=target_boundary.dtype, device=target_boundary.device
    )
    if target_boundary.shape[0] != weights.shape[1]:
        raise ValueError("target_boundary has the wrong number of vertices")
    return weights @ target_boundary
