"""Sparse implicit Tutte layer with adjoint VJP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse.linalg import factorized

from ..mesh import TriMesh
from .tutte import _interior_system


@dataclass
class TutteImplicitSystem:
    """Sparse interior matrix and boundary coupling for one fixed mesh."""

    loop: np.ndarray
    interior: np.ndarray
    n_vertices: int
    matrix: object
    boundary_coupling: object

    def __post_init__(self) -> None:
        self._solve = factorized(self.matrix.tocsc())
        self._solve_transpose = factorized(self.matrix.T.tocsc())

    @classmethod
    def from_mesh(cls, mesh: TriMesh) -> "TutteImplicitSystem":
        if len(mesh.boundary_loops) != 1:
            raise ValueError("Tutte embedding requires exactly one boundary loop")
        loop = np.asarray(mesh.boundary_loops[0], dtype=np.int64)
        interior, matrix, coupling = _interior_system(mesh, loop)
        return cls(
            loop,
            np.asarray(interior, dtype=np.int64),
            mesh.n_vertices,
            matrix,
            coupling,
        )

    def solve(self, target_boundary: np.ndarray) -> np.ndarray:
        rhs = self.boundary_coupling @ target_boundary
        return np.asarray(self._solve(rhs), dtype=np.float64)

    def solve_transpose(self, interior_gradient: np.ndarray) -> np.ndarray:
        return np.asarray(self._solve_transpose(interior_gradient), dtype=np.float64)


class _TutteImplicitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target_boundary: torch.Tensor, system: TutteImplicitSystem):
        if target_boundary.ndim != 2 or target_boundary.shape[1] != 2:
            raise ValueError("target_boundary must have shape (boundary_vertices, 2)")
        if not torch.is_floating_point(target_boundary):
            raise ValueError("target_boundary must be floating point")
        target = target_boundary.detach().cpu().numpy()
        if target.shape[0] != len(system.loop):
            raise ValueError("target_boundary has the wrong number of vertices")
        output = np.zeros((len(system.loop) + len(system.interior), 2), dtype=np.float64)
        # The mesh vertex order is not generally boundary-then-interior, so the
        # caller supplies the full vertex count through the system's index map.
        output = np.empty((system.n_vertices, 2), dtype=np.float64)
        output[system.loop] = target
        output[system.interior] = system.solve(target)
        ctx.system = system
        ctx.device = target_boundary.device
        ctx.dtype = target_boundary.dtype
        return torch.as_tensor(output, dtype=target_boundary.dtype, device=target_boundary.device)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        system = ctx.system
        grad = gradient.detach().cpu().numpy()
        boundary_gradient = grad[system.loop].copy()
        adjoint = system.solve_transpose(grad[system.interior])
        boundary_gradient += np.asarray(system.boundary_coupling.T @ adjoint)
        return torch.as_tensor(
            boundary_gradient, dtype=ctx.dtype, device=ctx.device
        ), None


def tutte_embedding_torch_implicit(
    mesh: TriMesh,
    target_boundary: torch.Tensor,
    system: TutteImplicitSystem | None = None,
) -> torch.Tensor:
    """Apply a sparse Tutte solve with an implicit transpose-solve backward pass."""

    if system is None:
        system = TutteImplicitSystem.from_mesh(mesh)
    if system.n_vertices != mesh.n_vertices:
        raise ValueError("system was built for a different mesh")
    return _TutteImplicitFunction.apply(target_boundary, system)
