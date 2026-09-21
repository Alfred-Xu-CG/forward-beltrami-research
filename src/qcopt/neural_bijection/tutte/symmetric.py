"""Matrix-free symmetric-conductance Tutte layer with implicit CG VJP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.autograd.function import once_differentiable

from ...forward.tutte_directed_implicit import (
    DirectedTutteSystem,
    _validate_dividing_edges,
    _validate_strictly_positive_faces,
    _validate_weakly_convex_boundary,
)
from ...mesh import TriMesh


@dataclass(frozen=True)
class KrylovStats:
    iterations: int
    relative_residual: float
    converged: bool


def _conjugate_gradient(
    layer: "MatrixFreeSymmetricTutteLayer",
    conductances: torch.Tensor,
    rhs: torch.Tensor,
) -> tuple[torch.Tensor, KrylovStats]:
    """Vectorized CG over batch and trailing right-hand-side columns."""
    if rhs.shape[1] == 0:
        return torch.zeros_like(rhs), KrylovStats(0, 0.0, True)
    if not bool(torch.isfinite(rhs).all()):
        raise RuntimeError("matrix-free symmetric CG requires a finite right-hand side")
    # Scale each independent RHS by its largest represented magnitude.  This
    # prevents both squared-norm overflow and underflow, including in the
    # implicit adjoint where a user may supply a very large/small cotangent.
    rhs_scale = rhs.abs().amax(dim=1, keepdim=True)
    safe_scale = torch.where(rhs_scale > 0.0, rhs_scale, torch.ones_like(rhs_scale))
    scaled_rhs = rhs / safe_scale
    x = torch.zeros_like(scaled_rhs)
    residual = scaled_rhs.clone()
    direction = residual.clone()
    residual_norm = torch.linalg.vector_norm(residual, dim=1)
    rhs_norm = residual_norm.clone()
    squared_norm = residual.square().sum(dim=1)
    relative_tolerance = layer.relative_tolerance
    if relative_tolerance is None:
        relative_tolerance = 1.0e-5 if rhs.dtype == torch.float32 else 1.0e-11
    scaled_absolute_tolerance = layer.absolute_tolerance / safe_scale.squeeze(1)
    threshold = scaled_absolute_tolerance + relative_tolerance * rhs_norm
    active = residual_norm > threshold
    iterations = 0

    for iteration in range(layer.max_iterations):
        if not bool(active.any()):
            break
        operator_direction = layer._apply_batched(conductances, direction)
        denominator = (direction * operator_direction).sum(dim=1)
        if bool(torch.any(active & ((denominator <= 0.0) | ~torch.isfinite(denominator)))):
            raise RuntimeError("matrix-free symmetric CG encountered a nonpositive or nonfinite curvature")
        safe_denominator = torch.where(active, denominator, torch.ones_like(denominator))
        alpha = torch.where(
            active,
            squared_norm / safe_denominator,
            torch.zeros_like(squared_norm),
        )
        x = x + alpha[:, None, :] * direction
        recursive_residual = residual - alpha[:, None, :] * operator_direction
        recursive_norm = torch.linalg.vector_norm(recursive_residual, dim=1)
        # Preserve CG conjugacy between reliable updates.  Whenever the
        # recursive residual appears converged, and periodically every 32
        # iterations, replace it by b-Ax.  A false recursive convergence then
        # restarts that RHS from its represented true residual instead of
        # returning or repeatedly destroying conjugacy at every iteration.
        reliable_update = active & (
            (recursive_norm <= threshold) | (((iteration + 1) % 32) == 0)
        )
        if bool(reliable_update.any()):
            true_residual = scaled_rhs - layer._apply_batched(conductances, x)
            residual = torch.where(
                reliable_update[:, None, :], true_residual, recursive_residual
            )
        else:
            residual = recursive_residual
        residual_norm = torch.linalg.vector_norm(residual, dim=1)
        if not bool(torch.isfinite(residual_norm).all()):
            raise RuntimeError("matrix-free symmetric CG produced a nonfinite residual")
        new_active = residual_norm > threshold
        new_squared_norm = residual.square().sum(dim=1)
        safe_previous = torch.where(
            active,
            squared_norm,
            torch.ones_like(squared_norm),
        )
        restarted = reliable_update & new_active
        beta = torch.where(
            new_active & ~restarted,
            new_squared_norm / safe_previous,
            torch.zeros_like(new_squared_norm),
        )
        direction = residual + beta[:, None, :] * direction
        direction = torch.where(new_active[:, None, :], direction, torch.zeros_like(direction))
        squared_norm = new_squared_norm
        active = new_active
        iterations = iteration + 1

    true_residual = layer._apply_batched(conductances, x) - scaled_rhs
    true_norm = torch.linalg.vector_norm(true_residual, dim=1)
    converged_mask = true_norm <= threshold
    denominator = torch.where(rhs_norm > 0.0, rhs_norm, torch.ones_like(rhs_norm))
    relative = torch.where(rhs_norm > 0.0, true_norm / denominator, true_norm)
    maximum_relative = float(relative.max().detach().cpu()) if relative.numel() else 0.0
    converged = bool(converged_mask.all())
    stats = KrylovStats(iterations, maximum_relative, converged)
    if not converged:
        raise RuntimeError(
            "matrix-free symmetric CG did not converge: "
            f"iterations={iterations}, max_relative_residual={maximum_relative:.3e}"
        )
    result = x * safe_scale
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("matrix-free symmetric CG solution left the finite dtype range")
    return result, stats


class _SymmetricSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, conductances, boundary, layer):
        rhs = layer._boundary_rhs(conductances, boundary)
        solution, stats = _conjugate_gradient(layer, conductances, rhs)
        output = torch.empty(
            (conductances.shape[0], layer.system.n_vertices, 2),
            dtype=boundary.dtype,
            device=boundary.device,
        )
        output[:, layer._loop] = boundary
        output[:, layer._interior] = solution
        mapped_cpu = output.detach().to(device="cpu", dtype=torch.float64).numpy()
        for mapped in mapped_cpu:
            _validate_strictly_positive_faces(layer.system.faces, mapped)
        ctx.layer = layer
        ctx.save_for_backward(conductances, boundary, solution.clone())
        layer.last_forward_stats = stats
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        conductances, boundary, solution = ctx.saved_tensors
        layer = ctx.layer
        adjoint, stats = _conjugate_gradient(
            layer,
            conductances,
            grad_output.index_select(1, layer._interior),
        )
        gradient_conductances = torch.zeros_like(conductances)
        if layer._ii_edges.numel():
            first = layer._ii_first
            second = layer._ii_second
            primal_difference = solution.index_select(1, first) - solution.index_select(1, second)
            adjoint_difference = adjoint.index_select(1, first) - adjoint.index_select(1, second)
            values = -(primal_difference * adjoint_difference).sum(dim=-1)
            gradient_conductances.index_copy_(1, layer._ii_edges, values)
        gradient_boundary = grad_output.index_select(1, layer._loop).clone()
        if layer._ib_edges.numel():
            interior = layer._ib_interior
            boundary_index = layer._ib_boundary
            primal_difference = boundary.index_select(1, boundary_index) - solution.index_select(1, interior)
            values = (adjoint.index_select(1, interior) * primal_difference).sum(dim=-1)
            gradient_conductances.index_copy_(1, layer._ib_edges, values)
            boundary_values = (
                conductances.index_select(1, layer._ib_edges)[..., None]
                * adjoint.index_select(1, interior)
            )
            gradient_boundary.index_add_(1, boundary_index, boundary_values)
        layer.last_adjoint_stats = stats
        return gradient_conductances, gradient_boundary, None


class MatrixFreeSymmetricTutteLayer(torch.nn.Module):
    """Positive symmetric conductances with matrix-free forward/adjoint CG.

    Only undirected edges incident to an interior vertex are parameterized;
    boundary-boundary conductances do not affect a Dirichlet extension and are
    intentionally omitted.  The implementation stores no Krylov history and
    supports first derivatives only.  The final hard topology screen currently
    copies the returned coordinates to CPU, so this is not yet a zero-transfer
    production GPU layer even though all linear algebra is device-native.
    """

    def __init__(
        self,
        mesh: TriMesh,
        *,
        minimum_conductance: float = 1.0e-4,
        relative_tolerance: float | None = None,
        absolute_tolerance: float = 0.0,
        max_iterations: int = 1000,
    ) -> None:
        super().__init__()
        if not np.isfinite(minimum_conductance) or minimum_conductance <= 0.0:
            raise ValueError("minimum_conductance must be finite and positive")
        if relative_tolerance is not None and (
            not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0
        ):
            raise ValueError("relative_tolerance must be finite and positive")
        if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.system = DirectedTutteSystem.from_mesh(mesh)
        self.minimum_conductance = float(minimum_conductance)
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = float(absolute_tolerance)
        self.max_iterations = int(max_iterations)
        self.last_forward_stats: KrylovStats | None = None
        self.last_adjoint_stats: KrylovStats | None = None

        boundary_set = set(self.system.loop.tolist())
        interior_index = {int(vertex): row for row, vertex in enumerate(self.system.interior)}
        boundary_index = {int(vertex): row for row, vertex in enumerate(self.system.loop)}
        all_edges = sorted(
            {
                tuple(sorted((int(first), int(second))))
                for face in mesh.faces
                for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            }
        )
        active_edges = [edge for edge in all_edges if not (edge[0] in boundary_set and edge[1] in boundary_set)]
        ii_positions: list[int] = []
        ii_first: list[int] = []
        ii_second: list[int] = []
        ib_positions: list[int] = []
        ib_interior: list[int] = []
        ib_boundary: list[int] = []
        for position, (first, second) in enumerate(active_edges):
            if first in interior_index and second in interior_index:
                ii_positions.append(position)
                ii_first.append(interior_index[first])
                ii_second.append(interior_index[second])
            else:
                interior_vertex, boundary_vertex = (
                    (first, second) if first in interior_index else (second, first)
                )
                ib_positions.append(position)
                ib_interior.append(interior_index[interior_vertex])
                ib_boundary.append(boundary_index[boundary_vertex])

        def register(name: str, values) -> None:
            self.register_buffer(name, torch.tensor(values, dtype=torch.int64), persistent=False)

        register("_active_edges_global", active_edges)
        register("_ii_edges", ii_positions)
        register("_ii_first", ii_first)
        register("_ii_second", ii_second)
        register("_ib_edges", ib_positions)
        register("_ib_interior", ib_interior)
        register("_ib_boundary", ib_boundary)
        register("_loop", self.system.loop.copy())
        register("_interior", self.system.interior.copy())

    @property
    def n_conductances(self) -> int:
        return int(self._active_edges_global.shape[0])

    @property
    def n_interior(self) -> int:
        return self.system.n_rows

    @property
    def active_edges(self) -> np.ndarray:
        return self._active_edges_global.detach().cpu().numpy().copy()

    @property
    def interior_vertices(self) -> np.ndarray:
        return self.system.interior.copy()

    @property
    def boundary_vertices(self) -> np.ndarray:
        return self.system.loop.copy()

    def conductances(self, edge_logits: torch.Tensor) -> torch.Tensor:
        values = self.minimum_conductance + torch_functional.softplus(edge_logits)
        if not bool(torch.isfinite(values).all()) or bool(torch.any(values <= 0.0)):
            raise ValueError("realized conductances must be finite and strictly positive")
        return values

    def _apply_batched(self, conductances: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(values)
        if self._ii_edges.numel():
            difference = values.index_select(1, self._ii_first) - values.index_select(1, self._ii_second)
            weighted = conductances.index_select(1, self._ii_edges)[..., None] * difference
            output.index_add_(1, self._ii_first, weighted)
            output.index_add_(1, self._ii_second, -weighted)
        if self._ib_edges.numel():
            weighted = (
                conductances.index_select(1, self._ib_edges)[..., None]
                * values.index_select(1, self._ib_interior)
            )
            output.index_add_(1, self._ib_interior, weighted)
        return output

    def apply_interior(self, conductances: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        unbatched = conductances.ndim == values.ndim - 1 == 1
        c = conductances.unsqueeze(0) if conductances.ndim == 1 else conductances
        x = values.unsqueeze(0) if values.ndim == 2 else values
        if c.ndim != 2 or c.shape[-1] != self.n_conductances:
            raise ValueError("conductances must have shape (E,) or (B,E)")
        if x.ndim != 3 or x.shape[1] != self.n_interior:
            raise ValueError("values must have shape (I,C) or (B,I,C)")
        batch = max(c.shape[0], x.shape[0])
        if c.shape[0] not in (1, batch) or x.shape[0] not in (1, batch):
            raise ValueError("conductance and value batches must match or be singleton")
        c = c.expand(batch, -1)
        x = x.expand(batch, -1, -1)
        result = self._apply_batched(c, x)
        return result[0] if unbatched else result

    def _boundary_rhs(self, conductances: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        rhs = torch.zeros(
            (conductances.shape[0], self.n_interior, 2),
            dtype=boundary.dtype,
            device=boundary.device,
        )
        if self._ib_edges.numel():
            values = (
                conductances.index_select(1, self._ib_edges)[..., None]
                * boundary.index_select(1, self._ib_boundary)
            )
            rhs.index_add_(1, self._ib_interior, values)
        return rhs

    def forward(self, edge_logits: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        for name, value, trailing in (
            ("edge_logits", edge_logits, (self.n_conductances,)),
            ("boundary", boundary, (len(self.system.loop), 2)),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch tensor")
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError(f"{name} must be float32 or float64")
            if value.ndim not in (len(trailing), len(trailing) + 1) or tuple(value.shape[-len(trailing):]) != trailing:
                raise ValueError(f"{name} must have trailing shape {trailing} and optional batch")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if edge_logits.dtype != boundary.dtype or edge_logits.device != boundary.device:
            raise TypeError("edge_logits and boundary must share dtype and device")
        if self._loop.device != edge_logits.device:
            raise ValueError("move MatrixFreeSymmetricTutteLayer to the input device before use")
        unbatched = edge_logits.ndim == 1 and boundary.ndim == 2
        z = edge_logits.unsqueeze(0) if edge_logits.ndim == 1 else edge_logits
        b = boundary.unsqueeze(0) if boundary.ndim == 2 else boundary
        batch = max(z.shape[0], b.shape[0])
        if batch == 0 or z.shape[0] not in (1, batch) or b.shape[0] not in (1, batch):
            raise ValueError("batch dimensions must match or be singleton")
        z = z.expand(batch, -1)
        b = b.expand(batch, -1, -1)
        for sample in b.detach().to(device="cpu", dtype=torch.float64).numpy():
            _validate_weakly_convex_boundary(sample)
            _validate_dividing_edges(sample, self.system.dividing_edges)
        conductances = self.conductances(z)
        # A Dirichlet conductance solve is invariant under one positive global
        # scale per sample.  Normalize before CG so finite but very large
        # logits cannot overflow residual norms or curvature products.  Keep
        # this operation in the autograd graph so the scale-gauge derivative
        # is handled by PyTorch outside the custom implicit solve.
        if self.n_conductances:
            conductance_scale = conductances.amax(dim=-1, keepdim=True)
            normalized_conductances = conductances / conductance_scale
        else:
            normalized_conductances = conductances
        if not bool(torch.isfinite(normalized_conductances).all()) or bool(
            torch.any(normalized_conductances <= 0.0)
        ):
            raise ValueError("normalized conductances must remain finite and strictly positive")
        output = _SymmetricSolve.apply(normalized_conductances, b, self)
        return output[0] if unbatched else output
