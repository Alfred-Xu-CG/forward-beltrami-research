"""Exact patch-interior Schur solve and implicit VJP for positive Tutte weights.

This is a CPU float64 reference implementation. It eliminates every patch
interior exactly and solves the complete interface Schur complement, then
recovers every vertex on the original fine mesh. It is not an approximate
multigrid or a coarse-map interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch
from torch.autograd.function import once_differentiable

from ...mesh import TriMesh
from ..tutte.symmetric import MatrixFreeSymmetricTutteLayer


@dataclass(frozen=True)
class BlockSchurStats:
    assembly_seconds: float
    block_factor_seconds: float
    interface_factor_seconds: float
    recover_seconds: float
    relative_residual: float
    minimum_signed_area_ratio: float
    interface_vertices: int
    schur_nonzeros: int
    factors_nonzeros: int


@dataclass
class _PatchFactor:
    vertices: np.ndarray
    neighbors: np.ndarray
    factor: object
    coupling: sparse.csr_matrix
    inverse_coupling: np.ndarray


@dataclass
class _FactorContext:
    matrix: sparse.csr_matrix
    interface_factor: object
    patches: list[_PatchFactor]
    forward_map: np.ndarray


class _ExactBlockSchurSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, layer: "ExactBlockSchurTutteLayer") -> torch.Tensor:
        values = logits.detach().numpy()
        outputs = []
        factors = []
        statistics = []
        for sample in values:
            mapped, factor, stats = layer._evaluate(sample)
            outputs.append(mapped)
            factors.append(factor)
            statistics.append(stats)
        ctx.layer = layer
        ctx.factors = factors
        ctx.save_for_backward(logits)
        layer.last_forward_stats = statistics
        return torch.from_numpy(np.stack(outputs))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (logits,) = ctx.saved_tensors
        layer = ctx.layer
        cotangents = grad_output.detach().numpy()
        gradients = []
        times = []
        for batch_index, factor in enumerate(ctx.factors):
            started = time.perf_counter()
            interior_cotangent = cotangents[batch_index, layer._interior]
            adjoint_interior = layer._solve_with_factor(factor, interior_cotangent)
            adjoint = np.zeros((layer.n_vertices, 2), dtype=np.float64)
            adjoint[layer._interior] = adjoint_interior
            primal_difference = factor.forward_map[layer._edge_a] - factor.forward_map[layer._edge_b]
            adjoint_difference = adjoint[layer._edge_a] - adjoint[layer._edge_b]
            grad_conductance = -np.einsum("ed,ed->e", primal_difference, adjoint_difference)
            sigmoid = 1.0 / (1.0 + np.exp(-logits.detach().numpy()[batch_index]))
            gradients.append(grad_conductance * sigmoid)
            times.append(time.perf_counter() - started)
        layer.last_adjoint_seconds = times
        return torch.from_numpy(np.stack(gradients)), None


class ExactBlockSchurTutteLayer(torch.nn.Module):
    """All-edge positive conductance P1 layer with exact block elimination.

    Input ``logits`` has shape (E,) or (B,E), where E is every edge touching
    an interior vertex. Conductances equal ``minimum+softplus(logits)``.
    Only CPU float64 is supported in this reference implementation; factors
    are rebuilt for every new conductance sample and reused for its adjoint.
    """

    def __init__(self, mesh: TriMesh, patch_cells: int, *, minimum_conductance: float = 1.0e-4) -> None:
        super().__init__()
        side = round(np.sqrt(mesh.n_vertices))
        if side * side != mesh.n_vertices or side < 3 or (side - 1) % patch_cells:
            raise ValueError("mesh must be a square structured grid divisible into patches")
        if patch_cells < 2 or patch_cells >= side:
            raise ValueError("invalid patch_cells")
        if not np.isfinite(minimum_conductance) or minimum_conductance <= 0:
            raise ValueError("minimum_conductance must be positive and finite")
        reference = MatrixFreeSymmetricTutteLayer(mesh)
        edges = reference.active_edges
        interior = reference.interior_vertices
        n = len(interior)
        local_index = np.full(mesh.n_vertices, -1, dtype=np.int64)
        local_index[interior] = np.arange(n)
        endpoints_a = edges[:, 0]
        endpoints_b = edges[:, 1]
        local_a = local_index[endpoints_a]
        local_b = local_index[endpoints_b]
        source_i = interior % side
        source_j = interior // side
        interface_mask = (source_i % patch_cells == 0) | (source_j % patch_cells == 0)
        interface = np.nonzero(interface_mask)[0]
        blocks = []
        for by in range((side - 1) // patch_cells):
            for bx in range((side - 1) // patch_cells):
                vertex_mask = (
                    ~interface_mask
                    & (source_i // patch_cells == bx)
                    & (source_j // patch_cells == by)
                )
                block = np.nonzero(vertex_mask)[0]
                if len(block) != (patch_cells - 1) ** 2:
                    raise RuntimeError("patch partition has an unexpected interior size")
                blocks.append(block)
        block_owner = np.full(n, -1, dtype=np.int64)
        for block_index, block in enumerate(blocks):
            block_owner[block] = block_index
        both_interior = (local_a >= 0) & (local_b >= 0)
        cross_block = (
            both_interior
            & (block_owner[np.maximum(local_a, 0)] >= 0)
            & (block_owner[np.maximum(local_b, 0)] >= 0)
            & (block_owner[np.maximum(local_a, 0)] != block_owner[np.maximum(local_b, 0)])
        )
        if np.any(cross_block):
            raise RuntimeError("an edge joins two distinct patch interiors")
        self.side = side
        self.patch_cells = patch_cells
        self.n_vertices = mesh.n_vertices
        self.n_interior = n
        self.n_conductances = len(edges)
        self.minimum_conductance = float(minimum_conductance)
        self._vertices = np.array(mesh.vertices, copy=True)
        self._faces = np.array(mesh.faces, copy=True)
        self._interior = interior
        self._edge_a = endpoints_a
        self._edge_b = endpoints_b
        self._local_a = local_a
        self._local_b = local_b
        self._interface = interface
        self._blocks = blocks
        self.last_forward_stats: list[BlockSchurStats] | None = None
        self.last_adjoint_seconds: list[float] | None = None

    def _assemble(self, conductance: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
        n = self.n_interior
        local_a = self._local_a
        local_b = self._local_b
        has_a = local_a >= 0
        has_b = local_b >= 0
        both = has_a & has_b
        diagonal = np.zeros(n, dtype=np.float64)
        np.add.at(diagonal, local_a[has_a], conductance[has_a])
        np.add.at(diagonal, local_b[has_b], conductance[has_b])
        rows = np.concatenate((local_a[both], local_b[both]))
        cols = np.concatenate((local_b[both], local_a[both]))
        off = np.concatenate((-conductance[both], -conductance[both]))
        matrix = (sparse.diags(diagonal) + sparse.coo_matrix((off, (rows, cols)), shape=(n, n))).tocsr()
        rhs = np.zeros((n, 2), dtype=np.float64)
        only_a = has_a & ~has_b
        only_b = has_b & ~has_a
        np.add.at(rhs, local_a[only_a], conductance[only_a, None] * self._vertices[self._edge_b[only_a]])
        np.add.at(rhs, local_b[only_b], conductance[only_b, None] * self._vertices[self._edge_a[only_b]])
        return matrix, rhs

    def _factor(self, matrix: sparse.csr_matrix) -> tuple[_FactorContext, float, float, int, int]:
        q = self._interface
        schur = matrix[q][:, q].tocoo()
        rows = [schur.row]
        cols = [schur.col]
        data = [schur.data]
        patch_factors = []
        block_started = time.perf_counter()
        factor_nnz = 0
        for block in self._blocks:
            app = matrix[block][:, block].tocsc()
            factor = sparse_linalg.splu(app)
            factor_nnz += int(factor.L.nnz + factor.U.nnz)
            apq_full = matrix[block][:, q].tocsr()
            neighbors = np.unique(apq_full.indices)
            apq = apq_full[:, neighbors].tocsr()
            z = factor.solve(apq.toarray())
            correction = np.asarray(apq.T @ z)
            grid_r, grid_c = np.meshgrid(neighbors, neighbors, indexing="ij")
            rows.append(grid_r.ravel())
            cols.append(grid_c.ravel())
            data.append(-correction.ravel())
            patch_factors.append(_PatchFactor(block, neighbors, factor, apq, z))
        block_seconds = time.perf_counter() - block_started
        interface_started = time.perf_counter()
        schur_matrix = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(q), len(q)),
        ).tocsc()
        schur_matrix.sum_duplicates()
        interface_factor = sparse_linalg.splu(schur_matrix)
        interface_seconds = time.perf_counter() - interface_started
        factor_nnz += int(interface_factor.L.nnz + interface_factor.U.nnz)
        context = _FactorContext(matrix, interface_factor, patch_factors, np.empty((self.n_vertices, 2)))
        return context, block_seconds, interface_seconds, int(schur_matrix.nnz), factor_nnz

    def _solve_with_factor(self, context: _FactorContext, rhs: np.ndarray) -> np.ndarray:
        q = self._interface
        reduced_rhs = rhs[q].copy()
        local_rhs = []
        for patch in context.patches:
            y = patch.factor.solve(rhs[patch.vertices])
            reduced_rhs[patch.neighbors] -= patch.coupling.T @ y
            local_rhs.append(y)
        q_solution = context.interface_factor.solve(reduced_rhs)
        result = np.empty_like(rhs)
        result[q] = q_solution
        for patch, y in zip(context.patches, local_rhs):
            result[patch.vertices] = y - patch.inverse_coupling @ q_solution[patch.neighbors]
        return result

    def _evaluate(self, logits: np.ndarray) -> tuple[np.ndarray, _FactorContext, BlockSchurStats]:
        if logits.shape != (self.n_conductances,) or not np.all(np.isfinite(logits)):
            raise ValueError("logits must be a finite vector of active-edge length")
        conductance = self.minimum_conductance + np.logaddexp(0.0, logits)
        start = time.perf_counter()
        matrix, rhs = self._assemble(conductance)
        assembly_seconds = time.perf_counter() - start
        context, block_seconds, interface_seconds, schur_nnz, factor_nnz = self._factor(matrix)
        recover_start = time.perf_counter()
        interior_solution = self._solve_with_factor(context, rhs)
        mapped = self._vertices.copy()
        mapped[self._interior] = interior_solution
        residual = matrix @ interior_solution - rhs
        relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny))
        triangles = mapped[self._faces]
        a = triangles[:, 1] - triangles[:, 0]
        b = triangles[:, 2] - triangles[:, 0]
        areas = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        minimum_ratio = float(areas.min() * (self.side - 1) ** 2)
        if not np.all(np.isfinite(mapped)) or minimum_ratio <= 0:
            raise RuntimeError("exact block Schur solve returned a folded or nonfinite P1 map")
        context.forward_map = mapped
        recover_seconds = time.perf_counter() - recover_start
        stats = BlockSchurStats(
            assembly_seconds, block_seconds, interface_seconds, recover_seconds,
            relative_residual, minimum_ratio, len(self._interface), schur_nnz, factor_nnz,
        )
        return mapped, context, stats

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or logits.dtype != torch.float64:
            raise ValueError("this exact Schur reference requires a CPU float64 tensor")
        unbatched = logits.ndim == 1
        values = logits.unsqueeze(0) if unbatched else logits
        if values.ndim != 2 or values.shape[1] != self.n_conductances:
            raise ValueError("logits must have shape (E,) or (B,E)")
        result = _ExactBlockSchurSolve.apply(values, self)
        return result[0] if unbatched else result

    def independent_full_solve(self, logits: np.ndarray) -> tuple[np.ndarray, float]:
        """Independent factorization of the assembled complete system."""
        conductance = self.minimum_conductance + np.logaddexp(0.0, np.asarray(logits))
        matrix, rhs = self._assemble(conductance)
        solution = sparse_linalg.spsolve(matrix.tocsc(), rhs)
        mapped = self._vertices.copy()
        mapped[self._interior] = solution
        residual = float(np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny))
        return mapped, residual
