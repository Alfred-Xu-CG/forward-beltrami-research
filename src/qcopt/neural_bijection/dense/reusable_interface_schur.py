"""Exact fine-grid Tutte solve with reusable patch factors and variable interface edges.

Only active edges with no strict patch-interior endpoint may change. This is a
restricted positive-conductance family, not an all-edge fast solver.
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
from .exact_block_schur import ExactBlockSchurTutteLayer


@dataclass(frozen=True)
class ReusableInterfaceStats:
    interface_factor_seconds: float
    recover_and_validate_seconds: float
    relative_residual: float
    minimum_signed_area_ratio: float
    interface_vertices: int
    interface_nonzeros: int


@dataclass
class _StaticPatch:
    vertices: np.ndarray
    neighbors: np.ndarray
    factor: object
    coupling: sparse.csr_matrix
    inverse_coupling: np.ndarray
    base_particular: np.ndarray


class _ReusableInterfaceSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, layer: "ReusableInterfaceSchurTutteLayer") -> torch.Tensor:
        outputs = []
        factors = []
        statistics = []
        for sample in logits.detach().numpy():
            mapped, factor, stats = layer._evaluate(sample)
            outputs.append(mapped)
            factors.append(factor)
            statistics.append(stats)
        ctx.layer = layer
        ctx.factors = factors
        ctx.outputs = outputs
        ctx.save_for_backward(logits)
        layer.last_forward_stats = statistics
        return torch.from_numpy(np.stack(outputs))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (logits,) = ctx.saved_tensors
        layer = ctx.layer
        gradients = []
        times = []
        for index, (factor, mapped) in enumerate(zip(ctx.factors, ctx.outputs)):
            began = time.perf_counter()
            adjoint = layer._solve_adjoint(factor, grad_output.detach().numpy()[index])
            edges = layer.variable_edges
            primal_jump = mapped[edges[:, 0]] - mapped[edges[:, 1]]
            adjoint_jump = adjoint[edges[:, 0]] - adjoint[edges[:, 1]]
            derivative = -np.einsum("ed,ed->e", primal_jump, adjoint_jump)
            sample = logits.detach().numpy()[index]
            sigmoid = 1.0 / (1.0 + np.exp(-sample))
            gradients.append(derivative * sigmoid)
            times.append(time.perf_counter() - began)
        layer.last_adjoint_seconds = times
        return torch.from_numpy(np.stack(gradients)), None


class ReusableInterfaceSchurTutteLayer(ExactBlockSchurTutteLayer):
    """CPU float64 exact P1 map, differentiable in all interface-only logits.

    The inherited partition/assembly validates the canonical square mesh and
    Floater boundary hypotheses. All non-variable conductances equal one;
    variable conductances equal minimum+softplus(logit), always positive.
    Patch-interior factors are computed once and shared across forward calls.
    """

    def __init__(self, mesh: TriMesh, patch_cells: int, *, minimum_conductance: float = 1.0e-4) -> None:
        began = time.perf_counter()
        super().__init__(mesh, patch_cells, minimum_conductance=minimum_conductance)
        base_weights = np.ones(self.n_conductances, dtype=np.float64)
        base_matrix, base_rhs = self._assemble(base_weights)
        q = self._interface
        is_q = np.zeros(self.n_interior, dtype=bool)
        is_q[q] = True
        local_a, local_b = self._local_a, self._local_b
        allowed_a = (local_a < 0) | is_q[np.maximum(local_a, 0)]
        allowed_b = (local_b < 0) | is_q[np.maximum(local_b, 0)]
        selected = np.nonzero(allowed_a & allowed_b & ((local_a >= 0) | (local_b >= 0)))[0]
        if len(selected) == 0:
            raise RuntimeError("patch partition produced no variable interface edge")

        rows = []
        cols = []
        data = []
        starting_schur = base_matrix[q][:, q].tocoo()
        rows.append(starting_schur.row)
        cols.append(starting_schur.col)
        data.append(starting_schur.data)
        patches = []
        for block in self._blocks:
            factor = sparse_linalg.splu(base_matrix[block][:, block].tocsc())
            full_coupling = base_matrix[block][:, q].tocsr()
            neighbors = np.unique(full_coupling.indices)
            coupling = full_coupling[:, neighbors].tocsr()
            inverse_coupling = factor.solve(coupling.toarray())
            correction = np.asarray(coupling.T @ inverse_coupling)
            grid_r, grid_c = np.meshgrid(neighbors, neighbors, indexing="ij")
            rows.append(grid_r.ravel())
            cols.append(grid_c.ravel())
            data.append(-correction.ravel())
            particular = factor.solve(base_rhs[block])
            patches.append(_StaticPatch(block, neighbors, factor, coupling, inverse_coupling, particular))
        schur0 = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(q), len(q)),
        ).tocsc()
        schur0.sum_duplicates()
        baseline_factor = sparse_linalg.splu(schur0)
        baseline_reduced_rhs = base_rhs[q].copy()
        for patch in patches:
            baseline_reduced_rhs[patch.neighbors] -= patch.coupling.T @ patch.base_particular
        baseline_q = baseline_factor.solve(baseline_reduced_rhs)
        baseline_map = self._vertices.copy()
        baseline_map[self._interior[q]] = baseline_q
        for patch in patches:
            baseline_map[self._interior[patch.vertices]] = (
                patch.base_particular - patch.inverse_coupling @ baseline_q[patch.neighbors]
            )

        local_q = np.full(self.n_interior, -1, dtype=np.int64)
        local_q[q] = np.arange(len(q))
        u_rows = []
        u_cols = []
        u_data = []
        for column, edge_index in enumerate(selected):
            if local_a[edge_index] >= 0:
                u_rows.append(local_q[local_a[edge_index]])
                u_cols.append(column)
                u_data.append(1.0)
            if local_b[edge_index] >= 0:
                u_rows.append(local_q[local_b[edge_index]])
                u_cols.append(column)
                u_data.append(-1.0)
        if min(u_rows) < 0:
            raise RuntimeError("selected edge unexpectedly touches patch interior")
        u = sparse.coo_matrix(
            (u_data, (u_rows, u_cols)), shape=(len(q), len(selected))
        ).tocsc()
        variable_edges = np.stack((self._edge_a[selected], self._edge_b[selected]), axis=1)
        base_edge_jump = baseline_map[variable_edges[:, 0]] - baseline_map[variable_edges[:, 1]]

        self.variable_edge_indices = selected
        self.variable_edges = variable_edges
        self.n_variable_edges = len(selected)
        self.n_interface = len(q)
        self._u = u
        self._schur0 = schur0
        self._static_patches = patches
        self._baseline_map = baseline_map
        self._base_edge_jump = base_edge_jump
        self._base_weights = base_weights
        self.precompute_seconds = time.perf_counter() - began
        self.precomputed_bytes = int(
            sum(patch.inverse_coupling.nbytes + patch.base_particular.nbytes for patch in patches)
            + schur0.data.nbytes + schur0.indices.nbytes + schur0.indptr.nbytes
            + u.data.nbytes + u.indices.nbytes + u.indptr.nbytes
        )
        self.last_forward_stats: list[ReusableInterfaceStats] | None = None
        self.last_adjoint_seconds: list[float] | None = None

    def _conductance(self, logits: np.ndarray) -> np.ndarray:
        return self.minimum_conductance + np.logaddexp(0.0, logits)

    def _evaluate(self, logits: np.ndarray) -> tuple[np.ndarray, object, ReusableInterfaceStats]:
        if logits.shape != (self.n_variable_edges,) or not np.all(np.isfinite(logits)):
            raise ValueError("logits must be a finite vector of variable-edge length")
        conductance = self._conductance(logits)
        delta = conductance - 1.0
        factor_started = time.perf_counter()
        updated_schur = (self._schur0 + self._u @ sparse.diags(delta) @ self._u.T).tocsc()
        interface_factor = sparse_linalg.splu(updated_schur)
        factor_seconds = time.perf_counter() - factor_started
        recover_started = time.perf_counter()
        rhs = -self._u @ (delta[:, None] * self._base_edge_jump)
        delta_q = interface_factor.solve(rhs)
        mapped = self._baseline_map.copy()
        mapped[self._interior[self._interface]] += delta_q
        for patch in self._static_patches:
            mapped[self._interior[patch.vertices]] -= patch.inverse_coupling @ delta_q[patch.neighbors]

        weights = self._base_weights.copy()
        weights[self.variable_edge_indices] = conductance
        full_edges = np.stack((self._edge_a, self._edge_b), axis=1)
        flux = weights[:, None] * (mapped[full_edges[:, 0]] - mapped[full_edges[:, 1]])
        balance = np.zeros((self.n_vertices, 2), dtype=np.float64)
        np.add.at(balance, full_edges[:, 0], flux)
        np.add.at(balance, full_edges[:, 1], -flux)
        local_a, local_b = self._local_a, self._local_b
        rhs_full = np.zeros((self.n_interior, 2), dtype=np.float64)
        only_a = (local_a >= 0) & (local_b < 0)
        only_b = (local_b >= 0) & (local_a < 0)
        np.add.at(rhs_full, local_a[only_a], weights[only_a, None] * self._vertices[self._edge_b[only_a]])
        np.add.at(rhs_full, local_b[only_b], weights[only_b, None] * self._vertices[self._edge_a[only_b]])
        relative_residual = float(
            np.linalg.norm(balance[self._interior]) / max(np.linalg.norm(rhs_full), np.finfo(float).tiny)
        )
        triangles = mapped[self._faces]
        first = triangles[:, 1] - triangles[:, 0]
        second = triangles[:, 2] - triangles[:, 0]
        signed_areas = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        minimum_area = float(signed_areas.min() * (self.side - 1)**2)
        if not np.all(np.isfinite(mapped)) or relative_residual > 1e-9 or minimum_area <= 0:
            raise RuntimeError("reusable-interface solve failed residual or represented topology check")
        recover_seconds = time.perf_counter() - recover_started
        stats = ReusableInterfaceStats(
            factor_seconds, recover_seconds, relative_residual, minimum_area,
            self.n_interface, int(updated_schur.nnz),
        )
        return mapped, interface_factor, stats

    def _solve_adjoint(self, interface_factor: object, full_cotangent: np.ndarray) -> np.ndarray:
        interior_cotangent = full_cotangent[self._interior]
        reduced_rhs = interior_cotangent[self._interface].copy()
        local_solutions = []
        for patch in self._static_patches:
            particular = patch.factor.solve(interior_cotangent[patch.vertices])
            reduced_rhs[patch.neighbors] -= patch.coupling.T @ particular
            local_solutions.append(particular)
        q_solution = interface_factor.solve(reduced_rhs)
        full = np.zeros((self.n_vertices, 2), dtype=np.float64)
        full[self._interior[self._interface]] = q_solution
        for patch, particular in zip(self._static_patches, local_solutions):
            full[self._interior[patch.vertices]] = (
                particular - patch.inverse_coupling @ q_solution[patch.neighbors]
            )
        return full

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or logits.dtype != torch.float64:
            raise ValueError("reusable-interface reference requires CPU float64 logits")
        unbatched = logits.ndim == 1
        values = logits.unsqueeze(0) if unbatched else logits
        if values.ndim != 2 or values.shape[1] != self.n_variable_edges:
            raise ValueError("logits must have shape (K,) or (B,K)")
        result = _ReusableInterfaceSolve.apply(values, self)
        return result[0] if unbatched else result

    def independent_full_solve(self, logits: np.ndarray) -> tuple[np.ndarray, float]:
        weights = self._base_weights.copy()
        weights[self.variable_edge_indices] = self._conductance(np.asarray(logits))
        matrix, rhs = self._assemble(weights)
        solution = sparse_linalg.spsolve(matrix.tocsc(), rhs)
        mapped = self._vertices.copy()
        mapped[self._interior] = solution
        residual = float(np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny))
        return mapped, residual
