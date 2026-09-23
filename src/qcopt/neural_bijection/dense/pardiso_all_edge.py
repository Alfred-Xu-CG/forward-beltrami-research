"""Optional exact all-edge SPD Tutte layer using oneMKL PARDISO.

The conductance family, assembled Dirichlet system, represented mesh, and
implicit edge gradient are identical to ExactBlockSchurTutteLayer. Only the
CPU factorization backend changes; PyPardiso is imported lazily.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sparse
from scipy.special import expit
import torch
from torch.autograd.function import once_differentiable

from .exact_block_schur import ExactBlockSchurTutteLayer


class _PardisoFactor:
    def __init__(self, solver: object, upper: sparse.csr_matrix, forward_map: np.ndarray) -> None:
        self.solver = solver
        self.upper = upper
        self.forward_map = forward_map
        self.closed = False

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self.closed:
            raise RuntimeError("PARDISO factor was already released")
        return self.solver.solve(self.upper, rhs)

    def close(self) -> None:
        if not self.closed:
            self.solver.free_memory(everything=True)
            self.closed = True

    def __del__(self) -> None:
        self.close()


class _PardisoAllEdgeSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, layer: "PardisoAllEdgeTutteLayer") -> torch.Tensor:
        outputs = []
        factors = []
        stats = []
        try:
            for sample in logits.detach().numpy():
                mapped, factor, measured = layer._evaluate_pardiso(sample)
                outputs.append(mapped)
                factors.append(factor)
                stats.append(measured)
        except Exception:
            for factor in factors:
                factor.close()
            raise
        ctx.layer = layer
        ctx.factors = factors
        ctx.save_for_backward(logits)
        layer.last_forward_stats = stats
        return torch.from_numpy(np.stack(outputs))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (logits,) = ctx.saved_tensors
        layer = ctx.layer
        cotangents = grad_output.detach().numpy()
        values = logits.detach().numpy()
        gradients = []
        timings = []
        try:
            for batch_index, factor in enumerate(ctx.factors):
                began = time.perf_counter()
                adjoint_interior = factor.solve(cotangents[batch_index, layer._interior])
                adjoint = np.zeros((layer.n_vertices, 2), dtype=np.float64)
                adjoint[layer._interior] = adjoint_interior
                primal_difference = factor.forward_map[layer._edge_a] - factor.forward_map[layer._edge_b]
                adjoint_difference = adjoint[layer._edge_a] - adjoint[layer._edge_b]
                grad_conductance = -np.einsum("ed,ed->e", primal_difference, adjoint_difference)
                gradients.append(grad_conductance * expit(values[batch_index]))
                timings.append(time.perf_counter() - began)
        finally:
            for factor in ctx.factors:
                factor.close()
            ctx.factors = []
        layer.last_adjoint_seconds = timings
        return torch.from_numpy(np.stack(gradients)), None


class PardisoAllEdgeTutteLayer(ExactBlockSchurTutteLayer):
    """Full fine-grid positive-conductance P1 layer with exact implicit VJP.

    Input shape is (E,) or (B,E), where E is every active fine-grid edge.
    Requires CPU float64 tensors and optional PyPardiso. No low-rank or
    interface-only restriction is imposed; factors are rebuilt for changed
    conductances and released after backward or no-gradient inference.
    """

    def _evaluate_pardiso(self, logits: np.ndarray) -> tuple[np.ndarray, _PardisoFactor, dict[str, float]]:
        try:
            from pypardiso import PyPardisoSolver
        except ImportError as error:
            raise RuntimeError("PyPardiso is required for PardisoAllEdgeTutteLayer") from error
        if logits.shape != (self.n_conductances,) or not np.all(np.isfinite(logits)):
            raise ValueError("logits must be a finite active-edge vector")
        conductance = self.minimum_conductance + np.logaddexp(0.0, logits)
        began = time.perf_counter()
        matrix, rhs = self._assemble(conductance)
        upper = sparse.triu(matrix, format="csr")
        assembly_seconds = time.perf_counter() - began
        solver = PyPardisoSolver(mtype=2)
        try:
            began = time.perf_counter()
            solver.factorize(upper)
            factor_seconds = time.perf_counter() - began
            began = time.perf_counter()
            interior_solution = solver.solve(upper, rhs)
            mapped = self._vertices.copy()
            mapped[self._interior] = interior_solution
            relative_residual = float(
                np.linalg.norm(matrix @ interior_solution - rhs)
                / max(np.linalg.norm(rhs), np.finfo(float).tiny)
            )
            triangles = mapped[self._faces]
            first = triangles[:, 1] - triangles[:, 0]
            second = triangles[:, 2] - triangles[:, 0]
            signed_double_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
            minimum_area_ratio = float(signed_double_area.min() * (self.side - 1) ** 2)
            if not np.all(np.isfinite(mapped)) or not np.isfinite(relative_residual) or minimum_area_ratio <= 0:
                raise RuntimeError("exact SPD solve produced a nonfinite or folded P1 map")
            solve_and_check_seconds = time.perf_counter() - began
        except Exception:
            solver.free_memory(everything=True)
            raise
        factor = _PardisoFactor(solver, upper, mapped)
        stats = {
            "assembly_seconds": assembly_seconds,
            "factor_seconds": factor_seconds,
            "solve_and_check_seconds": solve_and_check_seconds,
            "relative_residual": relative_residual,
            "minimum_signed_area_ratio": minimum_area_ratio,
        }
        return mapped, factor, stats

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or logits.dtype != torch.float64:
            raise ValueError("PARDISO exact layer requires a CPU float64 tensor")
        unbatched = logits.ndim == 1
        values = logits.unsqueeze(0) if unbatched else logits
        if values.ndim != 2 or values.shape[1] != self.n_conductances:
            raise ValueError("logits must have shape (E,) or (B,E)")
        if not values.requires_grad:
            outputs = []
            stats = []
            for sample in values.detach().numpy():
                mapped, factor, measured = self._evaluate_pardiso(sample)
                try:
                    outputs.append(mapped)
                    stats.append(measured)
                finally:
                    factor.close()
            self.last_forward_stats = stats
            result = torch.from_numpy(np.stack(outputs))
        else:
            result = _PardisoAllEdgeSolve.apply(values, self)
        return result[0] if unbatched else result
