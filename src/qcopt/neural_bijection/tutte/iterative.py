"""Directed matrix-free Tutte solves with first-order implicit differentiation.

Torch gather/reduce and scatter-add implement A=I-P_II and its transpose;
neither a sparse matrix nor a factorization is constructed. Vectorized
BiCGStab solves independent batch/RHS systems with constant-vector workspace.
Convergence is conditioning-dependent, never a fixed-iteration guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from ...forward.tutte_directed_implicit import (
    DirectedTutteSystem,
    _validate_strictly_positive_faces,
    _validate_weakly_convex_boundary,
)
from ...mesh import TriMesh


@dataclass(frozen=True)
class KrylovReport:
    """Detached CPU snapshots, shape (batch, RHS); no iteration trajectory.

    relative_residual is ||b-Ax||_2/||b||_2 per column (zero for exact zero
    RHS and residual). Acceptance uses absolute_residual <= atol+rtol*||b||.
    Diagnostics are not inputs to any forward/backward numerical computation.
    """

    iterations: torch.Tensor
    relative_residual: torch.Tensor
    absolute_residual: torch.Tensor
    converged: torch.Tensor


@dataclass
class SolveDiagnostics:
    """One invocation's handle; retain it before launching another forward.

    Backward fills this invocation's adjoint field, never a newer invocation's
    field. Repeated backwards replace its most recent adjoint report. This is
    observational mutable state, not a solver cache or an autograd dependency.
    """

    forward: KrylovReport | None = None
    adjoint: KrylovReport | None = None


class KrylovConvergenceError(RuntimeError):
    """Nonconvergence/breakdown with a detached diagnostic snapshot."""

    def __init__(self, message: str, report: KrylovReport):
        super().__init__(message)
        self.report = report


def _bicgstab(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    max_iter: int,
) -> tuple[torch.Tensor, KrylovReport]:
    """Independent BiCGStab recurrences vectorized over batch and RHS.

    Called without grad recording by the implicit Function. Only a constant
    number of vector buffers are live; no Krylov trajectory is returned or
    retained. Each iteration recomputes the true residual (an extra matvec),
    so recursive residual cancellation cannot silently certify convergence.
    All norms use max-component scaling. Dot products use scaled vectors and
    fail closed when their rescaling/product is subnormal or nonfinite; the
    guard is applied in EVERY recurrence, not only to the initial RHS. This
    backend deliberately rejects extreme magnitudes rather than silently
    losing a dot product to underflow. No preconditioner is used.
    """

    def norm(a):
        if a.shape[1] == 0:
            return a.new_zeros((a.shape[0], 1, a.shape[2]))
        scale = a.abs().amax(dim=1, keepdim=True)
        safe_scale = torch.where(scale > 0, scale, 1)
        return scale * torch.linalg.vector_norm(a / safe_scale, dim=1, keepdim=True)

    x = torch.zeros_like(rhs)
    r = rhs.clone()
    shadow = r.clone()
    norm_b = norm(rhs)
    valid_scale = (norm_b > 0) | ~(rhs != 0).any(dim=1, keepdim=True)
    tolerance = atol + rtol * norm_b
    counts = torch.zeros_like(norm_b, dtype=torch.int64)
    p, v = torch.zeros_like(rhs), torch.zeros_like(rhs)
    rho_old = torch.ones_like(norm_b)
    alpha, omega = torch.ones_like(norm_b), torch.ones_like(norm_b)
    eps = torch.finfo(rhs.dtype).eps

    def report():
        residual = norm(r)
        relative = residual / torch.where(norm_b > 0, norm_b, 1)
        def snapshot(a):
            return a.squeeze(1).detach().cpu().clone()
        converged = valid_scale & torch.isfinite(residual) & torch.isfinite(tolerance) & (residual <= tolerance)
        return KrylovReport(snapshot(counts), snapshot(relative), snapshot(residual), snapshot(converged))

    def dot(a, b, active):
        scale_a = a.abs().amax(dim=1, keepdim=True)
        scale_b = b.abs().amax(dim=1, keepdim=True)
        product_scale = scale_a * scale_b
        if bool((~torch.isfinite(product_scale) & active).any()):
            raise KrylovConvergenceError("BiCGStab breakdown: nonfinite dot-product scale", report())
        too_small = (scale_a > 0) & (scale_b > 0) & (product_scale < torch.finfo(rhs.dtype).tiny)
        if bool((too_small & active).any()):
            raise KrylovConvergenceError("BiCGStab breakdown: underflow risk in dot-product scale", report())
        normalized_a = a / torch.where(scale_a > 0, scale_a, 1)
        normalized_b = b / torch.where(scale_b > 0, scale_b, 1)
        value = (normalized_a * normalized_b).sum(dim=1, keepdim=True) * product_scale
        if bool((~torch.isfinite(value) & active).any()):
            raise KrylovConvergenceError("BiCGStab breakdown: nonfinite dot product", report())
        if bool(((value != 0) & (value.abs() < torch.finfo(rhs.dtype).tiny) & active).any()):
            raise KrylovConvergenceError("BiCGStab breakdown: underflow risk in dot product", report())
        return torch.where(active, value, 0)

    def check_breakdown(value, scale, active, name):
        bad = (~torch.isfinite(value)) | (value.abs() <= 4 * eps * scale)
        if bool((bad & active).any()):
            raise KrylovConvergenceError(f"BiCGStab breakdown in {name}", report())

    if (not bool(torch.isfinite(rhs).all()) or not bool(torch.isfinite(norm_b).all())
            or not bool(torch.isfinite(tolerance).all())):
        raise KrylovConvergenceError("BiCGStab breakdown: nonfinite RHS or residual scale", report())
    if not bool(valid_scale.all()):
        raise KrylovConvergenceError("BiCGStab breakdown: underflowed nonzero RHS norm", report())

    for _ in range(max_iter):
        active = norm(r) > tolerance
        if not bool(active.any()):
            return x, report()
        rho = dot(shadow, r, active)
        check_breakdown(rho, norm(shadow) * norm(r), active, "shadow residual")
        safe_rho_old = torch.where(active, rho_old, 1)
        safe_omega = torch.where(active, omega, 1)
        beta = (rho / safe_rho_old) * (alpha / safe_omega)
        p = torch.where(active, r + beta * (p - omega * v), 0)
        v = matvec(p)
        denominator = dot(shadow, v, active)
        check_breakdown(denominator, norm(shadow) * norm(v), active, "alpha denominator")
        alpha = torch.where(active, rho / torch.where(active, denominator, 1), 0)
        s = r - alpha * v
        half = x + alpha * p
        half_done = active & (norm(s) <= tolerance)
        working = active & ~half_done
        t = matvec(torch.where(working, s, 0))
        tt, ts = dot(t, t, working), dot(t, s, working)
        check_breakdown(tt, torch.zeros_like(tt), working, "omega denominator")
        check_breakdown(ts, norm(t) * norm(s), working, "omega numerator")
        omega = torch.where(working, ts / torch.where(working, tt, 1), 1)
        x = torch.where(working, half + omega * s, torch.where(half_done, half, x))
        counts = counts + active.to(counts.dtype)
        r = rhs - matvec(x)
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(r).all()):
            raise KrylovConvergenceError("BiCGStab breakdown: nonfinite iterate/residual", report())
        # If a recursive half-step residual was optimistic, restart just that
        # column with its true residual rather than return a false certificate.
        restart = half_done & (norm(r) > tolerance)
        shadow = torch.where(restart, r, shadow)
        p, v = torch.where(restart, 0, p), torch.where(restart, 0, v)
        rho_old = torch.where(restart, 1, rho)
        alpha, omega = torch.where(restart, 1, alpha), torch.where(restart, 1, omega)
    final = report()
    if bool(final.converged.all()):
        return x, final
    raise KrylovConvergenceError(f"BiCGStab did not converge within {max_iter} iterations", final)


@dataclass(frozen=True)
class _NeighborOperator:
    """References to this invocation's topology tensors, not mutable module state."""

    interior: torch.Tensor
    loop: torch.Tensor
    ii_indices: torch.Tensor
    ii_mask: torch.Tensor
    ib_indices: torch.Tensor
    ib_mask: torch.Tensor
    neighbor_vertices: torch.Tensor
    valid: torch.Tensor
    n_vertices: int
    faces: np.ndarray

    def matvec(self, probabilities, x, *, transpose=False):
        if x.shape[1] == 0:
            return x.clone()
        weights = probabilities * self.ii_mask
        if not transpose:
            return x - (weights.unsqueeze(-1) * x[:, self.ii_indices, :]).sum(dim=2)
        contributions = (weights.unsqueeze(-1) * x.unsqueeze(2)).flatten(1, 2)
        indices = self.ii_indices.flatten()[None, :, None].expand_as(contributions)
        return x - torch.zeros_like(x).scatter_add(1, indices, contributions)

    def rhs(self, probabilities, boundary):
        return ((probabilities * self.ib_mask).unsqueeze(-1)
                * boundary[:, self.ib_indices, :]).sum(dim=2)

    def boundary_adjoint(self, probabilities, adjoint, direct_gradient):
        contributions = ((probabilities * self.ib_mask).unsqueeze(-1)
                         * adjoint.unsqueeze(2)).flatten(1, 2)
        indices = self.ib_indices.flatten()[None, :, None].expand_as(contributions)
        return direct_gradient.scatter_add(1, indices, contributions)


class _ImplicitMatrixFree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities, boundary, operator, settings, diagnostics):
        rhs = operator.rhs(probabilities, boundary)
        try:
            interior, diagnostics.forward = _bicgstab(
                lambda x: operator.matvec(probabilities, x), rhs, **settings)
        except KrylovConvergenceError as error:
            diagnostics.forward = error.report
            raise
        output = boundary.new_empty((len(probabilities), operator.n_vertices, 2))
        output[:, operator.loop] = boundary
        output[:, operator.interior] = interior
        # Check returned coordinates, not a hidden higher-precision solution.
        for mapped in output.detach().double().cpu().numpy():
            _validate_strictly_positive_faces(operator.faces, mapped)
        ctx.operator, ctx.settings, ctx.diagnostics = operator, settings, diagnostics
        # A separate primal tensor prevents legal in-place output operations
        # from altering the implicit derivative. No Krylov vector is saved.
        ctx.save_for_backward(probabilities, output.clone())
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        probabilities, primal = ctx.saved_tensors
        operator = ctx.operator
        try:
            adjoint, ctx.diagnostics.adjoint = _bicgstab(
                lambda x: operator.matvec(probabilities, x, transpose=True),
                gradient[:, operator.interior], **ctx.settings)
        except KrylovConvergenceError as error:
            ctx.diagnostics.adjoint = error.report
            raise
        dp = (adjoint.unsqueeze(2) * primal[:, operator.neighbor_vertices, :]).sum(-1)
        dp = dp * operator.valid
        db = operator.boundary_adjoint(probabilities, adjoint, gradient[:, operator.loop])
        return dp, db, None, None, None


class MatrixFreeDirectedTutteLayer(torch.nn.Module):
    """CPU/CUDA torch matrix-free directed reference with implicit first VJP.

    API matches DirectTutteLayer: logits (I,D)/(batch,I,D), boundary
    (B,2)/(batch,B,2) in source boundary-loop order, singleton broadcasting;
    output is in source vertex order. Move this module to the input device.
    Inputs share CPU/CUDA device and float32/64 dtype; Krylov arithmetic uses
    that dtype without silently promoting float32. Masked slots have zero
    probabilities, supported slots must stay strictly positive.

    Defaults: rtol=5e-6 for float32, 1e-10 for float64; atol=0, max_iter=500.
    Each RHS is accepted only if its TRUE norm residual <= atol+rtol*||rhs||.
    A failed forward or adjoint raises KrylovConvergenceError; tiny residuals
    alone do not bound solution/gradient error without conditioning estimates.
    Max-scaled norms prevent residual-square underflow; recurrence dot products
    are scaled and extreme subnormal/overflow ranges are explicitly rejected.

    last_diagnostics is a per-invocation handle containing only detached CPU
    final reports. Retain the handle to inspect an earlier graph's backward.
    It does not retain inputs, solutions, or iteration history. The autograd
    context saves only probabilities and an independent primal (plus fixed
    topology and scalar settings). Workspace is O(batch*I*D*RHS), independent
    of iteration limit. No sparse assembly, SciPy solve, or preconditioner.

    Geometry validation reuses the certified CPU source/boundary/face checks,
    requiring host copies/synchronization of boundary/output even on CUDA.
    Iteration stopping also synchronizes scalar decisions; scatter-add may be
    nondeterministic on CUDA. GPU functionality is not a speed/production claim.
    Positive output areas certify the represented P1 map under the validated
    source-disk and simple boundary assumptions, not arbitrary sampled maps.
    """

    def __init__(self, mesh: TriMesh, *, rtol: float | None = None, atol: float = 0,
                 max_iter: int = 500):
        super().__init__()
        if rtol is not None and (not math.isfinite(rtol) or rtol <= 0):
            raise ValueError("rtol must be finite and positive")
        if not math.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and nonnegative")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.rtol, self.atol, self.max_iter = rtol, atol, max_iter
        self.last_diagnostics: SolveDiagnostics | None = None
        self.system = DirectedTutteSystem.from_mesh(mesh)
        s = self.system
        ii_mask = s.valid_mask & ~s.neighbor_is_boundary
        ib_mask = s.valid_mask & s.neighbor_is_boundary
        ii = np.where(ii_mask, s.neighbors, 0)
        ib = np.where(ib_mask, s.neighbors, 0)
        global_neighbors = np.zeros_like(s.neighbors)
        global_neighbors[ii_mask] = s.interior[s.neighbors[ii_mask]]
        global_neighbors[ib_mask] = s.loop[s.neighbors[ib_mask]]
        for name, data in (("_interior", s.interior), ("_loop", s.loop),
                           ("_ii_indices", ii), ("_ii_mask", ii_mask),
                           ("_ib_indices", ib), ("_ib_mask", ib_mask),
                           ("_neighbor_vertices", global_neighbors), ("_valid", s.valid_mask)):
            self.register_buffer(name, torch.from_numpy(np.array(data, copy=True)), persistent=False)

    def _operator(self):
        return _NeighborOperator(self._interior, self._loop, self._ii_indices, self._ii_mask,
                                 self._ib_indices, self._ib_mask, self._neighbor_vertices,
                                 self._valid, self.system.n_vertices, self.system.faces)

    def matvec(self, probabilities: torch.Tensor, x: torch.Tensor, *, transpose=False):
        """Apply A or A^T to batched RHS; probabilities are already normalized.

        This exposed linear operator is also useful for explicit residual checks.
        Shapes are (batch,I,D) and (batch,I,RHS), on the module's device.
        """
        return self._operator().matvec(probabilities, x, transpose=transpose)

    def forward(self, logits: torch.Tensor, boundary: torch.Tensor):
        s = self.system
        for name, value, shape in (("logits", logits, (s.n_rows, s.max_degree)),
                                   ("boundary", boundary, (len(s.loop), 2))):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch tensor")
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError(f"{name} must have float32/64 dtype")
            if value.device.type not in ("cpu", "cuda"):
                raise ValueError("only CPU and CUDA are supported")
            if value.ndim not in (2, 3) or tuple(value.shape[-2:]) != shape:
                raise ValueError(f"{name} must have trailing shape {shape} and optional batch")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if logits.dtype != boundary.dtype or logits.device != boundary.device:
            raise ValueError("logits and boundary must have the same dtype and device")
        if self._valid.device != logits.device:
            raise ValueError("move MatrixFreeDirectedTutteLayer to the input device before use")
        unbatched = logits.ndim == boundary.ndim == 2
        z = logits.unsqueeze(0) if logits.ndim == 2 else logits
        b = boundary.unsqueeze(0) if boundary.ndim == 2 else boundary
        batch = max(z.shape[0], b.shape[0])
        if batch == 0 or z.shape[0] not in (1, batch) or b.shape[0] not in (1, batch):
            raise ValueError("batch dimensions must match or be singleton")
        z, b = z.expand(batch, -1, -1), b.expand(batch, -1, -1)
        for sample in b.detach().double().cpu().numpy():
            _validate_weakly_convex_boundary(sample)
        probabilities = torch.softmax(z.masked_fill(~self._valid, -torch.inf), dim=-1)
        if not bool(torch.isfinite(probabilities).all()) or bool((probabilities[:, self._valid] <= 0).any()):
            raise ValueError("supported probabilities must be finite and strictly positive")
        settings = dict(rtol=self.rtol if self.rtol is not None else
                        (5e-6 if logits.dtype == torch.float32 else 1e-10),
                        atol=self.atol, max_iter=self.max_iter)
        diagnostics = SolveDiagnostics()
        self.last_diagnostics = diagnostics
        output = _ImplicitMatrixFree.apply(probabilities, b, self._operator(), settings, diagnostics)
        return output[0] if unbatched else output
