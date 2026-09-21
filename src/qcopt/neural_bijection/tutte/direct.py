"""CPU reference directed Tutte layer with forward/adjoint SuperLU reuse.

This is not a GPU or production solver. Each sample factors I-P_II once per
forward. Both coordinate RHS and subsequent transpose adjoints share that LU.
No numeric factors are reused across different forwards/parameter values.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch
from torch.autograd.function import once_differentiable

from ...forward.tutte_directed_implicit import (
    DirectedTutteSystem,
    _validate_strictly_positive_faces,
    _validate_weakly_convex_boundary,
)
from ...mesh import TriMesh


class _DirectSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities, boundary, layer):
        system = layer.system
        p = probabilities.detach().double().numpy()
        b = boundary.detach().double().numpy()
        result = np.empty((len(p), system.n_vertices, 2), dtype=np.float64)
        factors = []
        for sample in range(len(p)):
            result[sample, system.loop] = b[sample]
            if system.n_rows == 0:
                factors.append(None)
                continue
            values = p[sample, layer._ii_rows, layer._ii_slots]
            matrix = sparse.eye(system.n_rows, format="csc") - sparse.csc_matrix(
                (values, (layer._ii_rows, layer._ii_neighbors)),
                shape=(system.n_rows, system.n_rows),
            )
            rhs = np.zeros((system.n_rows, 2))
            np.add.at(rhs, layer._ib_rows,
                      p[sample, layer._ib_rows, layer._ib_slots, None] * b[sample, layer._ib_neighbors])
            try:
                factor = sparse_linalg.splu(matrix)
                result[sample, system.interior] = factor.solve(rhs)
            except RuntimeError as exc:
                raise ValueError("linear solve failed; positive output areas cannot be certified") from exc
            factors.append(factor)
        output = torch.from_numpy(result).to(dtype=boundary.dtype)
        # Certify the returned precision, not only the hidden double solution.
        certified = output.double().numpy()
        if not np.isfinite(certified).all():
            raise ValueError("direct Tutte output must be finite")
        for mapped in certified:
            _validate_strictly_positive_faces(system.faces, mapped)
        ctx.factors = factors
        ctx.layer = layer
        # The float64 output shares result's NumPy storage. Preserve a separate
        # immutable primal so legal output in-place operations cannot silently
        # alter the implicit VJP (float32 output already has separate storage).
        ctx.solution = result.copy()
        ctx.solution.setflags(write=False)
        ctx.save_for_backward(probabilities, boundary)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        probabilities, boundary = ctx.saved_tensors
        layer, system = ctx.layer, ctx.layer.system
        p = probabilities.detach().double().numpy()
        g = grad_output.detach().double().numpy()
        dp = np.zeros_like(p)
        db = g[:, system.loop].copy()
        for sample, factor in enumerate(ctx.factors):
            if factor is None:
                continue
            adjoint = factor.solve(g[sample, system.interior], trans="T")
            # dX_I=A^-1(dP_II X_I+dP_IB B+P_IB dB).
            dp[sample, layer._rows, layer._slots] = np.sum(
                adjoint[layer._rows] * ctx.solution[sample, layer._neighbor_vertices], axis=1
            )
            np.add.at(db[sample], layer._ib_neighbors,
                      p[sample, layer._ib_rows, layer._ib_slots, None] * adjoint[layer._ib_rows])
        return (torch.from_numpy(dp).to(probabilities.dtype),
                torch.from_numpy(db).to(boundary.dtype), None)


class DirectTutteLayer(torch.nn.Module):
    """Factor-reusing CPU reference for a fixed triangulated disk.

    Tested mesh family: ``structured_rectangle(nx, ny)`` (cell counts).
    ``logits`` has shape ``(I,D)`` or ``(batch,I,D)``; slots correspond to
    sorted neighboring vertex IDs in ``system``. ``boundary`` has shape
    ``(B,2)`` or ``(batch,B,2)``, in ``mesh.boundary_loops[0]`` order, NOT
    sorted vertex order. Singleton batches broadcast. Output is in global
    vertex order, unbatched only when both inputs are unbatched.

    Inputs must be CPU float32/float64 with the same dtype. Supported softmax
    probabilities must remain strictly positive in that dtype. Boundary must
    be a simple positively oriented weakly convex polygon; the returned map
    additionally passes a numerical positive-original-face-area certificate.
    This is fail-closed validation, not repair or an exact-arithmetic proof.
    Geometry must satisfy the triangulated-disk assumptions; arbitrary mesh
    topology is not independently certified here. First derivatives only.

    Factors live with each autograd graph until that graph is released. Batch
    factorization/solves are sequential; there is no cross-forward symbolic
    cache. The inherited boundary validator is quadratic in boundary size.
    """

    def __init__(self, mesh: TriMesh):
        super().__init__()
        self.system = DirectedTutteSystem.from_mesh(mesh)
        s = self.system
        self._rows, self._slots = np.nonzero(s.valid_mask)
        is_boundary = s.neighbor_is_boundary[self._rows, self._slots]
        neighbors = s.neighbors[self._rows, self._slots]
        self._ii_rows = self._rows[~is_boundary]
        self._ii_slots = self._slots[~is_boundary]
        self._ii_neighbors = neighbors[~is_boundary]
        self._ib_rows = self._rows[is_boundary]
        self._ib_slots = self._slots[is_boundary]
        self._ib_neighbors = neighbors[is_boundary]
        self._neighbor_vertices = np.empty_like(neighbors)
        self._neighbor_vertices[is_boundary] = s.loop[neighbors[is_boundary]]
        self._neighbor_vertices[~is_boundary] = s.interior[neighbors[~is_boundary]]
        self._valid_mask = torch.from_numpy(s.valid_mask.copy())

    def forward(self, logits: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        s = self.system
        for name, value, shape in (("logits", logits, (s.n_rows, s.max_degree)),
                                   ("boundary", boundary, (len(s.loop), 2))):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch tensor")
            if value.device.type != "cpu":
                raise ValueError("DirectTutteLayer is a CPU reference; CPU inputs are required")
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError(f"{name} must be float32 or float64")
            if value.ndim not in (2, 3) or tuple(value.shape[-2:]) != shape:
                raise ValueError(f"{name} must have trailing shape {shape} and optional batch")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if logits.dtype != boundary.dtype:
            raise TypeError("logits and boundary must have the same dtype")
        unbatched = logits.ndim == boundary.ndim == 2
        z = logits.unsqueeze(0) if logits.ndim == 2 else logits
        b = boundary.unsqueeze(0) if boundary.ndim == 2 else boundary
        batch = max(z.shape[0], b.shape[0])
        if batch == 0 or z.shape[0] not in (1, batch) or b.shape[0] not in (1, batch):
            raise ValueError("batch dimensions must match or be singleton")
        z, b = z.expand(batch, -1, -1), b.expand(batch, -1, -1)
        for sample in b.detach().double().numpy():
            _validate_weakly_convex_boundary(sample)
        p = torch.softmax(z.masked_fill(~self._valid_mask, -torch.inf), dim=-1)
        if not torch.isfinite(p).all() or torch.any(p[:, self._valid_mask] <= 0):
            raise ValueError("supported softmax probabilities must be finite and strictly positive")
        output = _DirectSolve.apply(p, b, self)
        return output[0] if unbatched else output
