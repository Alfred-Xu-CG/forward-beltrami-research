"""Full tensor Whitney 1-form Hodge: an explicit compatible P1 reference.

This is not a positive-conductance or injectivity-guaranteed decoder. The
CPU/float64 sparse solve has a first-order implicit tensor and boundary VJP.
Geometry/topology are fixed; gradients through vertices are not supported.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch


class WhitneyHodge:
    """Oriented embedded triangular 2-manifold, with nondegenerate faces.

    Global edge orientation defaults to increasing vertex index. ``edge_signs``
    reverses selected edges without changing the physical operator. Faces are
    normalized CCW. A disk is required for the documented dual exactness claim;
    on multiply connected domains harmonic periods can obstruct exactness.
    """

    def __init__(self, vertices, faces, *, edge_signs=None):
        self.vertices = np.asarray(vertices, dtype=np.float64).copy()
        f = np.asarray(faces, dtype=np.int64).copy()
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("expected planar vertices and triangular faces")
        if not np.isfinite(self.vertices).all() or f.min() < 0 or f.max() >= len(vertices):
            raise ValueError("invalid vertices/faces")
        xy = self.vertices[f]
        cross = np.linalg.det(np.stack((xy[:, 1]-xy[:, 0], xy[:, 2]-xy[:, 0]), axis=2))
        if np.any(cross == 0):
            raise ValueError("degenerate triangle")
        f[cross < 0] = f[cross < 0][:, [0, 2, 1]]
        self.faces = f
        xy = self.vertices[f]
        self.areas = torch.tensor(np.abs(cross)/2)
        affine = np.concatenate((np.ones((len(f), 3, 1)), xy), axis=2)
        self.gradients = torch.tensor(np.linalg.inv(affine)[:, 1:, :].transpose(0, 2, 1).copy())
        pairs = np.stack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=1)
        edges, inverse = np.unique(np.sort(pairs.reshape(-1, 2), axis=1), axis=0, return_inverse=True)
        self.edges = edges
        self.face_edges = inverse.reshape(-1, 3)
        orientation = np.ones(len(edges)) if edge_signs is None else np.asarray(edge_signs)
        if orientation.shape != (len(edges),) or not np.isin(orientation, [-1, 1]).all():
            raise ValueError("edge_signs must have one +1/-1 per sorted edge")
        self.edge_signs = orientation
        signs = np.where(pairs[:, :, 0] < pairs[:, :, 1], 1., -1.) * orientation[self.face_edges]
        counts = np.bincount(inverse, minlength=len(edges))
        if np.any(counts > 2):
            raise ValueError("non-manifold edge")
        self.interior_edges = np.flatnonzero(counts == 2)
        self.b0_scipy = sp.coo_matrix((np.column_stack((-orientation, orientation)).ravel(),
                                      (np.repeat(np.arange(len(edges)), 2), edges.ravel())), shape=(len(edges), len(vertices))).tocsr()
        self.b1_scipy = sp.coo_matrix((signs.ravel(), (np.repeat(np.arange(len(f)), 3), inverse)), shape=(len(f), len(edges))).tocsr()
        self.b0 = self._torch_sparse(self.b0_scipy)
        self.b1 = self._torch_sparse(self.b1_scipy)
        self.dual_incidence = self.b1_scipy.T.tocsr()[self.interior_edges]
        base = torch.tensor([[-1., 1., 0.], [0., -1., 1.], [1., 0., -1.]], dtype=torch.double)
        self.local_b0 = torch.tensor(signs)[:, :, None] * base
        # Three edge midpoints integrate every quadratic polynomial exactly.
        lam = torch.tensor([[.5, .5, 0.], [0., .5, .5], [.5, 0., .5]], dtype=torch.double)
        w = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            w.append(lam[None, :, i, None]*self.gradients[:, None, j] - lam[None, :, j, None]*self.gradients[:, None, i])
        self.whitney_quadrature = torch.stack(w, dim=2)*torch.tensor(signs)[:, None, :, None]

    @staticmethod
    def _torch_sparse(matrix):
        coo = matrix.tocoo()
        return torch.sparse_coo_tensor(torch.tensor(np.stack((coo.row, coo.col))), torch.tensor(coo.data), coo.shape).coalesce()

    def local_hodge(self, tensor):
        """Integral w_e^T sym(A_T) w_f; 3x3 blocks, retaining cross-edge terms."""
        w = self.whitney_quadrature.to(tensor)
        a = (tensor + tensor.transpose(-1, -2))/2
        return self.areas.to(tensor)[:, None, None]/3 * torch.einsum("tqei,tij,tqfj->tef", w, a, w)

    def hodge(self, tensor):
        local = self.local_hodge(tensor)
        e = self.face_edges
        indices = np.stack((np.repeat(e, 3, axis=1).ravel(), np.tile(e, (1, 3)).ravel()))
        return torch.sparse_coo_tensor(torch.tensor(indices, device=tensor.device), local.reshape(-1), (len(self.edges), len(self.edges))).coalesce()

    def local_stiffness(self, tensor):
        b = self.local_b0.to(tensor)
        return b.transpose(-1, -2) @ self.local_hodge(tensor) @ b

    def stiffness(self, tensor):
        """Dense diagnostic assembly only; the solve below assembles sparse COO."""
        local = self.local_stiffness(tensor)
        f = self.faces
        indices = np.stack((np.repeat(f, 3, axis=1).ravel(), np.tile(f, (1, 3)).ravel()))
        return torch.sparse_coo_tensor(torch.tensor(indices, device=tensor.device), local.reshape(-1), (len(self.vertices), len(self.vertices))).to_dense()

    def apply(self, tensor, values):
        """Element matrix-free action, O(F) work and storage per scalar RHS."""
        scalar = values.ndim == 1
        u = values[:, None] if scalar else values
        local = self.local_stiffness(tensor) @ u[self.faces]
        out = torch.zeros_like(u).index_add(0, torch.tensor(self.faces.ravel(), device=u.device), local.reshape(-1, u.shape[1]))
        return out[:, 0] if scalar else out

    def flux_cochain(self, tensor, values):
        """q=H_A B0u, a variational dual cochain, not face-normal RT0 flux."""
        return torch.sparse.mm(self.hodge(tensor), torch.sparse.mm(self.b0.to(values), values[:, None]))[:, 0]

    def dual_stream(self, flux):
        """Least-squares dual stream on faces, pin face 0 (non-differentiable).

        Interior dual edges run from the right face to the left face of the
        oriented primal edge. Returns stream and relative Euclidean residual.
        This fits the cochain; it is not a nodal P1 conjugate reconstruction.
        """
        q = flux.detach().cpu().numpy()[self.interior_edges]
        d = self.dual_incidence[:, 1:]
        if len(q) == 0:
            return np.zeros(len(self.faces)), 0.
        v = sla.lsqr(d, q, atol=1e-13, btol=1e-13, iter_lim=max(100, 10*len(self.faces)))[0]
        residual = np.linalg.norm(d @ v-q)/max(np.linalg.norm(q), np.finfo(float).tiny)
        return np.r_[0., v], float(residual)

    def solve(self, tensor, boundary_indices, boundary_values):
        """Homogeneous interior PDE with prescribed Dirichlet values.

        CPU float64, scalar or multiple RHS. No double backward, geometry VJP,
        source/load vector, GPU factorization, or automatic topology guarantee.
        At least one Dirichlet vertex per connected component is required.
        """
        if tensor.device.type != "cpu" or tensor.dtype != torch.double or boundary_values.device.type != "cpu" or boundary_values.dtype != torch.double:
            raise ValueError("sparse reference solve requires CPU float64")
        if tensor.shape != (len(self.faces), 2, 2) or not torch.isfinite(tensor).all():
            raise ValueError("expected finite facewise 2x2 tensors")
        if torch.linalg.eigvalsh((tensor+tensor.transpose(-1, -2))/2).min() <= 0:
            raise ValueError("symmetric part of tensor must be SPD")
        b = np.asarray(boundary_indices, dtype=np.int64)
        if b.ndim != 1 or len(b) == 0 or len(np.unique(b)) != len(b) or b.min() < 0 or b.max() >= len(self.vertices):
            raise ValueError("invalid Dirichlet indices")
        if boundary_values.ndim not in (1, 2) or boundary_values.shape[0] != len(b) or not torch.isfinite(boundary_values).all():
            raise ValueError("invalid Dirichlet values")
        return _DirichletSolve.apply(tensor, boundary_values, self, b)


class _DirichletSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, boundary, system, b):
        local = system.local_stiffness(tensor).numpy()
        f = system.faces
        n = len(system.vertices)
        k = sp.coo_matrix((local.ravel(), (np.repeat(f, 3, axis=1).ravel(), np.tile(f, (1, 3)).ravel())), shape=(n, n)).tocsr()
        interior = np.setdiff1d(np.arange(n), b)
        factor = sla.splu(k[interior][:, interior].tocsc()) if len(interior) else None
        bv = boundary.numpy()
        u = np.zeros((n,) + bv.shape[1:])
        u[b] = bv
        if factor is not None:
            u[interior] = factor.solve(-k[interior][:, b] @ bv)
        result = torch.from_numpy(u)
        ctx.system, ctx.b, ctx.interior, ctx.factor, ctx.k = system, b, interior, factor, k
        ctx.save_for_backward(result)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, output_gradient):
        (u,) = ctx.saved_tensors
        s = ctx.system
        g = output_gradient.numpy()
        lam = np.zeros_like(g)
        if ctx.factor is not None:
            lam[ctx.interior] = ctx.factor.solve(g[ctx.interior], trans="T")
        gb = torch.from_numpy(g[ctx.b] - (ctx.k.T @ lam)[ctx.b])
        l = torch.from_numpy(lam)
        if u.ndim == 1:
            u, l = u[:, None], l[:, None]
        gu = torch.einsum("tvi,tvr->tir", s.gradients, u[s.faces])
        gl = torch.einsum("tvi,tvr->tir", s.gradients, l[s.faces])
        ga = -s.areas[:, None, None]*torch.einsum("tir,tjr->tij", gl, gu)
        ga = (ga+ga.transpose(-1, -2))/2
        return ga, gb, None, None
