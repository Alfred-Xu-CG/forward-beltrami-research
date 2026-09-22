"""Exact low-rank positive-edge updates of one fine-grid Tutte system.

The base factorization is a one-time CPU setup. Repeated latent-dependent
forward/VJP calls use the stored inverse-action columns, not a dense inverse.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch
import torch.nn.functional as F

from ...mesh import TriMesh
from ..tutte.symmetric import MatrixFreeSymmetricTutteLayer


class SparseEdgeWoodburyTutteLayer(torch.nn.Module):
    """Fine-grid P1 Tutte map from positive increments on selected edges.

    ``selected_edges`` indexes the full active-edge ordering of
    ``MatrixFreeSymmetricTutteLayer.active_edges``. A single base Dirichlet
    factorization supplies W=A0^{-1}U; no dense n-by-n inverse is created.
    The fixed boundary is the original mesh boundary coordinates, so the
    present implementation is for an ordered unit-square triangulation.
    """

    def __init__(
        self,
        mesh: TriMesh,
        selected_edges: np.ndarray | list[int],
        *,
        minimum_increment: float = 1.0e-4,
        validate_faces: bool = True,
    ) -> None:
        super().__init__()
        if not np.isfinite(minimum_increment) or minimum_increment <= 0:
            raise ValueError("minimum_increment must be positive and finite")
        started = time.perf_counter()
        reference = MatrixFreeSymmetricTutteLayer(mesh)
        edges = reference.active_edges
        chosen = np.asarray(selected_edges, dtype=np.int64)
        if chosen.ndim != 1 or len(chosen) < 1 or len(np.unique(chosen)) != len(chosen):
            raise ValueError("selected_edges must be a nonempty unique vector")
        if np.any(chosen < 0) or np.any(chosen >= len(edges)):
            raise ValueError("selected edge index outside active edge range")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("mesh vertices must be planar coordinates")
        if np.min(vertices) < -1.0e-12 or np.max(vertices) > 1.0 + 1.0e-12:
            raise ValueError("mesh vertices must lie in the unit square")
        interior = reference.interior_vertices
        boundary = reference.boundary_vertices
        n = len(interior)
        global_to_interior = np.full(len(vertices), -1, dtype=np.int64)
        global_to_interior[interior] = np.arange(n)
        global_to_boundary = np.full(len(vertices), -1, dtype=np.int64)
        global_to_boundary[boundary] = np.arange(len(boundary))
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        rhs = np.zeros((n, 2), dtype=np.float64)
        for a, b in edges:
            ia, ib = global_to_interior[a], global_to_interior[b]
            if ia >= 0:
                rows.append(int(ia)); cols.append(int(ia)); values.append(1.0)
            if ib >= 0:
                rows.append(int(ib)); cols.append(int(ib)); values.append(1.0)
            if ia >= 0 and ib >= 0:
                rows.extend((int(ia), int(ib)))
                cols.extend((int(ib), int(ia)))
                values.extend((-1.0, -1.0))
            elif ia >= 0:
                rhs[ia] += vertices[b]
            elif ib >= 0:
                rhs[ib] += vertices[a]
        a0 = sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsc()
        factor = sparse_linalg.splu(a0)
        f0_interior = factor.solve(rhs)
        f0 = vertices.copy()
        f0[interior] = f0_interior
        k = len(chosen)
        u_rows: list[int] = []
        u_cols: list[int] = []
        u_values: list[float] = []
        d = np.empty((k, 2), dtype=np.float64)
        for r, edge_index in enumerate(chosen):
            a, b = edges[edge_index]
            ia, ib = global_to_interior[a], global_to_interior[b]
            if ia >= 0:
                u_rows.append(int(ia)); u_cols.append(r); u_values.append(1.0)
            if ib >= 0:
                u_rows.append(int(ib)); u_cols.append(r); u_values.append(-1.0)
            d[r] = f0[a] - f0[b]
        u = sparse.coo_matrix((u_values, (u_rows, u_cols)), shape=(n, k)).tocsc()
        w = factor.solve(u.toarray())
        gram = np.asarray(u.T @ w)
        self.register_buffer("_base_map", torch.as_tensor(f0), persistent=True)
        self.register_buffer("_w", torch.as_tensor(w), persistent=True)
        self.register_buffer("_gram", torch.as_tensor(0.5 * (gram + gram.T)), persistent=True)
        self.register_buffer("_d", torch.as_tensor(d), persistent=True)
        self.register_buffer("_interior", torch.as_tensor(interior, dtype=torch.int64), persistent=False)
        self.register_buffer("_faces", torch.as_tensor(np.array(mesh.faces, dtype=np.int64, copy=True)), persistent=False)
        self.register_buffer("_chosen", torch.as_tensor(chosen, dtype=torch.int64), persistent=False)
        self.n_vertices = len(vertices)
        self.n_interior = n
        self.n_active_edges = len(edges)
        self.rank = k
        self.minimum_increment = float(minimum_increment)
        self.validate_faces = bool(validate_faces)
        self.setup_seconds = time.perf_counter() - started
        self.factor_nnz = int(factor.L.nnz + factor.U.nnz)
        self.precomputed_bytes = int(f0.nbytes + w.nbytes + gram.nbytes + d.nbytes)
        self._a0 = a0
        self._u = u
        self._rhs0 = rhs
        self._base_full_numpy = f0
        self._interior_numpy = interior
        self._edges_numpy = edges

    @property
    def selected_edges(self) -> np.ndarray:
        return self._chosen.detach().cpu().numpy().copy()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        unbatched = latent.ndim == 1
        if unbatched:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2 or latent.shape[1] != self.rank:
            raise ValueError(f"latent must have shape (B,{self.rank})")
        if latent.dtype not in (torch.float32, torch.float64):
            raise ValueError("latent must be float32 or float64")
        if latent.device != self._w.device or latent.dtype != self._w.dtype:
            raise ValueError("move the layer to the input device and dtype before use")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("latent must be finite")
        delta = self.minimum_increment + F.softplus(latent)
        sqrt_delta = torch.sqrt(delta)
        small = torch.eye(self.rank, device=latent.device, dtype=latent.dtype)[None]
        small = small + sqrt_delta[:, :, None] * self._gram[None] * sqrt_delta[:, None, :]
        alpha = sqrt_delta[:, :, None] * torch.linalg.solve(small, sqrt_delta[:, :, None] * self._d[None])
        corrected = self._base_map[None].expand(latent.shape[0], -1, -1)
        interior_values = self._base_map[self._interior][None] - torch.matmul(self._w[None], alpha)
        result = corrected.index_copy(1, self._interior, interior_values)
        if self.validate_faces:
            triangles = result[:, self._faces]
            first = triangles[:, :, 1] - triangles[:, :, 0]
            second = triangles[:, :, 2] - triangles[:, :, 0]
            signed_double_area = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
            if not bool(torch.all(signed_double_area > 0.0)):
                raise RuntimeError("represented fine-grid map has a nonpositive face")
        return result[0] if unbatched else result

    def independent_sparse_solve(self, increments: np.ndarray) -> tuple[np.ndarray, float]:
        """Read-only oracle for small tests; not used by the neural forward."""
        values = np.asarray(increments, dtype=np.float64)
        if values.shape != (self.rank,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("increments must be a positive length-rank vector")
        update = self._u @ sparse.diags(values) @ self._u.T
        a = self._a0 + update
        full_difference = self._d.detach().cpu().numpy()
        rhs = self._rhs0 + self._u @ (values[:, None] * (self._u.T @ self._base_full_numpy[self._interior_numpy] - full_difference))
        # Since d = U^T f0_I + boundary contribution, the changed RHS is
        # U D (U^T f0_I - d). This includes interior-boundary selected edges.
        interior_solution = sparse_linalg.spsolve(a, rhs)
        full = self._base_full_numpy.copy()
        full[self._interior_numpy] = interior_solution
        residual = float(np.linalg.norm(a @ interior_solution - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny))
        return full, residual
