"""Precomputed barycentric queries for a fixed structured control mesh."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ...mesh import TriMesh


@dataclass(frozen=True)
class StructuredDenseQueryTable:
    """Map dense unit-square queries to fixed control triangles once.

    The table matches :func:`qcopt.mesh.structured_rectangle`, whose cells use
    the diagonal from the lower-left to the upper-right corner. Interpolation
    contains only tensor gather, multiply, and reduce operations; point
    location is never repeated by :meth:`interpolate`.  The separate
    :meth:`interpolate_points` method intentionally repeats structured-grid
    arithmetic for queries propagated through multiple composed maps.
    """

    _vertex_indices_cpu: torch.Tensor
    _barycentric_cpu: torch.Tensor
    height: int
    width: int
    control_vertices: int
    nx: int
    ny: int
    _device_cache: dict[tuple[str, int | None, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    @classmethod
    def from_mesh(
        cls,
        mesh: TriMesh,
        *,
        height: int,
        width: int,
    ) -> "StructuredDenseQueryTable":
        if height < 2 or width < 2:
            raise ValueError("height and width must both be at least two")
        x_values = np.unique(mesh.vertices[:, 0])
        y_values = np.unique(mesh.vertices[:, 1])
        nx = len(x_values) - 1
        ny = len(y_values) - 1
        if nx < 1 or ny < 1 or mesh.n_vertices != (nx + 1) * (ny + 1):
            raise ValueError("mesh is not a complete structured rectangle")
        expected_x = np.linspace(0.0, 1.0, nx + 1)
        expected_y = np.linspace(0.0, 1.0, ny + 1)
        xx, yy = np.meshgrid(expected_x, expected_y, indexing="xy")
        expected_vertices = np.column_stack((xx.ravel(), yy.ravel()))
        if not np.allclose(mesh.vertices, expected_vertices, atol=1e-14, rtol=0.0):
            raise ValueError("mesh vertices do not match structured_rectangle ordering")

        expected_faces = []
        stride = nx + 1
        for row in range(ny):
            for col in range(nx):
                v00 = row * stride + col
                v10 = v00 + 1
                v01 = v00 + stride
                v11 = v01 + 1
                expected_faces.extend(((v00, v10, v11), (v00, v11, v01)))
        if not np.array_equal(mesh.faces, np.asarray(expected_faces, dtype=np.int64)):
            raise ValueError("mesh faces do not match structured_rectangle diagonals")

        query_x = np.linspace(0.0, 1.0, width)
        query_y = np.linspace(0.0, 1.0, height)
        dense_x, dense_y = np.meshgrid(query_x, query_y, indexing="xy")
        scaled_x = dense_x.ravel() * nx
        scaled_y = dense_y.ravel() * ny
        cell_x = np.minimum(np.floor(scaled_x).astype(np.int64), nx - 1)
        cell_y = np.minimum(np.floor(scaled_y).astype(np.int64), ny - 1)
        local_x = scaled_x - cell_x
        local_y = scaled_y - cell_y

        v00 = cell_y * stride + cell_x
        v10 = v00 + 1
        v01 = v00 + stride
        v11 = v01 + 1
        lower = local_y <= local_x
        vertex_indices = np.empty((height * width, 3), dtype=np.int64)
        barycentric = np.empty((height * width, 3), dtype=np.float64)
        vertex_indices[lower] = np.column_stack((v00[lower], v10[lower], v11[lower]))
        barycentric[lower] = np.column_stack(
            (1.0 - local_x[lower], local_x[lower] - local_y[lower], local_y[lower])
        )
        upper = ~lower
        vertex_indices[upper] = np.column_stack((v00[upper], v11[upper], v01[upper]))
        barycentric[upper] = np.column_stack(
            (1.0 - local_y[upper], local_x[upper], local_y[upper] - local_x[upper])
        )
        index_tensor = torch.from_numpy(vertex_indices.copy())
        barycentric_tensor = torch.from_numpy(barycentric.copy())
        result = cls(
            index_tensor,
            barycentric_tensor,
            height,
            width,
            mesh.n_vertices,
            nx,
            ny,
        )
        result._device_cache[("cpu", None, torch.float64)] = (
            index_tensor,
            barycentric_tensor,
        )
        return result

    @property
    def query_count(self) -> int:
        return self.height * self.width

    @property
    def barycentric_numpy(self) -> np.ndarray:
        result = self._barycentric_cpu.numpy().copy()
        result.setflags(write=False)
        return result

    def prepare(self, *, device: torch.device | str, dtype: torch.dtype) -> None:
        """Move the immutable query table outside the timed forward path."""
        device = torch.device(device)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dense query dtype must be torch.float32 or torch.float64")
        prepared_indices = self._vertex_indices_cpu.to(device=device)
        prepared_weights = self._barycentric_cpu.to(device=device, dtype=dtype)
        actual_device = prepared_weights.device
        key = (actual_device.type, actual_device.index, dtype)
        self._device_cache.setdefault(key, (prepared_indices, prepared_weights))

    def _tensors_for(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (reference.device.type, reference.device.index, reference.dtype)
        cached = self._device_cache.get(key)
        if cached is None:
            raise ValueError(
                "query table is not prepared for this device/dtype; call prepare before interpolate"
            )
        return cached

    def interpolate(self, control_vertices: torch.Tensor) -> torch.Tensor:
        """Interpolate one ``(V,2)`` or batched ``(B,V,2)`` control map."""
        if control_vertices.ndim not in (2, 3) or control_vertices.shape[-1] != 2:
            raise ValueError("control_vertices must have shape (V,2) or (B,V,2)")
        if control_vertices.shape[-2] != self.control_vertices:
            raise ValueError("control vertex count does not match the query table")
        if not torch.is_floating_point(control_vertices):
            raise ValueError("control_vertices must be floating point")
        indices, weights = self._tensors_for(control_vertices)
        if control_vertices.ndim == 2:
            values = (control_vertices[indices] * weights[..., None]).sum(dim=1)
            return values.reshape(self.height, self.width, 2)
        values = (control_vertices[:, indices, :] * weights[None, ..., None]).sum(dim=2)
        return values.reshape(control_vertices.shape[0], self.height, self.width, 2)

    def interpolate_points(
        self,
        control_vertices: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the P1 map at dynamic unit-square points.

        Unlike :meth:`interpolate`, this performs structured-grid point
        location on every call.  It is intended for composition layers whose
        queries are outputs of an earlier map.  Shapes are ``(V,2)`` or
        ``(B,V,2)`` and ``(Q,2)`` or ``(B,Q,2)``; singleton batches broadcast.
        The floor/triangle decision is piecewise constant, while barycentric
        weights remain differentiable inside each source triangle.
        """
        if control_vertices.ndim not in (2, 3) or control_vertices.shape[-2:] != (
            self.control_vertices,
            2,
        ):
            raise ValueError("control_vertices must have shape (V,2) or (B,V,2)")
        if query_points.ndim not in (2, 3) or query_points.shape[-1] != 2:
            raise ValueError("query_points must have shape (Q,2) or (B,Q,2)")
        if control_vertices.dtype not in (torch.float32, torch.float64) or query_points.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("control_vertices and query_points must be float32 or float64")
        if (
            control_vertices.dtype != query_points.dtype
            or control_vertices.device != query_points.device
        ):
            raise ValueError("control_vertices and query_points must share dtype and device")
        if not bool(torch.isfinite(control_vertices).all()) or not bool(
            torch.isfinite(query_points).all()
        ):
            raise ValueError("control_vertices and query_points must be finite")
        # Admit only dtype-roundoff-sized excursions before clamping to the
        # closed source square.  Larger excursions are a domain error, not a
        # padding convention.  The clamped coordinate has zero outward VJP.
        tolerance = 64.0 * torch.finfo(query_points.dtype).eps
        if bool(torch.any(query_points < -tolerance)) or bool(
            torch.any(query_points > 1.0 + tolerance)
        ):
            raise ValueError("dynamic query points must lie in the closed unit square")

        unbatched = control_vertices.ndim == query_points.ndim == 2
        control = control_vertices.unsqueeze(0) if control_vertices.ndim == 2 else control_vertices
        query = query_points.unsqueeze(0) if query_points.ndim == 2 else query_points
        batch = max(control.shape[0], query.shape[0])
        if control.shape[0] not in (1, batch) or query.shape[0] not in (1, batch):
            raise ValueError("control and query batch dimensions must match or be singleton")
        control = control.expand(batch, -1, -1)
        query = query.expand(batch, -1, -1).clamp(0.0, 1.0)

        scaled_x = query[..., 0] * self.nx
        scaled_y = query[..., 1] * self.ny
        cell_x = torch.floor(scaled_x).to(torch.int64).clamp(max=self.nx - 1)
        cell_y = torch.floor(scaled_y).to(torch.int64).clamp(max=self.ny - 1)
        local_x = scaled_x - cell_x
        local_y = scaled_y - cell_y
        stride = self.nx + 1
        v00 = cell_y * stride + cell_x
        v10 = v00 + 1
        v01 = v00 + stride
        v11 = v01 + 1
        lower = local_y <= local_x
        lower_indices = torch.stack((v00, v10, v11), dim=-1)
        upper_indices = torch.stack((v00, v11, v01), dim=-1)
        indices = torch.where(lower[..., None], lower_indices, upper_indices)
        lower_weights = torch.stack(
            (1.0 - local_x, local_x - local_y, local_y), dim=-1
        )
        upper_weights = torch.stack(
            (1.0 - local_y, local_x, local_y - local_x), dim=-1
        )
        weights = torch.where(lower[..., None], lower_weights, upper_weights)
        batch_indices = torch.arange(batch, device=control.device)[:, None, None]
        values = control[batch_indices, indices]
        result = (values * weights[..., None]).sum(dim=-2)
        return result[0] if unbatched else result
