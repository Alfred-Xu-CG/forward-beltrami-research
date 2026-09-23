"""Solve-free latent vertex motions with a strict original-grid P1 orientation margin."""

from __future__ import annotations

import math

import numpy as np
import torch

from .convex_quad import HierarchicalConvexQuadFreeCenterLayer


class SafeColoredVertexRelaxation(torch.nn.Module):
    """Move four independent vertex colors inside their incident-face safe disks.

    The input must already be an oriented P1 homeomorphism with a fixed,
    injective boundary. For finite logits and ``0 < safety_fraction < 1``,
    each color pass preserves all incident triangle orientations; the output
    stays P1 on the *same* triangulation, without a linear system or a search.
    """

    def __init__(
        self, side: int, *, safety_fraction: float = 0.75,
        motion_mode: str = "disk", raw_span: float = 2.0,
    ) -> None:
        super().__init__()
        if side < 3 or not 0.0 < safety_fraction < 1.0:
            raise ValueError("side >= 3 and 0 < safety_fraction < 1 are required")
        if motion_mode not in ("disk", "radial") or not math.isfinite(raw_span) or raw_span <= 0:
            raise ValueError("motion_mode must be disk/radial and raw_span must be positive and finite")
        self.side = side
        self.safety_fraction = float(safety_fraction)
        self.motion_mode = motion_mode
        self.raw_span = float(raw_span)
        # On the fixed southwest-to-northeast triangulation, these are the
        # cyclic opposite edges of the six incident triangles.  Generating
        # them arithmetically avoids millions of Python tuples at 1025².
        opposite_offsets = np.asarray([
            (-side - 1, -side), (-1, -side - 1), (-side, 1),
            (side, -1), (1, side + 1), (side + 1, side),
        ], dtype=np.int64)
        for color in range(4):
            parity_row, parity_column = divmod(color, 2)
            vertices = [
                row * side + column
                for row in range(1, side - 1) for column in range(1, side - 1)
                if row % 2 == parity_row and column % 2 == parity_column
            ]
            edges = np.asarray(vertices, dtype=np.int64)[:, None, None] + opposite_offsets[None]
            if edges.shape != (len(vertices), 6, 2):
                raise AssertionError("every strict interior structured vertex must have six incident triangles")
            local = [(vertex // side - 1) * (side - 2) + (vertex % side - 1) for vertex in vertices]
            self.register_buffer(f"_vertices_{color}", torch.as_tensor(vertices, dtype=torch.long), persistent=False)
            self.register_buffer(f"_opposite_{color}", torch.as_tensor(edges, dtype=torch.long), persistent=False)
            self.register_buffer(f"_local_{color}", torch.as_tensor(local, dtype=torch.long), persistent=False)

    def forward(self, base: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if base.ndim != 4 or base.shape[1:] != (self.side, self.side, 2):
            raise ValueError("base must have shape (batch,side,side,2)")
        if logits.shape != (base.shape[0], self.side - 2, self.side - 2, 2):
            raise ValueError("logits must have one two-vector per strict interior vertex")
        if logits.device != base.device or logits.dtype != base.dtype:
            raise ValueError("base and logits must have the same device and dtype")
        batch = base.shape[0]
        current = base.reshape(batch, self.side * self.side, 2)
        latent = logits.reshape(batch, -1, 2)
        for color in range(4):
            vertices = getattr(self, f"_vertices_{color}")
            opposite = getattr(self, f"_opposite_{color}")
            local = getattr(self, f"_local_{color}")
            point = current[:, vertices]
            start = current[:, opposite[..., 0]]
            end = current[:, opposite[..., 1]]
            edge = end - start
            relative = point[:, :, None] - start
            signed_double_area = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
            if self.motion_mode == "disk":
                altitude = signed_double_area / torch.linalg.vector_norm(edge, dim=-1)
                radius = altitude.amin(dim=-1)
                displacement = (self.safety_fraction / math.sqrt(2.0)) * radius[..., None] * torch.tanh(latent[:, local])
            else:
                raw = (self.raw_span / (self.side - 1)) * torch.tanh(latent[:, local])
                adverse = -(edge[..., 0] * raw[:, :, None, 1] - edge[..., 1] * raw[:, :, None, 0])
                adverse = adverse.clamp_min(0.0)
                radial_limit = (signed_double_area / adverse.clamp_min(torch.finfo(base.dtype).tiny)).amin(dim=-1)
                scale = (self.safety_fraction * radial_limit).clamp(max=1.0)
                displacement = scale[..., None] * raw
            current = current.index_copy(1, vertices, point + displacement)
        return current.reshape(batch, self.side, self.side, 2)


class HierarchicalConvexQuadLocalLayer(torch.nn.Module):
    """One original-grid P1 layer with multiscale and fine local latents."""

    def __init__(
        self, side: int, *, safety_fraction: float = 0.85,
        motion_mode: str = "disk", raw_span: float = 2.0,
    ) -> None:
        super().__init__()
        self.base = HierarchicalConvexQuadFreeCenterLayer(side)
        self.local = SafeColoredVertexRelaxation(
            side, safety_fraction=safety_fraction, motion_mode=motion_mode, raw_span=raw_span
        )

    def forward(self, root: torch.Tensor, levels, local_logits: torch.Tensor) -> torch.Tensor:
        return self.local(self.base(root, levels), local_logits)
