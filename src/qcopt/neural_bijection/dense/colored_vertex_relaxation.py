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
        floor_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        if side < 3 or not 0.0 < safety_fraction < 1.0:
            raise ValueError("side >= 3 and 0 < safety_fraction < 1 are required")
        if motion_mode not in ("disk", "radial") or not math.isfinite(raw_span) or raw_span <= 0:
            raise ValueError("motion_mode must be disk/radial and raw_span must be positive and finite")
        if not 0.0 <= floor_fraction < 1.0 or (floor_fraction and motion_mode != "radial"):
            raise ValueError("floor_fraction must be in [0,1) and requires radial mode")
        self.side = side
        self.safety_fraction = float(safety_fraction)
        self.motion_mode = motion_mode
        self.raw_span = float(raw_span)
        self.floor_fraction = float(floor_fraction)
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

    def compute_area_floor(self, base: torch.Tensor) -> torch.Tensor:
        """A batch-wise absolute double-area floor tied to the initial base."""
        if not self.floor_fraction:
            raise ValueError("compute_area_floor requires positive floor_fraction")
        a = base[:, :-1, :-1]
        b = base[:, :-1, 1:]
        c = base[:, 1:, 1:]
        d = base[:, 1:, :-1]
        def cross(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        lower = cross(b - a, c - a).amin(dim=(1, 2))
        upper = cross(c - a, d - a).amin(dim=(1, 2))
        return self.floor_fraction * torch.minimum(lower, upper)

    def forward(
        self, base: torch.Tensor, logits: torch.Tensor, *,
        area_floor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base.ndim != 4 or base.shape[1:] != (self.side, self.side, 2):
            raise ValueError("base must have shape (batch,side,side,2)")
        if logits.shape != (base.shape[0], self.side - 2, self.side - 2, 2):
            raise ValueError("logits must have one two-vector per strict interior vertex")
        if logits.device != base.device or logits.dtype != base.dtype:
            raise ValueError("base and logits must have the same device and dtype")
        batch = base.shape[0]
        if area_floor is None and self.floor_fraction:
            area_floor = self.compute_area_floor(base)
        if area_floor is not None and (
            area_floor.shape != (batch,) or area_floor.device != base.device or area_floor.dtype != base.dtype
        ):
            raise ValueError("area_floor must have shape (batch,) and match base device/dtype")
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
                allowable_loss = self.safety_fraction * signed_double_area
                if area_floor is not None:
                    allowable_loss = torch.minimum(
                        allowable_loss, (signed_double_area - area_floor[:, None, None]).clamp_min(0.0)
                    )
                # We only need min(1, allowable_loss / adverse).  Computing
                # the raw quotient with a machine-tiny denominator can make
                # an inactive branch overflow and poison a later VJP.
                guard = math.sqrt(torch.finfo(base.dtype).eps) / (self.side - 1) ** 2
                denominator = torch.maximum(
                    torch.maximum(adverse, allowable_loss),
                    adverse.new_tensor(guard),
                )
                face_scale = torch.where(adverse > 0, allowable_loss / denominator, torch.ones_like(adverse))
                scale = face_scale.amin(dim=-1)
                displacement = scale[..., None] * raw
            current = current.index_copy(1, vertices, point + displacement)
        return current.reshape(batch, self.side, self.side, 2)


class HierarchicalConvexQuadLocalLayer(torch.nn.Module):
    """One original-grid P1 layer with multiscale and fine local latents."""

    def __init__(
        self, side: int, *, safety_fraction: float = 0.85,
        motion_mode: str = "disk", raw_span: float = 2.0,
        floor_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = HierarchicalConvexQuadFreeCenterLayer(side)
        self.local = SafeColoredVertexRelaxation(
            side, safety_fraction=safety_fraction, motion_mode=motion_mode,
            raw_span=raw_span, floor_fraction=floor_fraction,
        )

    def forward(self, root: torch.Tensor, levels, local_logits: torch.Tensor) -> torch.Tensor:
        return self.local(self.base(root, levels), local_logits)
