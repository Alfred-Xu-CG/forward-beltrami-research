"""Conservative local radial step that preserves a facewise Beltrami cap."""

from __future__ import annotations

import math

import numpy as np
import torch


def _complex_derivative_norms(jacobian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_re = 0.5 * (jacobian[..., 0, 0] + jacobian[..., 1, 1])
    alpha_im = 0.5 * (jacobian[..., 1, 0] - jacobian[..., 0, 1])
    beta_re = 0.5 * (jacobian[..., 0, 0] - jacobian[..., 1, 1])
    beta_im = 0.5 * (jacobian[..., 1, 0] + jacobian[..., 0, 1])
    # vector_norm specifies a zero subgradient at the origin.  A literal
    # sqrt(x*x+y*y) gives 0/0 in backward when a proposed vertex step is zero.
    alpha = torch.linalg.vector_norm(torch.stack((alpha_re, alpha_im), dim=-1), dim=-1)
    beta = torch.linalg.vector_norm(torch.stack((beta_re, beta_im), dim=-1), dim=-1)
    return alpha, beta


class SafeColoredQCRadialRelaxation(torch.nn.Module):
    """Update interior vertices without exceeding a pre-existing QC cap.

    Contract: the input P1 map has fixed injective boundary and each source
    face satisfies |mu| < qc_cap < 1. Under exact arithmetic the output has
    the same property. An input that violates the cap cannot be repaired by
    this preservation layer; call sites must verify its precondition.
    """

    def __init__(
        self, side: int, *, qc_cap: float = 0.8,
        safety_fraction: float = 0.9, raw_span: float = 2.0,
        floor_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        if (side < 3 or not 0 < qc_cap < 1 or not 0 < safety_fraction < 1
                or raw_span <= 0 or not 0 <= floor_fraction < 1):
            raise ValueError("invalid QC radial configuration")
        self.side = side
        self.qc_cap = float(qc_cap)
        self.safety_fraction = float(safety_fraction)
        self.raw_span = float(raw_span)
        self.floor_fraction = float(floor_fraction)
        opposite_offsets = np.asarray([
            (-side - 1, -side), (-1, -side - 1), (-side, 1),
            (side, -1), (1, side + 1), (side + 1, side),
        ], dtype=np.int64)
        h = 1.0 / (side - 1)
        inverse = []
        for first, second in opposite_offsets:
            def xy(offset: int) -> tuple[float, float]:
                row, column = divmod(side + 1 + int(offset), side)
                return (h * (column - 1), h * (row - 1))
            start_x, start_y = xy(first)
            end_x, end_y = xy(second)
            source = np.asarray([
                [end_x - start_x, -start_x],
                [end_y - start_y, -start_y],
            ], dtype=np.float64)
            inverse.append(np.linalg.inv(source))
        self.register_buffer("_source_inverse", torch.tensor(np.stack(inverse), dtype=torch.float64),
                             persistent=False)
        for color in range(4):
            parity_row, parity_column = divmod(color, 2)
            vertices = np.asarray([
                row * side + column
                for row in range(1, side - 1) for column in range(1, side - 1)
                if row % 2 == parity_row and column % 2 == parity_column
            ], dtype=np.int64)
            edges = vertices[:, None, None] + opposite_offsets[None]
            local = np.asarray([
                (vertex // side - 1) * (side - 2) + (vertex % side - 1) for vertex in vertices
            ], dtype=np.int64)
            self.register_buffer(f"_vertices_{color}", torch.from_numpy(vertices), persistent=False)
            self.register_buffer(f"_opposite_{color}", torch.from_numpy(edges), persistent=False)
            self.register_buffer(f"_local_{color}", torch.from_numpy(local), persistent=False)

    def compute_area_floor(self, base: torch.Tensor) -> torch.Tensor:
        if not self.floor_fraction:
            raise ValueError("floor_fraction must be positive")
        a, b = base[:, :-1, :-1], base[:, :-1, 1:]
        c, d = base[:, 1:, 1:], base[:, 1:, :-1]
        lower = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
            b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        upper = (c[..., 0] - a[..., 0]) * (d[..., 1] - a[..., 1]) - (
            c[..., 1] - a[..., 1]) * (d[..., 0] - a[..., 0])
        return self.floor_fraction * torch.minimum(
            lower.amin(dim=(1, 2)), upper.amin(dim=(1, 2))
        )

    def forward(
        self, base: torch.Tensor, logits: torch.Tensor, *,
        area_floor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base.ndim != 4 or base.shape[1:] != (self.side, self.side, 2):
            raise ValueError("base must be (batch,side,side,2)")
        if logits.shape != (base.shape[0], self.side - 2, self.side - 2, 2):
            raise ValueError("logits must be interior two-vectors")
        if logits.device != base.device or logits.dtype != base.dtype:
            raise ValueError("base/logits device and dtype must match")
        batch = base.shape[0]
        if area_floor is None and self.floor_fraction:
            area_floor = self.compute_area_floor(base)
        if area_floor is not None and (
            area_floor.shape != (batch,) or area_floor.device != base.device
            or area_floor.dtype != base.dtype
        ):
            raise ValueError("area_floor must be a batch vector matching base")
        current = base.reshape(batch, self.side * self.side, 2)
        latent = logits.reshape(batch, -1, 2)
        inverse = self._source_inverse.to(device=base.device, dtype=base.dtype)
        machine_guard = math.sqrt(torch.finfo(base.dtype).eps)
        for color in range(4):
            vertices = getattr(self, f"_vertices_{color}")
            opposite = getattr(self, f"_opposite_{color}")
            local = getattr(self, f"_local_{color}")
            point = current[:, vertices]
            start = current[:, opposite[..., 0]]
            end = current[:, opposite[..., 1]]
            edge = end - start
            relative = point[:, :, None] - start
            raw = (self.raw_span / (self.side - 1)) * torch.tanh(latent[:, local])
            columns = torch.stack((edge, relative), dim=-1)
            jacobian = torch.matmul(columns, inverse)
            gradient_hat = inverse[:, 1, :]
            derivative = raw[:, :, None, :, None] * gradient_hat[None, None, :, None, :]
            alpha, beta = _complex_derivative_norms(jacobian)
            dalpha, dbeta = _complex_derivative_norms(derivative)
            margin = (self.qc_cap * alpha - beta).clamp_min(0)
            slope = self.qc_cap * dalpha + dbeta
            ratio = self.safety_fraction * margin / slope.clamp_min(machine_guard)
            scale = ratio.amin(dim=-1).clamp(max=1.0)
            if area_floor is not None:
                signed_area = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
                adverse = (-(edge[..., 0] * raw[:, :, None, 1]
                             - edge[..., 1] * raw[:, :, None, 0])).clamp_min(0)
                allowable = torch.minimum(
                    self.safety_fraction * signed_area,
                    (signed_area - area_floor[:, None, None]).clamp_min(0),
                )
                guard = machine_guard / (self.side - 1) ** 2
                denominator = torch.maximum(
                    torch.maximum(adverse, allowable), adverse.new_tensor(guard)
                )
                area_ratio = torch.where(adverse > 0, allowable / denominator,
                                         torch.ones_like(adverse))
                scale = torch.minimum(scale, area_ratio.amin(dim=-1))
            current = current.index_copy(1, vertices, point + scale[..., None] * raw)
        return current.reshape(batch, self.side, self.side, 2)
