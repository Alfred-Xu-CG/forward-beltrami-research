"""Solve-free monotone triangular maps on a fixed rectangular triangulation.

For ``axis='vertical'``, vertices (i,j) map to (X_i, V_ij), where X is
strictly increasing in i and V is strictly increasing in j for each i.  Both
triangles of each axis-aligned cell then have positive signed area.  The four
sides map monotonically onto the four sides of the unit square, so the P1
extension is a homeomorphism.  ``axis='horizontal'`` swaps x and y.
"""

from __future__ import annotations

import torch


class DenseMonotoneGridLayer(torch.nn.Module):
    """Decode positive increments directly into a fine-grid P1 homeomorphism.

    Inputs have shapes ``(side-1,)`` and ``(side, side-1)`` or the same shapes
    with a common leading batch dimension.  For the vertical layer the first
    tensor controls horizontal column spacing and the second controls vertical
    spacing in each column.  For the horizontal layer the first controls row
    spacing and the second horizontal spacing in each row.  Output is row-major
    ``(side,side,2)`` or ``(batch,side,side,2)``.

    ``floor_fraction`` reserves this fraction of every normalized spacing for
    the uniform distribution.  It keeps represented float32 spacings distinct
    at the intended dense resolutions even for extreme finite logits.
    """

    def __init__(
        self,
        side: int,
        *,
        axis: str = "vertical",
        floor_fraction: float = 0.01,
    ) -> None:
        super().__init__()
        if side < 2:
            raise ValueError("side must be at least two")
        if axis not in ("vertical", "horizontal"):
            raise ValueError("axis must be vertical or horizontal")
        if not 0.0 < floor_fraction < 1.0:
            raise ValueError("floor_fraction must be strictly between zero and one")
        self.side = int(side)
        self.axis = axis
        self.floor_fraction = float(floor_fraction)

    def _coordinates(self, logits: torch.Tensor) -> torch.Tensor:
        intervals = self.side - 1
        spacing = (1.0 - self.floor_fraction) * torch.softmax(logits, dim=-1)
        spacing = spacing + self.floor_fraction / intervals
        origin = torch.zeros_like(spacing[..., :1])
        interior = torch.cumsum(spacing, dim=-1)[..., :-1]
        endpoint = torch.ones_like(origin)
        return torch.cat((origin, interior, endpoint), dim=-1)

    def forward(
        self,
        global_logits: torch.Tensor,
        line_logits: torch.Tensor,
    ) -> torch.Tensor:
        unbatched = global_logits.ndim == 1 and line_logits.ndim == 2
        if unbatched:
            global_logits = global_logits.unsqueeze(0)
            line_logits = line_logits.unsqueeze(0)
        if (
            global_logits.ndim != 2
            or line_logits.ndim != 3
            or global_logits.shape != (line_logits.shape[0], self.side - 1)
            or line_logits.shape != (line_logits.shape[0], self.side, self.side - 1)
        ):
            raise ValueError("logits must have shapes (B,N-1) and (B,N,N-1)")
        if global_logits.device != line_logits.device or global_logits.dtype != line_logits.dtype:
            raise ValueError("logits must share device and dtype")
        if global_logits.dtype not in (torch.float32, torch.float64):
            raise ValueError("logits must be float32 or float64")
        if not bool(torch.isfinite(global_logits).all() & torch.isfinite(line_logits).all()):
            raise ValueError("logits must be finite")

        global_coord = self._coordinates(global_logits)
        line_coord = self._coordinates(line_logits)
        batch = global_logits.shape[0]
        if self.axis == "vertical":
            x_coord = global_coord[:, None, :].expand(batch, self.side, self.side)
            y_coord = line_coord.transpose(1, 2)
        else:
            x_coord = line_coord
            y_coord = global_coord[:, :, None].expand(batch, self.side, self.side)
        if not bool(torch.all(global_coord[..., 1:] > global_coord[..., :-1])):
            raise RuntimeError("global spacings collapsed in represented precision")
        if not bool(torch.all(line_coord[..., 1:] > line_coord[..., :-1])):
            raise RuntimeError("line spacings collapsed in represented precision")
        control = torch.stack((x_coord, y_coord), dim=-1)
        return control[0] if unbatched else control
