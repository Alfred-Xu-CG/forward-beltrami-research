"""Exact composition of full-resolution vertical/horizontal monotone maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..tutte.composition import compose_control_maps
from ..tutte.dense_warp import StructuredDenseQueryTable
from .monotone import MultiscaleMonotoneGridLayer


@dataclass(frozen=True)
class AlternatingMonotoneResult:
    controls: tuple[torch.Tensor, ...]
    dense: torch.Tensor


class ExactAlternatingMonotoneComposition(torch.nn.Module):
    """Decode every layer on one fine grid, then compose exact PL functions.

    Each layer latent is a sequence of coarse-to-fine ``(global,line)`` pairs.
    The returned ``dense`` samples the exact composition at fixed queries;
    its PL partition is generally a refinement of the original control grid.
    """

    def __init__(self, side: int, query_table: StructuredDenseQueryTable, axes: Sequence[str] = ("vertical", "horizontal")) -> None:
        super().__init__()
        if not axes or any(axis not in ("vertical", "horizontal") for axis in axes):
            raise ValueError("axes must be a nonempty vertical/horizontal sequence")
        if query_table.control_vertices != side**2:
            raise ValueError("query table does not match the control grid")
        self.layers = torch.nn.ModuleList(MultiscaleMonotoneGridLayer(side, axis=axis) for axis in axes)
        self.query_table = query_table

    def forward(self, latents: Sequence[tuple[tuple[torch.Tensor, torch.Tensor], ...]]) -> AlternatingMonotoneResult:
        if len(latents) != len(self.layers):
            raise ValueError("one multiscale latent tuple is needed per layer")
        controls = tuple(layer(levels) for layer, levels in zip(self.layers, latents))
        flattened = [control.reshape(*control.shape[:-3], -1, 2) for control in controls]
        dense = compose_control_maps([self.query_table] * len(controls), flattened)
        return AlternatingMonotoneResult(controls=controls, dense=dense)


def evaluate_structured_p1_with_jacobian(
    control: torch.Tensor, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a batched square-grid P1 map and its exact affine Jacobian.

    Jacobians are defined almost everywhere. On triangle boundaries the lower
    triangle is selected consistently; this is a branch derivative, not a
    claim of classical differentiability across those boundaries.
    """
    if control.ndim != 4 or control.shape[1] != control.shape[2] or control.shape[-1] != 2:
        raise ValueError("control must have shape (B,N,N,2)")
    if points.ndim != 3 or points.shape[0] != control.shape[0] or points.shape[-1] != 2:
        raise ValueError("points must have shape (B,Q,2)")
    batch, side, _, _ = control.shape
    x = points[..., 0].clamp(0.0, 1.0)
    y = points[..., 1].clamp(0.0, 1.0)
    sx = x * (side - 1)
    sy = y * (side - 1)
    ix = torch.floor(sx).to(torch.int64).clamp(0, side - 2)
    iy = torch.floor(sy).to(torch.int64).clamp(0, side - 2)
    s = sx - ix
    t = sy - iy
    flattened = control.reshape(batch, side**2, 2)

    def gather(index: torch.Tensor) -> torch.Tensor:
        return torch.gather(flattened, 1, index[..., None].expand(-1, -1, 2))

    index00 = iy * side + ix
    f00 = gather(index00)
    f10 = gather(index00 + 1)
    f01 = gather(index00 + side)
    f11 = gather(index00 + side + 1)
    lower = t <= s
    lower_dx = f10 - f00
    lower_dy = f11 - f10
    upper_dx = f11 - f01
    upper_dy = f01 - f00
    dx = torch.where(lower[..., None], lower_dx, upper_dx) * (side - 1)
    dy = torch.where(lower[..., None], lower_dy, upper_dy) * (side - 1)
    value = f00 + s[..., None] * torch.where(lower[..., None], lower_dx, upper_dx) + t[..., None] * torch.where(lower[..., None], lower_dy, upper_dy)
    jacobian = torch.stack((dx, dy), dim=-1)
    return value, jacobian
