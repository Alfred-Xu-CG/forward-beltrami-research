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
