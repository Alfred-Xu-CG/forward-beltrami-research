"""Exact coarse-to-fine composition of convex-cell P1 homeomorphisms."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...mesh import structured_rectangle
from ..tutte.composition import compose_control_maps
from ..tutte.dense_warp import StructuredDenseQueryTable
from .convex_quad import HierarchicalConvexQuadFreeCenterLayer


@dataclass(frozen=True)
class CoarseFineConvexQuadResult:
    controls: tuple[torch.Tensor, torch.Tensor]
    dense: torch.Tensor


class CoarseFineConvexQuadComposition(torch.nn.Module):
    """Two certified factors; exact PL composition need not be original-grid P1."""

    def __init__(self, coarse_side: int, fine_side: int, query_side: int) -> None:
        super().__init__()
        if not 3 <= coarse_side <= fine_side or query_side < 2:
            raise ValueError("require 3 <= first-side <= second-side and query-side >= 2")
        self.coarse = HierarchicalConvexQuadFreeCenterLayer(coarse_side)
        self.fine = HierarchicalConvexQuadFreeCenterLayer(fine_side)
        self.coarse_table = StructuredDenseQueryTable.from_mesh(
            structured_rectangle(coarse_side - 1, coarse_side - 1), height=query_side, width=query_side
        )
        self.fine_table = StructuredDenseQueryTable.from_mesh(
            structured_rectangle(fine_side - 1, fine_side - 1), height=query_side, width=query_side
        )

    def prepare(self, *, device: torch.device | str, dtype: torch.dtype) -> None:
        self.coarse_table.prepare(device=device, dtype=dtype)
        self.fine_table.prepare(device=device, dtype=dtype)

    def forward(
        self,
        coarse_root: torch.Tensor,
        coarse_latents: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        fine_root: torch.Tensor,
        fine_latents: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    ) -> CoarseFineConvexQuadResult:
        coarse = self.coarse(coarse_root, coarse_latents)
        fine = self.fine(fine_root, fine_latents)
        dense = compose_control_maps(
            (self.coarse_table, self.fine_table),
            (coarse.reshape(coarse.shape[0], -1, 2), fine.reshape(fine.shape[0], -1, 2)),
        )
        return CoarseFineConvexQuadResult((coarse, fine), dense)
