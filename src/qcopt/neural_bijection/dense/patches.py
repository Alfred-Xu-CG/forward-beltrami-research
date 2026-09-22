"""Parallel local PL maps with identity patch boundaries and exact composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..tutte.composition import compose_control_maps
from ..tutte.dense_warp import StructuredDenseQueryTable


class LocalPatchMonotoneLayer(torch.nn.Module):
    """Independent local vertical or horizontal monotone patches.

``latent`` has shape ``(B,P,p-1,p)`` where ``P`` is the number of patches and
``p`` their cell side.  For a vertical layer, each of the ``p-1`` interior
columns has ``p`` positive y increments; horizontal layers use interior rows
and x increments.  Every patch boundary is pointwise fixed.  Patches at the
selected offset are disjoint, and regions outside them remain identity.
    """

    def __init__(
        self,
        side: int,
        patch_cells: int,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
        axis: str = "vertical",
        floor_fraction: float = 0.01,
    ) -> None:
        super().__init__()
        if side < 3 or patch_cells < 2 or patch_cells >= side:
            raise ValueError("side/patch_cells must provide at least one interior patch vertex")
        if axis not in ("vertical", "horizontal"):
            raise ValueError("axis must be vertical or horizontal")
        if not 0 <= offset_x < patch_cells or not 0 <= offset_y < patch_cells:
            raise ValueError("offsets must be within one patch side")
        if not 0.0 < floor_fraction < 1.0:
            raise ValueError("floor_fraction must be strictly between zero and one")
        starts_x = range(offset_x, side - patch_cells, patch_cells)
        starts_y = range(offset_y, side - patch_cells, patch_cells)
        origins: list[tuple[int, int]] = []
        indices: list[int] = []
        for y0 in starts_y:
            for x0 in starts_x:
                origins.append((x0, y0))
                for local_y in range(1, patch_cells):
                    for local_x in range(1, patch_cells):
                        indices.append((y0 + local_y) * side + x0 + local_x)
        if not origins:
            raise ValueError("offset leaves no complete patch")
        self.side = int(side)
        self.patch_cells = int(patch_cells)
        self.axis = axis
        self.floor_fraction = float(floor_fraction)
        self.register_buffer("_origins", torch.tensor(origins, dtype=torch.int64), persistent=False)
        self.register_buffer("_interior_indices", torch.tensor(indices, dtype=torch.int64), persistent=False)

    @property
    def patch_count(self) -> int:
        return int(self._origins.shape[0])

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        unbatched = latent.ndim == 3
        if unbatched:
            latent = latent.unsqueeze(0)
        p = self.patch_cells
        if latent.ndim != 4 or latent.shape[1:] != (self.patch_count, p - 1, p):
            raise ValueError("latent must have shape (B,patch_count,p-1,p)")
        if latent.dtype not in (torch.float32, torch.float64):
            raise ValueError("latent must be float32 or float64")
        if latent.device != self._origins.device:
            raise ValueError("move the patch layer to the latent device before use")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("latent must be finite")
        spacing = (1.0 - self.floor_fraction) * torch.softmax(latent, dim=-1)
        spacing = spacing + self.floor_fraction / p
        coordinate = torch.cumsum(spacing, dim=-1)[..., :-1]
        if (
            not bool(torch.all(coordinate[..., 0] > 0.0))
            or not bool(torch.all(coordinate[..., -1] < 1.0))
            or not bool(torch.all(coordinate[..., 1:] > coordinate[..., :-1]))
        ):
            raise RuntimeError("patch increments collapsed in represented precision")

        line = torch.linspace(0.0, 1.0, self.side, dtype=latent.dtype, device=latent.device)
        yy, xx = torch.meshgrid(line, line, indexing="ij")
        identity = torch.stack((xx, yy), dim=-1).reshape(1, self.side**2, 2)
        identity = identity.expand(latent.shape[0], -1, -1)
        source_interior = identity.index_select(1, self._interior_indices)
        origins = self._origins.to(dtype=latent.dtype)
        if self.axis == "vertical":
            local = coordinate.transpose(-1, -2)
            moved = (origins[None, :, None, None, 1] + p * local) / (self.side - 1)
            updated = torch.stack((source_interior[..., 0], moved.flatten(start_dim=1)), dim=-1)
        else:
            local = coordinate
            moved = (origins[None, :, None, None, 0] + p * local) / (self.side - 1)
            updated = torch.stack((moved.flatten(start_dim=1), source_interior[..., 1]), dim=-1)
        output = identity.index_copy(1, self._interior_indices, updated)
        output = output.reshape(latent.shape[0], self.side, self.side, 2)
        return output[0] if unbatched else output


@dataclass(frozen=True)
class LocalPatchCompositionResult:
    controls: tuple[torch.Tensor, ...]
    dense: torch.Tensor


class LocalPatchComposition(torch.nn.Module):
    """Evaluate exact PL composition using dynamic point location per layer."""

    def __init__(
        self,
        layers: Sequence[LocalPatchMonotoneLayer],
        query_table: StructuredDenseQueryTable,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("at least one local layer is required")
        if any(layer.side**2 != query_table.control_vertices for layer in layers):
            raise ValueError("all layers must match the query table's fine mesh")
        self.layers = torch.nn.ModuleList(layers)
        self.query_table = query_table

    def forward(self, latents: Sequence[torch.Tensor]) -> LocalPatchCompositionResult:
        if len(latents) != len(self.layers):
            raise ValueError("one latent tensor is required per layer")
        controls = tuple(layer(latent) for layer, latent in zip(self.layers, latents))
        flattened = [control.reshape(*control.shape[:-3], -1, 2) for control in controls]
        dense = compose_control_maps([self.query_table] * len(controls), flattened)
        return LocalPatchCompositionResult(controls=controls, dense=dense)
