"""A mixed F2-D/F1-D pyramid producing one fixed-grid Q1 map.

The coarse patch update and every fine single-vertex update act on the same
vertex table. Dyadic refinement is the exact restriction of its Q1 function,
not a composition followed by unverified resampling. The exact-arithmetic
homeomorphism claim assumes finite latents, a valid input map at each update,
and the fixed, ordered rectangle boundary. Audit actual saved coordinates
separately before making a floating-point output claim.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .digital_q1 import (
    SafeColoredQ1Relaxation,
    StaggeredPatchQ1Layer,
    q1_dyadic_refine,
)


class HybridPatchSeedVertexQ1Pyramid(nn.Module):
    """Four staggered F2-D seed passes, then new-vertex-only F1-D levels."""

    def __init__(
        self, seed_side: int, final_side: int, *,
        patch_cells: int = 4,
        safety_fraction: float = 0.75,
        minimum_jacobian: float | None = 0.05,
    ) -> None:
        super().__init__()
        if seed_side < 3 or final_side < seed_side:
            raise ValueError("invalid seed or final side")
        sides: list[int] = []
        current = seed_side
        while current < final_side:
            current = 2 * current - 1
            sides.append(current)
        if current != final_side:
            raise ValueError("final side must be a dyadic refinement of seed side")
        self.seed_side = seed_side
        self.final_side = final_side
        self.level_sides = tuple(sides)
        self.seed_update = StaggeredPatchQ1Layer(
            seed_side, patch_cells=patch_cells,
            safety_fraction=safety_fraction,
            minimum_jacobian=minimum_jacobian,
        )
        self.seed_passes = len(self.seed_update.passes)
        self.level_updates = nn.ModuleList(
            SafeColoredQ1Relaxation(
                side, safety_fraction=safety_fraction,
                minimum_jacobian=minimum_jacobian,
            ) for side in sides
        )
        for side in sides:
            row = torch.arange(1, side - 1)[:, None]
            column = torch.arange(1, side - 1)[None, :]
            self.register_buffer(
                f"_new_mask_{side}", ((row % 2 != 0) | (column % 2 != 0))[None, :, :, None],
                persistent=False,
            )

    def forward(
        self,
        seed_logits: Sequence[torch.Tensor],
        level_logits: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(seed_logits) != self.seed_passes:
            raise ValueError("one seed latent field is required per patch pass")
        first = seed_logits[0]
        expected = (first.shape[0], self.seed_side - 2, self.seed_side - 2, 2)
        if first.ndim != 4 or first.shape != expected or first.shape[0] < 1:
            raise ValueError("seed latent field has wrong shape")
        if any(field.shape != expected or field.device != first.device or field.dtype != first.dtype
               for field in seed_logits):
            raise ValueError("seed latent fields must have matching shape, device and dtype")
        axis = torch.arange(self.seed_side, device=first.device, dtype=first.dtype) / (self.seed_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        identity = torch.stack((xx, yy), dim=-1)[None].expand(first.shape[0], -1, -1, -1)
        coarse = self.seed_update(identity, tuple(seed_logits))
        return self.forward_from(coarse, level_logits, start_index=0)

    def forward_from(
        self,
        mapped: torch.Tensor,
        level_logits: Sequence[torch.Tensor],
        *, start_index: int,
    ) -> torch.Tensor:
        """Continue from a valid Q1 map at an existing pyramid level."""
        if not 0 <= start_index <= len(self.level_sides):
            raise ValueError("invalid start_index")
        previous_side = self.seed_side if start_index == 0 else self.level_sides[start_index - 1]
        if mapped.ndim != 4 or mapped.shape[1:] != (previous_side, previous_side, 2):
            raise ValueError("starting map has wrong shape")
        if len(level_logits) != len(self.level_sides) - start_index:
            raise ValueError("wrong number of refinement latent fields")
        for side, layer, latent in zip(
            self.level_sides[start_index:],
            self.level_updates[start_index:],
            level_logits,
        ):
            if latent.shape != (mapped.shape[0], side - 2, side - 2, 2):
                raise ValueError("refinement latent field has wrong shape")
            if latent.device != mapped.device or latent.dtype != mapped.dtype:
                raise ValueError("refinement latent field must match map device and dtype")
            mapped = q1_dyadic_refine(mapped)
            new_vertices = getattr(self, f"_new_mask_{side}")
            masked = torch.where(new_vertices, latent, torch.zeros_like(latent))
            mapped = layer(mapped, masked)
        return mapped
