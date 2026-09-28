"""Compare F1/F2 schedules on one exactly refined, fixed-grid Q1 map.

Each stage consumes one full interior latent field for F1 or four fields for
the four staggered F2 patch passes. At dyadic fine stages, only new vertices
may move. Each accepted update protects all four Q1 cell-corner determinants.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .digital_q1 import SafeColoredQ1Relaxation, StaggeredPatchQ1Layer, q1_dyadic_refine


class ScheduledQ1Pyramid(nn.Module):
    """A declared string like ``22211`` selects seed and each dyadic stage."""

    def __init__(
        self, seed_side: int, final_side: int, schedule: str, *,
        patch_cells: int = 4, safety_fraction: float = .75,
        minimum_jacobian: float | None = .05,
        seed_repeats: int = 1,
        checkpoint_colors: bool = False,
    ) -> None:
        super().__init__()
        if seed_side < 3 or final_side < seed_side or patch_cells < 2 or patch_cells % 2:
            raise ValueError("invalid sides or patch width")
        if seed_repeats < 1:
            raise ValueError("seed_repeats must be positive")
        sides = [seed_side]
        while sides[-1] < final_side:
            sides.append(2 * sides[-1] - 1)
        if sides[-1] != final_side or len(schedule) != len(sides) or any(c not in "12" for c in schedule):
            raise ValueError("schedule must have one F1/F2 digit per exact dyadic side")
        self.sides = tuple(sides)
        self.schedule = schedule
        self.seed_repeats = seed_repeats
        self.checkpoint_colors = checkpoint_colors
        self.passes_per_stage = tuple(
            (1 if mode == "1" else 4) * (seed_repeats if index == 0 else 1)
            for index, mode in enumerate(schedule)
        )
        self.stages = nn.ModuleList(
            SafeColoredQ1Relaxation(
                side, safety_fraction=safety_fraction,
                minimum_jacobian=minimum_jacobian,
                checkpoint_colors=checkpoint_colors,
            ) if mode == "1" else StaggeredPatchQ1Layer(
                side, patch_cells=patch_cells, safety_fraction=safety_fraction,
                minimum_jacobian=minimum_jacobian,
            )
            for side, mode in zip(sides, schedule)
        )
        if any(len(stage.passes) != 4 for mode, stage in zip(schedule, self.stages) if mode == "2"):
            raise ValueError("this ablation needs four F2 passes per stage")
        for side in sides[1:]:
            row = torch.arange(1, side - 1)[:, None]
            col = torch.arange(1, side - 1)[None, :]
            mask = ((row % 2 != 0) | (col % 2 != 0))[None, :, :, None]
            self.register_buffer(f"_new_{side}", mask, persistent=False)

    def active_latent_scalar_count_per_sample(self) -> int:
        """Count structurally selected scalar logits, excluding masked/unused entries."""
        count = 0
        for index, (side, mode, layer) in enumerate(zip(
            self.sides, self.schedule, self.stages,
        )):
            if mode == "1":
                vertices = (side - 2) ** 2 if index == 0 else int(
                    getattr(self, f"_new_{side}").sum().item()
                )
                count += 2 * vertices * (self.seed_repeats if index == 0 else 1)
            else:
                new_mask = None if index == 0 else getattr(
                    self, f"_new_{side}",
                ).reshape(-1)
                for patch_pass in layer.passes:
                    selected = patch_pass.latent_ids
                    selected_count = selected.numel() if new_mask is None else int(
                        new_mask[selected].sum().item()
                    )
                    count += 2 * selected_count * (
                        self.seed_repeats if index == 0 else 1
                    )
        return count

    def forward(self, stage_logits: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
        if len(stage_logits) != len(self.sides):
            raise ValueError("one latent group per stage required")
        first = stage_logits[0][0]
        if first.ndim != 4 or first.shape[0] < 1 or first.shape[1:] != (
            self.sides[0] - 2, self.sides[0] - 2, 2
        ):
            raise ValueError("invalid seed latent shape")
        axis = torch.arange(self.sides[0], device=first.device, dtype=first.dtype) / (
            self.sides[0] - 1
        )
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        mapped = torch.stack((xx, yy), dim=-1)[None].expand(first.shape[0], -1, -1, -1)
        for index, (side, mode, layer, fields) in enumerate(zip(
            self.sides, self.schedule, self.stages, stage_logits,
        )):
            if len(fields) != self.passes_per_stage[index]:
                raise ValueError("wrong number of F1/F2 latent fields")
            if index:
                mapped = q1_dyadic_refine(mapped)
                new_mask = getattr(self, f"_new_{side}")
            prepared = []
            for field in fields:
                if field.shape != (first.shape[0], side - 2, side - 2, 2) or (
                    field.device != first.device or field.dtype != first.dtype
                ):
                    raise ValueError("latent field has wrong shape, dtype or device")
                prepared.append(field if index == 0 else torch.where(
                    new_mask, field, torch.zeros_like(field),
                ))
            if mode == "1":
                for field in prepared:
                    mapped = layer(mapped, field)
            else:
                for start in range(0, len(prepared), 4):
                    mapped = layer(mapped, tuple(prepared[start:start + 4]))
        return mapped
