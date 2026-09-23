"""Solve-free dyadic P1 refinement with latent-driven safe vertex passes.

The output is one P1 map on the final fixed SW--NE triangulation, not a
composition of maps evaluated on changing triangulations. Exact-arithmetic
topology follows from exact P1 prolongation and the per-color orientation
invariant of :class:`SafeColoredVertexRelaxation`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .colored_vertex_relaxation import SafeColoredVertexRelaxation


def exact_dyadic_p1_refine(coarse: torch.Tensor) -> torch.Tensor:
    """Return the *same continuous P1 map* on the next fixed grid.

    The center of a coarse square lies on its SW--NE diagonal, hence uses
    the two diagonal endpoints. Bilinear four-corner interpolation would not
    preserve the original coarse P1 map.
    """
    if coarse.ndim != 4 or coarse.shape[1] != coarse.shape[2] or coarse.shape[-1] != 2:
        raise ValueError("coarse must have shape (batch,side,side,2)")
    batch, side = coarse.shape[:2]
    fine = coarse.new_empty((batch, 2 * side - 1, 2 * side - 1, 2))
    fine[:, ::2, ::2] = coarse
    fine[:, ::2, 1::2] = (coarse[:, :, :-1] + coarse[:, :, 1:]) * 0.5
    fine[:, 1::2, ::2] = (coarse[:, :-1, :] + coarse[:, 1:, :]) * 0.5
    fine[:, 1::2, 1::2] = (coarse[:, :-1, :-1] + coarse[:, 1:, 1:]) * 0.5
    return fine


class ForwardP1Pyramid(nn.Module):
    """Map dense multiscale latents to one fixed-grid P1 homeomorphism.

    ``seed_logits`` is a sequence of (B,S-2,S-2,2) arrays, one per seed
    pass. ``level_logits`` has one such array at each dyadic fine level.
    Old even/even vertices have their logits masked to zero during a level
    pass. Finite logits, float arithmetic aside, always preserve orientation
    and the pointwise identity boundary. This class makes no universal
    approximation claim for a *fixed* seed pass count.
    """

    def __init__(
        self,
        seed_side: int,
        final_side: int,
        *,
        seed_passes: int = 2,
        safety_fraction: float = 0.85,
        raw_span: float = 2.0,
        checkpoint_passes: bool = False,
    ) -> None:
        super().__init__()
        if seed_side < 3 or final_side < seed_side or seed_passes < 0:
            raise ValueError("invalid seed/final sides or pass count")
        sides: list[int] = []
        side = seed_side
        while side < final_side:
            side = 2 * side - 1
            sides.append(side)
        if side != final_side:
            raise ValueError("final_side must be reachable by dyadic refinement")
        self.seed_side = seed_side
        self.final_side = final_side
        self.seed_passes = seed_passes
        self.level_sides = tuple(sides)
        self.checkpoint_passes = checkpoint_passes
        self.seed_update = SafeColoredVertexRelaxation(
            seed_side, safety_fraction=safety_fraction,
            motion_mode="radial", raw_span=raw_span,
        )
        self.level_updates = nn.ModuleList(
            SafeColoredVertexRelaxation(
                n, safety_fraction=safety_fraction,
                motion_mode="radial", raw_span=raw_span,
            ) for n in sides
        )
        for n in sides:
            rows = torch.arange(1, n - 1)[:, None]
            cols = torch.arange(1, n - 1)[None, :]
            is_new = ((rows % 2 != 0) | (cols % 2 != 0))[None, :, :, None]
            self.register_buffer(f"_new_mask_{n}", is_new, persistent=False)

    def _update(self, layer: nn.Module, mapped: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_passes and torch.is_grad_enabled():
            return checkpoint(layer, mapped, latent, use_reentrant=False)
        return layer(mapped, latent)

    def forward(
        self,
        seed_logits: Sequence[torch.Tensor],
        level_logits: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(seed_logits) != self.seed_passes or len(level_logits) != len(self.level_sides):
            raise ValueError("incorrect number of seed or refinement latent arrays")
        inputs = tuple(seed_logits) + tuple(level_logits)
        if not inputs:
            raise ValueError("at least one latent tensor is needed to set batch/device/dtype")
        first = inputs[0]
        if first.ndim != 4 or first.shape[0] < 1 or first.shape[-1] != 2:
            raise ValueError("latents need shape (batch,side-2,side-2,2)")
        batch = first.shape[0]
        axis = torch.arange(self.seed_side, device=first.device, dtype=first.dtype) / (self.seed_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        mapped = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        for latent in seed_logits:
            mapped = self._update(self.seed_update, mapped, latent)
        for n, layer, latent in zip(self.level_sides, self.level_updates, level_logits):
            mapped = exact_dyadic_p1_refine(mapped)
            if latent.shape != (batch, n - 2, n - 2, 2):
                raise ValueError("refinement latent has wrong shape")
            masked = torch.where(getattr(self, f"_new_mask_{n}"), latent, torch.zeros_like(latent))
            mapped = self._update(layer, mapped, masked)
        return mapped
