"""Solve-free fiber-monotone maps that are P1 homeomorphisms on one grid."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class MonotoneFiberP1Layer(nn.Module):
    """Positive normalized edge lengths along horizontal or vertical fibers.

    Input ``logits`` has shape (B,side-2,side-1) for horizontal fibers:
    one log weight per horizontal edge on each strictly interior row. For
    vertical fibers it has shape (B,side-1,side-2). The other coordinate
    remains the source coordinate, and the entire boundary is the identity.
    Each fixed-grid triangle has positive area because its horizontal
    (vertical) edge is positively oriented. This is a restricted but very
    fast forward generator; arbitrary two-component maps need more layers.
    """

    def __init__(self, side: int, *, axis: str = "horizontal",
                 floor_fraction: float = 0.05, logit_span: float = 8.0) -> None:
        super().__init__()
        if side < 3 or axis not in ("horizontal", "vertical"):
            raise ValueError("side >= 3 and a valid axis are required")
        if not 0 < floor_fraction < 1 or not math.isfinite(logit_span) or logit_span <= 0:
            raise ValueError("floor_fraction must be in (0,1); logit_span positive")
        self.side = side
        self.axis = axis
        self.floor_fraction = floor_fraction
        self.logit_span = logit_span

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError("logits must be a batch of two-dimensional fields")
        shape = (
            (logits.shape[0], self.side - 2, self.side - 1)
            if self.axis == "horizontal" else
            (logits.shape[0], self.side - 1, self.side - 2)
        )
        if logits.shape != shape or logits.shape[0] < 1:
            raise ValueError(f"logits must have shape {shape}")
        if logits.dtype not in (torch.float32, torch.float64):
            raise ValueError("float32/float64 are required for the face guarantee")
        rows = logits if self.axis == "horizontal" else logits.transpose(1, 2)
        weights = torch.softmax(self.logit_span * torch.tanh(rows), dim=-1)
        edge_lengths = (
            (1.0 - self.floor_fraction) * weights
            + self.floor_fraction / (self.side - 1)
        )
        starts = torch.cat((
            edge_lengths.new_zeros((*edge_lengths.shape[:-1], 1)),
            edge_lengths.cumsum(dim=-1)[..., :-1],
            edge_lengths.new_ones((*edge_lengths.shape[:-1], 1)),
        ), dim=-1)
        axis = torch.arange(self.side, device=logits.device,
                            dtype=logits.dtype) / (self.side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        horizontal = xx[None].expand(logits.shape[0], -1, -1).clone()
        vertical = yy[None].expand(logits.shape[0], -1, -1).clone()
        if self.axis == "horizontal":
            horizontal[:, 1:-1] = starts
        else:
            vertical[:, :, 1:-1] = starts.transpose(1, 2)
        return torch.stack((horizontal, vertical), dim=-1)


class MultiscaleMonotoneFiberP1Layer(nn.Module):
    """Upsample/sum multiscale *log edge lengths*, then normalize at fine scale.

    This does not interpolate coarse maps. Every positive edge length is
    formed on the final grid, and the final-grid P1 topology proof applies
    unchanged. A coarse-only latent has limited expressivity; optional fine
    residual fields restore per-fiber detail without changing safety.
    """

    def __init__(self, side: int, *, axis: str = "horizontal",
                 floor_fraction: float = 0.05, logit_span: float = 8.0) -> None:
        super().__init__()
        self.fiber = MonotoneFiberP1Layer(
            side, axis=axis, floor_fraction=floor_fraction,
            logit_span=logit_span,
        )

    def forward(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        if not levels:
            raise ValueError("at least one latent level is required")
        first = levels[0]
        if first.ndim != 3:
            raise ValueError("each latent level needs shape (batch,rows,edges)")
        target_shape = (
            (self.fiber.side - 2, self.fiber.side - 1)
            if self.fiber.axis == "horizontal" else
            (self.fiber.side - 1, self.fiber.side - 2)
        )
        dense = first.new_zeros((first.shape[0], *target_shape))
        for level in levels:
            if (level.ndim != 3 or level.shape[0] != first.shape[0]
                    or level.device != first.device or level.dtype != first.dtype
                    or min(level.shape[1:]) < 2):
                raise ValueError("latent levels must share batch/device/dtype and have extent >= 2")
            dense = dense + F.interpolate(
                level[:, None], size=target_shape, mode="bilinear",
                align_corners=True,
            )[:, 0]
        return self.fiber(dense)
