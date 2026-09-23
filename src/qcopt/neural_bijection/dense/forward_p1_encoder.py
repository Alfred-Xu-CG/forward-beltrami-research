"""Image-pair encoder for the solve-free fixed-grid P1 pyramid."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ForwardP1ImageEncoder(nn.Module):
    """Predict every seed/refinement latent from two images and coordinates.

    All heads start at zero, so a fresh network initially emits the identity
    map. The output is a sequence of latent fields, *not* an unconstrained
    displacement field. Topology is enforced by the downstream decoder.
    """

    def __init__(
        self,
        seed_side: int,
        level_sides: tuple[int, ...],
        *,
        seed_passes: int = 2,
        feature_side: int | None = None,
        width: int = 16,
    ) -> None:
        super().__init__()
        if width < 2 or seed_side < 3 or seed_passes < 0:
            raise ValueError("invalid encoder width, seed side or pass count")
        self.seed_side = seed_side
        self.level_sides = tuple(level_sides)
        self.seed_passes = seed_passes
        self.feature_side = feature_side or min(self.level_sides[-1] if self.level_sides else seed_side, 257)
        self.stem = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.context = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU())
            for _ in range(4)
        )
        self.fuse = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.seed_heads = nn.ModuleList(nn.Conv2d(width, 2, 1) for _ in range(seed_passes))
        self.level_heads = nn.ModuleList(nn.Conv2d(width, 2, 1) for _ in self.level_sides)
        for head in (*self.seed_heads, *self.level_heads):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, fixed: torch.Tensor, moving: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1:
            raise ValueError("fixed and moving must be matching grayscale image batches")
        batch = fixed.shape[0]
        pair = F.interpolate(torch.cat((fixed, moving), dim=1),
                             size=(self.feature_side, self.feature_side),
                             mode="bilinear", align_corners=True)
        axis = torch.arange(self.feature_side, device=pair.device, dtype=pair.dtype) / (self.feature_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=0)[None].expand(batch, -1, -1, -1)
        fine = self.stem(torch.cat((pair, coordinates), dim=1))
        current, fused = fine, fine
        for block in self.context:
            current = block(F.avg_pool2d(current, 2))
            fused = fused + F.interpolate(current, size=fine.shape[-2:], mode="bilinear", align_corners=True)
        features = self.fuse(fused)
        seed_feature = F.interpolate(features, size=(self.seed_side, self.seed_side),
                                     mode="bilinear", align_corners=True)
        seed = tuple(
            head(seed_feature)[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            for head in self.seed_heads
        )
        levels = tuple(
            head(F.interpolate(features, size=(side, side), mode="bilinear", align_corners=True))
            [:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            for side, head in zip(self.level_sides, self.level_heads)
        )
        return seed, levels
