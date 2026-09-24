"""Image-pair encoder for the solve-free fixed-grid P1 pyramid."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .photometric_hint import physical_image_gradient


def ridge_local_flow_features(
    fixed: torch.Tensor, moving: torch.Tensor, *,
    window: int = 5, ridge: float = 1.0, scale: float = .02,
) -> torch.Tensor:
    """Differentiable two-channel local first-order displacement proposal.

    At each image pixel solve the positive 2x2 ridge normal matrix formed
    from a local average of moving-image gradients. This is a feature, not
    a deformation or a topology guarantee; the P1 decoder remains decisive.
    """
    if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1:
        raise ValueError("fixed/moving must be matching BCHW grayscale images")
    if window < 1 or window % 2 != 1 or ridge <= 0 or scale <= 0:
        raise ValueError("window must be odd and ridge/scale positive")
    gradient = physical_image_gradient(moving)
    gx, gy = gradient[:, :1], gradient[:, 1:]
    residual = fixed - moving
    mean = lambda values: F.avg_pool2d(
        values, window, stride=1, padding=window // 2,
        count_include_pad=False,
    )
    gxx = mean(gx * gx) + ridge
    gxy = mean(gx * gy)
    gyy = mean(gy * gy) + ridge
    bx = mean(gx * residual)
    by = mean(gy * residual)
    determinant = gxx * gyy - gxy.square()
    dx = (gyy * bx - gxy * by) / determinant
    dy = (gxx * by - gxy * bx) / determinant
    return torch.tanh(torch.cat((dx, dy), dim=1) / scale)


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
        flow_hint: bool = False,
    ) -> None:
        super().__init__()
        if width < 2 or seed_side < 3 or seed_passes < 0:
            raise ValueError("invalid encoder width, seed side or pass count")
        self.seed_side = seed_side
        self.level_sides = tuple(level_sides)
        self.seed_passes = seed_passes
        self.flow_hint = flow_hint
        self.flow_feature_gain = 1.0
        self.feature_side = feature_side or min(self.level_sides[-1] if self.level_sides else seed_side, 257)
        self.stem = nn.Sequential(
            nn.Conv2d(6 if flow_hint else 4, width, 3, padding=1), nn.GELU(),
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

    def features(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1:
            raise ValueError("fixed and moving must be matching grayscale image batches")
        batch = fixed.shape[0]
        pair = F.interpolate(torch.cat((fixed, moving), dim=1),
                             size=(self.feature_side, self.feature_side),
                             mode="bilinear", align_corners=True)
        axis = torch.arange(self.feature_side, device=pair.device, dtype=pair.dtype) / (self.feature_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=0)[None].expand(batch, -1, -1, -1)
        feature_channels = (pair, coordinates)
        if self.flow_hint:
            feature_channels += (
                self.flow_feature_gain * ridge_local_flow_features(pair[:, :1], pair[:, 1:]),
            )
        fine = self.stem(torch.cat(feature_channels, dim=1))
        current, fused = fine, fine
        for block in self.context:
            current = block(F.avg_pool2d(current, 2))
            fused = fused + F.interpolate(current, size=fine.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse(fused)

    def forward(
        self, fixed: torch.Tensor, moving: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        features = self.features(fixed, moving)
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


class PatchPyramidImageEncoder(ForwardP1ImageEncoder):
    """Predict four staggered patch fields at the seed and every fine level."""

    def __init__(
        self, seed_side: int, level_sides: tuple[int, ...], *,
        feature_side: int | None = None, width: int = 16,
        flow_hint: bool = False,
    ) -> None:
        super().__init__(seed_side, level_sides, seed_passes=4,
                         feature_side=feature_side, width=width,
                         flow_hint=flow_hint)
        self.level_heads = nn.ModuleList(
            nn.ModuleList(nn.Conv2d(width, 2, 1) for _ in range(4))
            for _ in level_sides
        )
        for level in self.level_heads:
            for head in level:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(
        self, fixed: torch.Tensor, moving: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[tuple[torch.Tensor, ...], ...]]:
        features = self.features(fixed, moving)
        seed_feature = F.interpolate(
            features, size=(self.seed_side, self.seed_side),
            mode="bilinear", align_corners=True,
        )
        seed = tuple(
            head(seed_feature)[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            for head in self.seed_heads
        )
        levels = tuple(
            tuple(
                head(feature)[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
                for head in heads
            )
            for feature, heads in (
                (F.interpolate(features, size=(side, side), mode="bilinear",
                               align_corners=True), heads)
                for side, heads in zip(self.level_sides, self.level_heads)
            )
        )
        return seed, levels
