"""Original-grid P1 image-conditioned spectral safe feedback decoder."""

from __future__ import annotations

import torch

from .colored_vertex_relaxation import HierarchicalConvexQuadLocalLayer, SafeColoredVertexRelaxation
from .photometric_hint import local_photometric_logits
from .sine_spectral import spectralize_bounded_logits


class SpectralSafeFeedbackLayer(torch.nn.Module):
    """Decode one A6 map and additional safe photometric residual passes.

    The only trainable quantities arrive through the encoder-provided latent
    tuple. Every pass remains P1 on the same structured triangulation; no
    image is warped and resampled between passes. The spectral top-k switch
    is piecewise differentiable away from coefficient ordering ties.
    """

    def __init__(
        self,
        side: int,
        *,
        initial_window: int = 3,
        initial_ridge: float = 1.0,
        initial_gain: float = 1.0,
        initial_modes: int = 16,
        extra_passes: int = 1,
        extra_gain: float = 0.25,
        extra_modes: int = 16,
        floor_fraction: float = 0.2,
    ) -> None:
        super().__init__()
        if extra_passes < 0 or initial_modes < 1 or extra_modes < 1 or extra_gain <= 0:
            raise ValueError("invalid spectral feedback configuration")
        self.side = side
        self.initial_window = initial_window
        self.initial_ridge = initial_ridge
        self.initial_gain = initial_gain
        self.initial_modes = initial_modes
        self.extra_passes = extra_passes
        self.extra_gain = extra_gain
        self.extra_modes = extra_modes
        self.initial = HierarchicalConvexQuadLocalLayer(side, motion_mode="radial")
        self.refiner = SafeColoredVertexRelaxation(
            side, motion_mode="radial", floor_fraction=floor_fraction,
        )

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor, latent) -> torch.Tensor:
        root, levels, local_logits = latent
        base = self.initial.base(root, levels)
        hint = local_photometric_logits(
            fixed, moving, base, window=self.initial_window,
            ridge=self.initial_ridge, raw_span=self.initial.local.raw_span,
        )
        hint = spectralize_bounded_logits(
            hint, side=self.side, raw_span=self.initial.local.raw_span,
            count=self.initial_modes,
        )
        mapped = self.initial.local(base, local_logits + self.initial_gain * hint)
        if not self.extra_passes:
            return mapped
        area_floor = self.refiner.compute_area_floor(mapped)
        for _ in range(self.extra_passes):
            residual_hint = local_photometric_logits(
                fixed, moving, mapped, window=self.initial_window,
                ridge=self.initial_ridge, raw_span=self.refiner.raw_span,
            )
            residual_hint = spectralize_bounded_logits(
                residual_hint, side=self.side, raw_span=self.refiner.raw_span,
                count=self.extra_modes,
            )
            mapped = self.refiner(mapped, self.extra_gain * residual_hint,
                                  area_floor=area_floor)
        return mapped
