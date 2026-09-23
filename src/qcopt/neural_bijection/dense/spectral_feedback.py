"""Original-grid P1 image-conditioned spectral safe feedback decoder."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .colored_vertex_relaxation import HierarchicalConvexQuadLocalLayer, SafeColoredVertexRelaxation
from .photometric_hint import local_photometric_logits
from .qc_initial_homotopy import cap_initial_map
from .qc_radial_relaxation import SafeColoredQCRadialRelaxation
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
        extra_qc_cap: float | None = None,
        initial_qc_cap: float | None = None,
        extra_checkpoint: bool = False,
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
        self.extra_qc_cap = extra_qc_cap
        self.initial_qc_cap = initial_qc_cap
        self.extra_checkpoint = bool(extra_checkpoint)
        self.initial = HierarchicalConvexQuadLocalLayer(side, motion_mode="radial")
        self.refiner = (
            SafeColoredVertexRelaxation(side, motion_mode="radial", floor_fraction=floor_fraction)
            if extra_qc_cap is None else
            SafeColoredQCRadialRelaxation(side, qc_cap=extra_qc_cap,
                                          floor_fraction=floor_fraction)
        )

    def initial_map(self, fixed: torch.Tensor, moving: torch.Tensor, latent) -> torch.Tensor:
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
        return mapped if self.initial_qc_cap is None else cap_initial_map(
            mapped, qc_cap=self.initial_qc_cap)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor, latent) -> torch.Tensor:
        mapped = self.initial_map(fixed, moving, latent)
        if not self.extra_passes:
            return mapped
        area_floor = self.refiner.compute_area_floor(mapped) if self.refiner.floor_fraction else None
        def refine(base_map: torch.Tensor, proposal: torch.Tensor,
                   floor: torch.Tensor | None) -> torch.Tensor:
            return self.refiner(base_map, proposal, area_floor=floor)
        for _ in range(self.extra_passes):
            residual_hint = local_photometric_logits(
                fixed, moving, mapped, window=self.initial_window,
                ridge=self.initial_ridge, raw_span=self.refiner.raw_span,
            )
            residual_hint = spectralize_bounded_logits(
                residual_hint, side=self.side, raw_span=self.refiner.raw_span,
                count=self.extra_modes,
            )
            proposal = self.extra_gain * residual_hint
            mapped = (
                checkpoint(refine, mapped, proposal, area_floor, use_reentrant=False)
                if self.extra_checkpoint and torch.is_grad_enabled()
                else refine(mapped, proposal, area_floor)
            )
        return mapped
