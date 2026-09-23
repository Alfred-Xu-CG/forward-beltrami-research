"""A tiny spectral latent expanded by safe coarse-to-fine P1 updates."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .colored_vertex_relaxation import SafeColoredVertexRelaxation
from .forward_p1_pyramid import exact_dyadic_p1_refine
from .patch_field import StaggeredPatchP1Layer


class SineModeP1Refiner(nn.Module):
    """Map one or more amplitudes onto the same final fixed-grid P1 map.

    Each supplied basis is sin(2*pi*kx*x)*sin(2*pi*ky*y), multiplied by a
    fixed two-vector direction and an image-derived amplitude. The explicit
    dictionary diagnoses learnable fine-only detail; it is not universal.
    Finite amplitudes are clipped to finite logits and passed through safe
    local P1 updates.
    """

    def __init__(
        self,
        coarse_side: int,
        final_side: int,
        *,
        cycles: int | Sequence[tuple[int, int]] = 128,
        directions: Sequence[tuple[float, float]] | None = None,
        mechanism: str = "colored",
        patch_cells: int = 8,
        minimum_jacobian: float = 0.05,
        sweeps: int = 1,
        amplitude_limit: float = 0.1,
    ) -> None:
        super().__init__()
        if mechanism not in ("colored", "patch"):
            raise ValueError("mechanism must be colored/patch")
        modes = ((cycles, cycles),) if isinstance(cycles, int) else tuple(cycles)
        if not modes or any(
            not isinstance(kx, int) or not isinstance(ky, int)
            or kx < 1 or ky < 1 or max(kx, ky) > final_side - 1
            for kx, ky in modes
        ):
            raise ValueError("cycles must be positive integer pairs within the final grid")
        if directions is None:
            directions = ((1.0, 1.0),) * len(modes)
        if len(directions) != len(modes) or any(
            not all(math.isfinite(v) and abs(v) <= 1e3 for v in direction)
            for direction in directions
        ):
            raise ValueError("directions must be bounded and match the mode count")
        if sweeps < 1 or not math.isfinite(amplitude_limit) or amplitude_limit <= 0:
            raise ValueError("sweeps and amplitude_limit must be positive")
        sides = []
        side = coarse_side
        while side < final_side:
            side = 2 * side - 1
            sides.append(side)
        if side != final_side or not sides:
            raise ValueError("final_side must be a strictly finer dyadic level")
        self.coarse_side = coarse_side
        self.final_side = final_side
        self.level_sides = tuple(sides)
        self.cycles = modes
        self.register_buffer(
            "directions",
            torch.tensor(directions, dtype=torch.float64),
            persistent=False,
        )
        self.mechanism = mechanism
        self.minimum_jacobian = minimum_jacobian
        self.sweeps = sweeps
        self.amplitude_limit = amplitude_limit
        if mechanism == "colored":
            self.levels = nn.ModuleList(
                SafeColoredVertexRelaxation(
                    n, motion_mode="radial", raw_span=2.0,
                ) for n in sides
            )
        else:
            self.levels = nn.ModuleList(
                StaggeredPatchP1Layer(
                    n, patch_cells, minimum_jacobian=minimum_jacobian,
                ) for n in sides
            )

    def _mode(self, side: int, reference: torch.Tensor) -> torch.Tensor:
        axis = torch.arange(side, device=reference.device, dtype=torch.float64) / (side - 1)
        waves = []
        for kx, ky in self.cycles:
            wx = torch.sin(2 * math.pi * kx * axis)
            wy = torch.sin(2 * math.pi * ky * axis)
            wx = torch.where(wx.abs() < 1e-12, torch.zeros_like(wx), wx)
            wy = torch.where(wy.abs() < 1e-12, torch.zeros_like(wy), wy)
            waves.append(wy[:, None] * wx[None, :])
        return torch.stack(waves).to(dtype=reference.dtype)

    def forward(self, coarse: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        if coarse.ndim != 4 or coarse.shape[1:] != (self.coarse_side, self.coarse_side, 2):
            raise ValueError("coarse must have shape (B,coarse_side,coarse_side,2)")
        if amplitude.ndim == 1 and len(self.cycles) == 1:
            amplitude = amplitude[:, None]
        if amplitude.shape != (coarse.shape[0], len(self.cycles)) or not torch.isfinite(amplitude).all():
            raise ValueError("amplitude must be one finite value per batch item and mode")
        if amplitude.dtype != coarse.dtype or amplitude.device != coarse.device:
            raise ValueError("amplitude and coarse must match dtype/device")
        amplitude = amplitude.clamp(-self.amplitude_limit, self.amplitude_limit)
        mapped = coarse
        baseline = coarse
        for side, layer in zip(self.level_sides, self.levels):
            mapped = exact_dyadic_p1_refine(mapped)
            baseline = exact_dyadic_p1_refine(baseline)
            mode = self._mode(side, mapped)
            desired = baseline + torch.einsum(
                "bk,khw,kd->bhwd", amplitude, mode,
                self.directions.to(dtype=mapped.dtype),
            )
            for _ in range(self.sweeps):
                if self.mechanism == "colored":
                    delta = desired - mapped
                    span = 2.0 / (side - 1)
                    logits = torch.atanh((delta[:, 1:-1, 1:-1] / span).clamp(-0.95, 0.95))
                    floor = mapped.new_full((mapped.shape[0],),
                                            self.minimum_jacobian / (side - 1) ** 2)
                    mapped = layer(mapped, logits, area_floor=floor)
                else:
                    assigned = torch.zeros(side * side, device=mapped.device, dtype=torch.bool)
                    for patch_pass in layer.passes:
                        ids = patch_pass.interior_ids
                        new = ~assigned[ids]
                        delta = (desired - mapped).reshape(mapped.shape[0], -1, 2)[:, ids]
                        raw = torch.where(new[None, :, None], delta, torch.zeros_like(delta))
                        span = patch_pass.raw_span * patch_pass.patch_cells / (side - 1)
                        logits = torch.atanh((raw / span).clamp(-0.95, 0.95))
                        mapped = patch_pass(mapped, logits)
                        assigned[ids] = True
        return mapped
