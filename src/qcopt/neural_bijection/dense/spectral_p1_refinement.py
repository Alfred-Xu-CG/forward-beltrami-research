"""A tiny spectral latent expanded by safe coarse-to-fine P1 updates."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .colored_vertex_relaxation import SafeColoredVertexRelaxation
from .forward_p1_pyramid import exact_dyadic_p1_refine
from .patch_field import StaggeredPatchP1Layer


def _wave(side: int, kx: int, ky: int, *, device: torch.device,
          dtype: torch.dtype, axis: torch.Tensor,
          window: tuple[float, float, float, float] | None = None) -> torch.Tensor:
    wx = torch.sin(2 * math.pi * kx * axis)
    wy = torch.sin(2 * math.pi * ky * axis)
    wx = torch.where(wx.abs() < 1e-12, torch.zeros_like(wx), wx)
    wy = torch.where(wy.abs() < 1e-12, torch.zeros_like(wy), wy)
    if window is not None:
        xlo, xhi, ylo, yhi = window
        tx = (axis - xlo) / (xhi - xlo)
        ty = (axis - ylo) / (yhi - ylo)
        wx = wx * torch.where((tx >= 0) & (tx <= 1),
                              torch.sin(math.pi * tx).square(), 0.0)
        wy = wy * torch.where((ty >= 0) & (ty <= 1),
                              torch.sin(math.pi * ty).square(), 0.0)
    return (wy[:, None] * wx[None, :]).to(device=device, dtype=dtype)


class _StreamedSineDisplacement(torch.autograd.Function):
    """Linear spectral synthesis with recomputed modes in the amplitude VJP.

    Avoids retaining a K x side x side basis stack when the dictionary grows.
    The fixed cycles and directions are architecture constants, not trainable.
    """

    @staticmethod
    def forward(ctx, amplitude: torch.Tensor, directions: torch.Tensor,
                cycles: tuple[tuple[int, int], ...],
                windows: tuple[tuple[float, float, float, float] | None, ...],
                side: int) -> torch.Tensor:
        batch = amplitude.shape[0]
        output = amplitude.new_zeros((batch, side, side, 2))
        axis = torch.arange(side, device=amplitude.device, dtype=torch.float64) / (side - 1)
        for k, (kx, ky) in enumerate(cycles):
            wave = _wave(side, kx, ky, device=amplitude.device,
                         dtype=amplitude.dtype, axis=axis, window=windows[k])
            output += (amplitude[:, k, None, None, None]
                       * wave[None, :, :, None]
                       * directions[k, None, None, :])
        ctx.save_for_backward(directions)
        ctx.cycles = cycles
        ctx.windows = windows
        ctx.side = side
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        (directions,) = ctx.saved_tensors
        side = ctx.side
        axis = torch.arange(side, device=grad_output.device, dtype=torch.float64) / (side - 1)
        components = []
        for k, (kx, ky) in enumerate(ctx.cycles):
            wave = _wave(side, kx, ky, device=grad_output.device,
                         dtype=grad_output.dtype, axis=axis,
                         window=ctx.windows[k])
            components.append(torch.einsum(
                "bhwd,hw,d->b", grad_output, wave, directions[k],
            ))
        return torch.stack(components, dim=1), None, None, None, None


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
        windows: Sequence[tuple[float, float, float, float]] | None = None,
        mechanism: str = "colored",
        patch_cells: int = 8,
        minimum_jacobian: float = 0.05,
        sweeps: int = 1,
        amplitude_limit: float = 0.1,
        checkpoint_updates: bool = False,
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
        if windows is None:
            mode_windows: tuple[tuple[float, float, float, float] | None, ...] = (
                (None,) * len(modes)
            )
        else:
            mode_windows = tuple(windows)
            if len(mode_windows) != len(modes) or any(
                len(window) != 4 or not all(math.isfinite(v) for v in window)
                or not (0 <= window[0] < window[1] <= 1)
                or not (0 <= window[2] < window[3] <= 1)
                for window in mode_windows
            ):
                raise ValueError("windows must be valid normalized rectangles per mode")
        if sweeps < 1 or not math.isfinite(amplitude_limit) or amplitude_limit <= 0:
            raise ValueError("sweeps and amplitude_limit must be positive")
        if not math.isfinite(minimum_jacobian) or not 0 < minimum_jacobian < 1:
            raise ValueError("minimum_jacobian must be finite and in (0,1)")
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
        self.windows = mode_windows
        self.register_buffer(
            "directions",
            torch.tensor(directions, dtype=torch.float64),
            persistent=False,
        )
        self.mechanism = mechanism
        self.minimum_jacobian = minimum_jacobian
        self.sweeps = sweeps
        self.amplitude_limit = amplitude_limit
        self.checkpoint_updates = checkpoint_updates
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

    def forward(self, coarse: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        if coarse.ndim != 4 or coarse.shape[1:] != (self.coarse_side, self.coarse_side, 2):
            raise ValueError("coarse must have shape (B,coarse_side,coarse_side,2)")
        if coarse.dtype not in (torch.float32, torch.float64):
            raise ValueError("this topology-preserving implementation requires float32/float64")
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
            desired = baseline + _StreamedSineDisplacement.apply(
                amplitude, self.directions.to(dtype=mapped.dtype),
                self.cycles, self.windows, side,
            )
            for _ in range(self.sweeps):
                if self.mechanism == "colored":
                    delta = desired - mapped
                    span = 2.0 / (side - 1)
                    logits = torch.atanh((delta[:, 1:-1, 1:-1] / span).clamp(-0.95, 0.95))
                    floor = mapped.new_full((mapped.shape[0],),
                                            self.minimum_jacobian / (side - 1) ** 2)
                    def apply_color(current: torch.Tensor, raw: torch.Tensor,
                                    active_layer=layer, active_floor=floor) -> torch.Tensor:
                        return active_layer(current, raw, area_floor=active_floor)
                    mapped = (
                        checkpoint(apply_color, mapped, logits, use_reentrant=False)
                        if self.checkpoint_updates and torch.is_grad_enabled() else
                        apply_color(mapped, logits)
                    )
                else:
                    assigned = torch.zeros(side * side, device=mapped.device, dtype=torch.bool)
                    for patch_pass in layer.passes:
                        ids = patch_pass.interior_ids
                        new = ~assigned[ids]
                        delta = (desired - mapped).reshape(mapped.shape[0], -1, 2)[:, ids]
                        raw = torch.where(new[None, :, None], delta, torch.zeros_like(delta))
                        span = patch_pass.raw_span * patch_pass.patch_cells / (side - 1)
                        logits = torch.atanh((raw / span).clamp(-0.95, 0.95))
                        def apply_patch(current: torch.Tensor, raw: torch.Tensor,
                                        active_pass=patch_pass) -> torch.Tensor:
                            return active_pass(current, raw)
                        mapped = (
                            checkpoint(apply_patch, mapped, logits, use_reentrant=False)
                            if self.checkpoint_updates and torch.is_grad_enabled() else
                            apply_patch(mapped, logits)
                        )
                        assigned[ids] = True
        return mapped
