"""Coarse patch motion followed by exact refinement and fine P1 updates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .colored_vertex_relaxation import SafeColoredVertexRelaxation
from .forward_p1_pyramid import ForwardP1Pyramid, exact_dyadic_p1_refine
from .patch_field import ResidualStaggeredPatchP1Layer


def certify_p1_or_identity(candidate: torch.Tensor,
                           identity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter actual binary-float P1 coordinates with conservative orientation tests.

    The 16u term bounds two rounded edge differences, their product, and
    the final subtraction under IEEE arithmetic.  Coordinates are restricted
    to [-2,2], so the 64*tiny guard also covers subnormal intermediate terms
    (including flush-to-zero).  A failed sample
    returns the exact identity, and its gradient is zero.  The Boolean branch
    is nondifferentiable at the acceptance boundary, as intended.
    """
    if candidate.ndim != 4 or candidate.shape[-1] != 2 or (
        candidate.shape[1] != candidate.shape[2]
    ) or identity.shape not in (candidate.shape, (1, *candidate.shape[1:])):
        raise ValueError("candidate/identity must have matching square P1 grids")
    if candidate.dtype not in (torch.float32, torch.float64) or (
        candidate.dtype != identity.dtype or candidate.device != identity.device
    ):
        raise ValueError("candidate and identity must share device and float dtype")
    reference = identity.expand_as(candidate)

    def positive_triangle(a: torch.Tensor, b: torch.Tensor,
                          c: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ba = b - a
            ca = c - a
            left = ba[..., 0] * ca[..., 1]
            right = ba[..., 1] * ca[..., 0]
            determinant = left - right
            finfo = torch.finfo(candidate.dtype)
            error_bound = 16 * finfo.eps * (left.abs() + right.abs()) + 64 * finfo.tiny
            return determinant > error_bound

    with torch.no_grad():
        sw = candidate[:, :-1, :-1]
        se = candidate[:, :-1, 1:]
        nw = candidate[:, 1:, :-1]
        ne = candidate[:, 1:, 1:]
        valid = (torch.isfinite(candidate).all(dim=(1, 2, 3))
                 & (candidate.abs() <= 2).all(dim=(1, 2, 3))
                 & torch.eq(candidate[:, 0], reference[:, 0]).all(dim=(1, 2))
                 & torch.eq(candidate[:, -1], reference[:, -1]).all(dim=(1, 2))
                 & torch.eq(candidate[:, :, 0], reference[:, :, 0]).all(dim=(1, 2))
                 & torch.eq(candidate[:, :, -1], reference[:, :, -1]).all(dim=(1, 2))
                 & positive_triangle(sw, se, ne).all(dim=(1, 2))
                 & positive_triangle(sw, ne, nw).all(dim=(1, 2)))
    return torch.where(valid[:, None, None, None], candidate, reference), valid


class CertifiedForwardP1Pyramid(nn.Module):
    """Pyramid with a conservative check on the actual final P1 coordinates."""

    def __init__(self, seed_side: int, final_side: int, *, seed_passes: int,
                 safety_fraction: float = 0.85, raw_span: float = 2.0,
                 minimum_jacobian: float = 0.05,
                 checkpoint_passes: bool = False,
                 compute_dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        if compute_dtype not in (torch.float32, torch.float64):
            raise ValueError("compute_dtype must be float32 or float64")
        exact_limit = 2 ** (23 if compute_dtype == torch.float32 else 52)
        if final_side - 1 > exact_limit or (final_side - 1) & (final_side - 2):
            raise ValueError("final side must be 2^k+1 with exact identity spacing")
        self.core = ForwardP1Pyramid(
            seed_side, final_side, seed_passes=seed_passes,
            safety_fraction=safety_fraction, raw_span=raw_span,
            minimum_jacobian=minimum_jacobian,
            checkpoint_passes=checkpoint_passes,
        )
        self.level_sides = self.core.level_sides
        self.compute_dtype = compute_dtype
        axis = torch.arange(final_side, dtype=compute_dtype) / (final_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("identity", torch.stack((xx, yy), dim=-1)[None],
                             persistent=False)

    def forward(self, seed_logits: Sequence[torch.Tensor],
                level_logits: Sequence[torch.Tensor]) -> torch.Tensor:
        values = tuple(seed_logits) + tuple(level_logits)
        if not values:
            raise ValueError("at least one latent tensor is required")
        if any(z.device != self.identity.device for z in values):
            raise ValueError("move layer and latents to the same device")
        result = self.core(
            [z.to(self.compute_dtype) for z in seed_logits],
            [z.to(self.compute_dtype) for z in level_logits],
        )
        return certify_p1_or_identity(result, self.identity)[0]


class CoarsePatchFineVertexP1Layer(nn.Module):
    """Decode two latent fields to one fixed fine-grid P1 map.

    The positive-face invariant is exact-arithmetic; float64 and an absolute
    area floor reduce, but do not formally eliminate, IEEE rounding risk.
    The returned tensor retains ``compute_dtype`` so topology is checked on
    the actual output coordinates rather than a lower-precision cast.
    """

    def __init__(
        self,
        coarse_side: int,
        fine_side: int,
        *,
        coarse_patch_cells: int,
        coarse_cycles: int = 2,
        minimum_jacobian: float = 0.05,
        compute_dtype: torch.dtype = torch.float64,
        certify_output: bool = True,
        checkpoint_fine: bool = False,
    ) -> None:
        super().__init__()
        if coarse_side < 3 or (coarse_side - 1) & (coarse_side - 2):
            raise ValueError("coarse side must be 2^k+1")
        if fine_side != 4 * (coarse_side - 1) + 1:
            raise ValueError("fine grid must be two dyadic refinements of coarse grid")
        if not 0 < minimum_jacobian < 1:
            raise ValueError("minimum_jacobian must lie in (0,1)")
        if compute_dtype not in (torch.float32, torch.float64):
            raise ValueError("compute_dtype must be float32 or float64")
        exact_limit = 2 ** (23 if compute_dtype == torch.float32 else 52)
        if fine_side - 1 > exact_limit:
            raise ValueError("fine identity spacing is not representable")
        self.coarse_side = coarse_side
        self.fine_side = fine_side
        self.minimum_jacobian = minimum_jacobian
        self.compute_dtype = compute_dtype
        self.certify_output = certify_output
        self.checkpoint_fine = checkpoint_fine
        self.coarse = ResidualStaggeredPatchP1Layer(
            coarse_side, coarse_patch_cells, cycles=coarse_cycles,
            minimum_jacobian=minimum_jacobian,
        )
        self.fine = SafeColoredVertexRelaxation(
            fine_side, safety_fraction=0.85, motion_mode="radial",
            raw_span=2.0, floor_fraction=0.0,
        )
        axis = torch.arange(coarse_side, dtype=compute_dtype) / (coarse_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        identity = torch.stack((xx, yy), dim=-1)[None]
        self.register_buffer("identity", identity, persistent=False)
        fine_axis = torch.arange(fine_side, dtype=compute_dtype) / (fine_side - 1)
        fine_yy, fine_xx = torch.meshgrid(fine_axis, fine_axis, indexing="ij")
        self.register_buffer("fine_identity", torch.stack((fine_xx, fine_yy), dim=-1)[None],
                             persistent=False)

    def forward(self, coarse_latent: torch.Tensor,
                fine_latent: torch.Tensor) -> torch.Tensor:
        c, f = self.coarse_side, self.fine_side
        if coarse_latent.ndim != 4 or coarse_latent.shape[1:] != (c - 2, c - 2, 2):
            raise ValueError("coarse latent has wrong shape")
        if fine_latent.shape != (coarse_latent.shape[0], f - 2, f - 2, 2):
            raise ValueError("fine latent has wrong shape")
        if (coarse_latent.device != self.identity.device or
                fine_latent.device != self.identity.device):
            raise ValueError("move the layer and both latents to the same device")
        if coarse_latent.dtype not in (torch.float32, torch.float64) or (
            fine_latent.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("latents must be float32 or float64")
        batch = coarse_latent.shape[0]
        coarse_map = self.coarse(
            self.identity.expand(batch, -1, -1, -1),
            coarse_latent.to(self.compute_dtype),
        )
        fine_base = exact_dyadic_p1_refine(exact_dyadic_p1_refine(coarse_map))
        floor = fine_base.new_full((batch,),
                                   self.minimum_jacobian / (f - 1) ** 2)
        fine_logit = fine_latent.to(self.compute_dtype)
        if self.checkpoint_fine and torch.is_grad_enabled():
            result = checkpoint(
                lambda base, latent: self.fine(base, latent, area_floor=floor),
                fine_base, fine_logit, use_reentrant=False,
            )
        else:
            result = self.fine(fine_base, fine_logit, area_floor=floor)
        if self.certify_output:
            result, _ = certify_p1_or_identity(result,
                                               self.fine_identity.expand(batch, -1, -1, -1))
        return result


class HybridPatchSeedVertexP1Pyramid(nn.Module):
    """Long-range patch seed plus new-vertex-only dyadic P1 refinements.

    F2 moves a genuinely coarse seed map. Each F1 refinement exactly
    prolongs the previous P1 map, then moves only vertices new to that level.
    Therefore the final output is one P1 map on the original fixed grid;
    it is not a resampled composition. A conservative final-coordinate
    orientation check handles finite-precision violations by returning the
    exact identity for the affected sample.
    """

    def __init__(
        self, seed_side: int, final_side: int, *,
        patch_cells: int = 4, seed_cycles: int = 4,
        seed_steps: int = 1,
        minimum_jacobian: float = 0.05,
        checkpoint_levels: bool = False,
        compute_dtype: torch.dtype = torch.float64,
        certify_output: bool = True,
    ) -> None:
        super().__init__()
        if compute_dtype not in (torch.float32, torch.float64):
            raise ValueError("compute_dtype must be float32 or float64")
        exact_limit = 2 ** (23 if compute_dtype == torch.float32 else 52)
        if final_side - 1 > exact_limit:
            raise ValueError("final identity spacing is not representable")
        if not 0 < minimum_jacobian < 1:
            raise ValueError("minimum_jacobian must lie in (0,1)")
        if seed_steps < 1:
            raise ValueError("seed_steps must be positive")
        self.seed_side = seed_side
        self.final_side = final_side
        self.seed_steps = seed_steps
        self.compute_dtype = compute_dtype
        self.certify_output = certify_output
        self.seed_layer = ResidualStaggeredPatchP1Layer(
            seed_side, patch_cells, cycles=seed_cycles,
            minimum_jacobian=minimum_jacobian,
        )
        self.refinement = ForwardP1Pyramid(
            seed_side, final_side, seed_passes=0,
            minimum_jacobian=minimum_jacobian,
            checkpoint_passes=checkpoint_levels,
        )
        self.level_sides = self.refinement.level_sides
        seed_axis = torch.arange(seed_side, dtype=compute_dtype) / (seed_side - 1)
        seed_y, seed_x = torch.meshgrid(seed_axis, seed_axis, indexing="ij")
        self.register_buffer(
            "seed_identity", torch.stack((seed_x, seed_y), dim=-1)[None],
            persistent=False,
        )
        fine_axis = torch.arange(final_side, dtype=compute_dtype) / (final_side - 1)
        fine_y, fine_x = torch.meshgrid(fine_axis, fine_axis, indexing="ij")
        self.register_buffer(
            "final_identity", torch.stack((fine_x, fine_y), dim=-1)[None],
            persistent=False,
        )

    def forward(
        self, seed_latent: torch.Tensor | Sequence[torch.Tensor],
        level_latents: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        seed_fields = (
            (seed_latent,) if isinstance(seed_latent, torch.Tensor)
            else tuple(seed_latent)
        )
        if len(seed_fields) != self.seed_steps:
            raise ValueError("wrong number of seed endpoint fields")
        first = seed_fields[0]
        if first.ndim != 4 or first.shape[1:] != (
            self.seed_side - 2, self.seed_side - 2, 2
        ) or first.shape[0] < 1 or any(z.shape != first.shape for z in seed_fields):
            raise ValueError("seed_latent has wrong shape")
        if len(level_latents) != len(self.level_sides):
            raise ValueError("wrong number of refinement latent arrays")
        values = (*seed_fields, *level_latents)
        if any(z.device != self.seed_identity.device for z in values):
            raise ValueError("move layer and all latents to the same device")
        if any(z.dtype not in (torch.float32, torch.float64) for z in values):
            raise ValueError("latents must be floating point")
        batch = first.shape[0]
        coarse_map = self.seed_identity.expand(batch, -1, -1, -1)
        for field in seed_fields:
            coarse_map = self.seed_layer(
                coarse_map, field.to(self.compute_dtype),
            )
        result = self.refinement.forward_from(
            coarse_map,
            [z.to(self.compute_dtype) for z in level_latents],
            start_index=0,
        )
        if self.certify_output:
            result, _ = certify_p1_or_identity(result, self.final_identity)
        return result
