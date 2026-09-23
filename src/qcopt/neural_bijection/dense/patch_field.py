"""Simultaneous multi-vertex patch fields with an exact face-area bound."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .forward_p1_pyramid import exact_dyadic_p1_refine


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class SafePatchFieldPass(nn.Module):
    """Update all patch interiors at once while keeping every fixed-grid face positive.

    A pass contains nonoverlapping ``patch_cells`` x ``patch_cells`` patches.
    Patch boundaries stay fixed and so outside faces cannot change. For each
    patch, the common latent-field amplitude is chosen from the exact quadratic
    area polynomial of *every* patch face. The patch may contain many moving
    vertices in the same triangle, unlike a four-color single-vertex pass.

    Exact-arithmetic guarantee assumes a positive-area input P1 map with
    pointwise identity boundary. ``minimum_jacobian`` is a normalized double-
    area floor relative to the regular source triangle, independent of depth.
    """

    def __init__(
        self,
        side: int,
        patch_cells: int,
        *,
        offset_row: int = 0,
        offset_column: int = 0,
        raw_span: float = 0.5,
        safety_fraction: float = 0.85,
        minimum_jacobian: float | None = 0.05,
    ) -> None:
        super().__init__()
        if side < 3 or patch_cells < 2 or (side - 1) % patch_cells:
            raise ValueError("patch_cells must divide side-1 and be at least 2")
        if not 0 <= offset_row < patch_cells or not 0 <= offset_column < patch_cells:
            raise ValueError("patch offsets must be in [0,patch_cells)")
        if not math.isfinite(raw_span) or raw_span <= 0:
            raise ValueError("raw_span must be finite and positive")
        if not 0 < safety_fraction < 1 or (
            minimum_jacobian is not None and not 0 < minimum_jacobian < 1
        ):
            raise ValueError("safety_fraction must be in (0,1); minimum_jacobian in (0,1) or None")
        self.side = side
        self.patch_cells = patch_cells
        self.offset_row = offset_row
        self.offset_column = offset_column
        self.raw_span = raw_span
        self.safety_fraction = safety_fraction
        self.minimum_jacobian = minimum_jacobian
        starts = [
            (row, column)
            for row in range(offset_row, side - patch_cells, patch_cells)
            for column in range(offset_column, side - patch_cells, patch_cells)
        ]
        if not starts:
            raise ValueError("offset leaves no complete patch")
        ids = np.asarray([
            [[(r + dr) * side + (c + dc) for dc in range(patch_cells + 1)]
             for dr in range(patch_cells + 1)]
            for r, c in starts
        ], dtype=np.int64)
        interior = ids[:, 1:-1, 1:-1]
        interior_rows, interior_columns = np.divmod(interior, side)
        latent_ids = (interior_rows - 1) * (side - 2) + (interior_columns - 1)
        if len(set(interior.reshape(-1))) != interior.size:
            raise AssertionError("one pass must have disjoint patch interiors")
        self.register_buffer("patch_ids", torch.from_numpy(ids), persistent=False)
        self.register_buffer("interior_ids", torch.from_numpy(interior.reshape(-1)), persistent=False)
        self.register_buffer("latent_ids", torch.from_numpy(latent_ids), persistent=False)

    def forward(self, base: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        side, cells = self.side, self.patch_cells
        if base.ndim != 4 or base.shape[1:] != (side, side, 2):
            raise ValueError("base must have shape (batch,side,side,2)")
        full_shape = (base.shape[0], side - 2, side - 2, 2)
        compact_shape = (base.shape[0], self.interior_ids.numel(), 2)
        if logits.shape not in (full_shape, compact_shape):
            raise ValueError("logits need a full interior field or one vector per active patch interior")
        if logits.dtype != base.dtype or logits.device != base.device:
            raise ValueError("base and logits must match dtype/device")
        batch = base.shape[0]
        patch_count = self.patch_ids.shape[0]
        current = base.reshape(batch, side * side, 2)
        patch = current[:, self.patch_ids]
        selected = (
            logits.reshape(batch, -1, 2)[:, self.latent_ids]
            if logits.shape == full_shape else logits
        )
        raw = self.raw_span * cells / (side - 1) * torch.tanh(selected).reshape(
            batch, patch_count, cells - 1, cells - 1, 2,
        )
        displacement = torch.zeros_like(patch)
        displacement[:, :, 1:-1, 1:-1] = raw

        a, b = patch[:, :, :-1, :-1], patch[:, :, :-1, 1:]
        c, d = patch[:, :, 1:, 1:], patch[:, :, 1:, :-1]
        da, db = displacement[:, :, :-1, :-1], displacement[:, :, :-1, 1:]
        dc, dd = displacement[:, :, 1:, 1:], displacement[:, :, 1:, :-1]

        def face_terms(
            p: torch.Tensor, q: torch.Tensor, r: torch.Tensor,
            dp: torch.Tensor, dq: torch.Tensor, dr: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            e, f = q - p, r - p
            de, df = dq - dp, dr - dp
            area = _cross(e, f)
            linear = _cross(de, f) + _cross(e, df)
            quadratic = _cross(de, df)
            # For 0 <= s <= 1, positive linear/quadratic terms never need
            # a safety budget. The adverse-only bound remains rigorous:
            # A(s) >= A(0) - s*((-L)_+ + (-Q)_+).
            return area, (-linear).clamp_min(0) + (-quadratic).clamp_min(0)

        low_area, low_bound = face_terms(a, b, c, da, db, dc)
        up_area, up_bound = face_terms(a, c, d, da, dc, dd)
        areas = torch.stack((low_area, up_area), dim=-1).reshape(batch, patch_count, -1)
        bounds = torch.stack((low_bound, up_bound), dim=-1).reshape(batch, patch_count, -1)
        allowance = self.safety_fraction * areas
        if self.minimum_jacobian is not None:
            floor = self.minimum_jacobian / (side - 1) ** 2
            allowance = torch.minimum(allowance, (areas - floor).clamp_min(0))
        # `tiny` (about 1e-38 in float32) makes d(scale)/d(area) explode on
        # a face exactly at the floor with zero adverse proposal. Such faces
        # occur at fixed patch corners after earlier staggered passes.
        guard = math.sqrt(torch.finfo(base.dtype).eps) / (side - 1) ** 2
        guarded_scales = allowance / torch.maximum(
            torch.maximum(bounds, allowance), bounds.new_tensor(guard),
        )
        # C=0 cannot lower a face's area at any s in [0,1]. In particular,
        # an unmoving face exactly on the floor must not freeze its patch.
        face_scales = torch.where(bounds > 0, guarded_scales, torch.ones_like(bounds))
        patch_scales = face_scales.amin(dim=-1)
        updated = patch[:, :, 1:-1, 1:-1] + patch_scales[:, :, None, None, None] * raw
        return current.index_copy(1, self.interior_ids, updated.reshape(batch, -1, 2)).reshape(
            batch, side, side, 2,
        )


class StaggeredPatchP1Layer(nn.Module):
    """Four shifted patch passes so every strict interior vertex can move."""

    def __init__(self, side: int, patch_cells: int = 8, **kwargs: float | None) -> None:
        super().__init__()
        if patch_cells % 2:
            raise ValueError("patch_cells must be even for the staggered schedule")
        shift = patch_cells // 2
        self.passes = nn.ModuleList(
            SafePatchFieldPass(side, patch_cells, offset_row=dr, offset_column=dc, **kwargs)
            for dr, dc in ((0, 0), (shift, 0), (0, shift), (shift, shift))
        )

    def forward(self, base: torch.Tensor, logits: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(logits) != len(self.passes):
            raise ValueError("one latent field per patch pass is required")
        current = base
        for layer, latent in zip(self.passes, logits):
            current = layer(current, latent)
        return current


class TiedStaggeredPatchP1Layer(nn.Module):
    """Reuse one bounded per-vertex latent across shifted patch passes.

    Each active visit receives an equal fraction of the vertex's proposed
    displacement. This avoids separate, mutually cancelling latent fields
    while retaining the area-bound guarantee of every constituent pass.
    """

    def __init__(self, side: int, patch_cells: int = 8, *, cycles: int = 1,
                 **kwargs: float | None) -> None:
        super().__init__()
        if cycles < 1:
            raise ValueError("cycles must be positive")
        self.side = side
        self.cycles = cycles
        self.staggered = StaggeredPatchP1Layer(side, patch_cells, **kwargs)
        visits = np.zeros((side - 2) ** 2, dtype=np.float32)
        for patch_pass in self.staggered.passes:
            np.add.at(visits, patch_pass.latent_ids.reshape(-1).numpy(), cycles)
        if np.any(visits == 0):
            raise AssertionError("every interior vertex needs at least one visit")
        self.register_buffer("visit_counts", torch.from_numpy(visits), persistent=False)

    def forward(self, base: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape != (base.shape[0], self.side - 2, self.side - 2, 2):
            raise ValueError("latent must give one vector per interior vertex")
        if latent.dtype != base.dtype or latent.device != base.device:
            raise ValueError("base and latent must match dtype/device")
        eps = 4 * torch.finfo(latent.dtype).eps
        bounded = torch.tanh(latent).clamp(-1 + eps, 1 - eps)
        flat = bounded.reshape(base.shape[0], -1, 2)
        counts = self.visit_counts.to(dtype=latent.dtype)
        current = base
        for _ in range(self.cycles):
            for patch_pass in self.staggered.passes:
                ids = patch_pass.latent_ids.reshape(-1)
                field = torch.atanh(flat[:, ids] / counts[ids][None, :, None])
                current = patch_pass(current, field)
        return current


class ResidualStaggeredPatchP1Layer(nn.Module):
    """One latent endpoint field, approached by repeated safe patch passes.

    The latent endpoint itself need not be a homeomorphism; only the sequence
    of accepted patch updates is returned. Every pass is topology-preserving
    under the constituent pass's exact-arithmetic assumptions.
    """

    def __init__(self, side: int, patch_cells: int = 8, *, cycles: int = 1,
                 **kwargs: float | None) -> None:
        super().__init__()
        if cycles < 1:
            raise ValueError("cycles must be positive")
        self.side = side
        self.cycles = cycles
        self.staggered = StaggeredPatchP1Layer(side, patch_cells, **kwargs)

    def forward(self, base: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        side = self.side
        if base.ndim != 4 or base.shape[1:] != (side, side, 2):
            raise ValueError("base must have shape (batch,side,side,2)")
        if latent.shape != (base.shape[0], side - 2, side - 2, 2):
            raise ValueError("latent must give one vector per interior vertex")
        if latent.dtype != base.dtype or latent.device != base.device:
            raise ValueError("base and latent must match dtype/device")
        span = self.staggered.passes[0].raw_span * self.staggered.passes[0].patch_cells / (side - 1)
        delta = torch.zeros_like(base)
        delta[:, 1:-1, 1:-1] = span * torch.tanh(latent)
        eps = 4 * torch.finfo(base.dtype).eps
        current = base
        for cycle in range(self.cycles):
            stage_target = base + ((cycle + 1) / self.cycles) * delta
            flat_target = stage_target.reshape(base.shape[0], -1, 2)
            for patch_pass in self.staggered.passes:
                ids = patch_pass.interior_ids
                desired = (flat_target[:, ids] - current.reshape(base.shape[0], -1, 2)[:, ids]) / span
                field = torch.atanh(desired.clamp(-1 + eps, 1 - eps))
                current = patch_pass(current, field)
        return current


class ForwardPatchP1Pyramid(nn.Module):
    """Coarse-to-fine fixed-grid P1 decoder using simultaneous patch fields.

    This is a distinct refinement primitive from vertex coloring. At each
    level, the current P1 map is reproduced exactly on the fine triangulation,
    then four shifted patch passes update many adjacent vertices together.
    """

    def __init__(
        self,
        seed_side: int,
        final_side: int,
        *,
        patch_cells: int = 8,
        seed_cycles: int = 1,
        level_cycles: int = 1,
        minimum_jacobian: float | None = 0.05,
        raw_span: float = 0.5,
    ) -> None:
        super().__init__()
        if seed_cycles < 1 or level_cycles < 1 or final_side < seed_side:
            raise ValueError("invalid cycles or final side")
        sides = []
        side = seed_side
        while side < final_side:
            side = 2 * side - 1
            sides.append(side)
        if side != final_side:
            raise ValueError("final_side must be reachable by dyadic refinement")
        self.seed_side = seed_side
        self.final_side = final_side
        self.level_sides = tuple(sides)
        self.seed_cycles = seed_cycles
        self.level_cycles = level_cycles
        options = {"minimum_jacobian": minimum_jacobian, "raw_span": raw_span}
        self.seed_layer = StaggeredPatchP1Layer(seed_side, patch_cells, **options)
        self.level_layers = nn.ModuleList(
            StaggeredPatchP1Layer(n, patch_cells, **options) for n in sides
        )

    def forward(
        self,
        seed_logits: tuple[torch.Tensor, ...],
        level_logits: tuple[tuple[torch.Tensor, ...], ...],
    ) -> torch.Tensor:
        if len(seed_logits) != 4 * self.seed_cycles or len(level_logits) != len(self.level_sides):
            raise ValueError("wrong number of seed or refinement latent fields")
        first = seed_logits[0]
        if first.ndim not in (3, 4) or first.shape[0] < 1 or first.shape[-1] != 2:
            raise ValueError("latents need full fields or compact active-vertex arrays")
        axis = torch.arange(self.seed_side, device=first.device, dtype=first.dtype) / (self.seed_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        current = torch.stack((xx, yy), dim=-1)[None].expand(first.shape[0], -1, -1, -1)
        for cycle in range(self.seed_cycles):
            current = self.seed_layer(current, seed_logits[4 * cycle:4 * (cycle + 1)])
        return self.forward_from(current, level_logits, start_index=0)

    def forward_from(
        self,
        current: torch.Tensor,
        level_logits: tuple[tuple[torch.Tensor, ...], ...],
        *,
        start_index: int,
    ) -> torch.Tensor:
        """Continue the pyramid after an exact P1 modification on one level."""
        if not 0 <= start_index <= len(self.level_sides):
            raise ValueError("invalid refinement start_index")
        expected_side = self.seed_side if start_index == 0 else self.level_sides[start_index - 1]
        if current.ndim != 4 or current.shape[1:] != (expected_side, expected_side, 2):
            raise ValueError("current has the wrong starting grid side")
        if len(level_logits) != len(self.level_sides) - start_index:
            raise ValueError("incorrect number of remaining refinement latent groups")
        for n, layer, fields in zip(
            self.level_sides[start_index:],
            self.level_layers[start_index:],
            level_logits,
        ):
            valid_shapes = tuple(
                ((current.shape[0], n - 2, n - 2, 2),
                 (current.shape[0], patch_pass.interior_ids.numel(), 2))
                for _ in range(self.level_cycles) for patch_pass in layer.passes
            )
            if len(fields) != 4 * self.level_cycles or any(
                field.shape not in shapes for field, shapes in zip(fields, valid_shapes)
            ):
                raise ValueError("refinement latent fields have wrong count or shape")
            current = exact_dyadic_p1_refine(current)
            for cycle in range(self.level_cycles):
                current = layer(current, fields[4 * cycle:4 * (cycle + 1)])
        return current


class ResidualPatchP1Pyramid(nn.Module):
    """One target residual field and one staggered patch cycle per level.

    Each field specifies an endpoint proposal, while the four nonoverlapping
    shifted patch passes approach it without solving a global linear system.
    The returned function is one P1 map on the final fixed triangulation.
    """

    def __init__(self, seed_side: int, final_side: int, *, patch_cells: int = 8,
                 seed_passes: int = 4, minimum_jacobian: float | None = 0.05,
                 raw_span: float = 0.5) -> None:
        super().__init__()
        if seed_passes < 1 or final_side < seed_side:
            raise ValueError("need positive seed passes and final_side >= seed_side")
        sides = []
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
        options = dict(cycles=1, minimum_jacobian=minimum_jacobian,
                       raw_span=raw_span)
        self.seed_layer = ResidualStaggeredPatchP1Layer(
            seed_side, patch_cells, **options,
        )
        self.level_layers = nn.ModuleList(
            ResidualStaggeredPatchP1Layer(n, patch_cells, **options)
            for n in sides
        )

    def forward(self, seed_logits: list[torch.Tensor] | tuple[torch.Tensor, ...],
                level_logits: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(seed_logits) != self.seed_passes or len(level_logits) != len(self.level_sides):
            raise ValueError("incorrect number of seed or level latent fields")
        first = seed_logits[0]
        if first.ndim != 4 or first.shape[1:] != (self.seed_side - 2, self.seed_side - 2, 2):
            raise ValueError("seed latent must be a full interior field")
        axis = torch.arange(self.seed_side, dtype=first.dtype,
                            device=first.device) / (self.seed_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        current = torch.stack((xx, yy), dim=-1)[None].expand(first.shape[0], -1, -1, -1)
        for latent in seed_logits:
            current = self.seed_layer(current, latent)
        for n, layer, latent in zip(self.level_sides, self.level_layers, level_logits):
            if latent.shape != (first.shape[0], n - 2, n - 2, 2):
                raise ValueError("level latent has wrong shape")
            current = exact_dyadic_p1_refine(current)
            current = layer(current, latent)
        return current
