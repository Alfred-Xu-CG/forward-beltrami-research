"""Geometry of a tensor-product Q1 map on one declared rectangular grid."""

from __future__ import annotations

import math

import torch

from .colored_vertex_relaxation import SafeColoredVertexRelaxation
from .patch_field import SafePatchFieldPass


def _cross(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def q1_corner_determinants(vertices: torch.Tensor) -> torch.Tensor:
    """Return unnormalized SW, SE, NE, NW Jacobians of every Q1 cell.

    ``vertices`` has shape ``(batch, rows, columns, 2)``. Rows increase in
    physical y, columns in physical x; the output shape is
    ``(batch, rows-1, columns-1, 4)``. Divide by each source cell's positive
    physical area to obtain dimensionless Jacobian determinants.
    """
    if vertices.ndim != 4 or vertices.shape[-1] != 2 or min(vertices.shape[1:3]) < 2:
        raise ValueError("vertices must have shape (batch,rows>=2,columns>=2,2)")
    a = vertices[:, :-1, :-1]
    b = vertices[:, :-1, 1:]
    c = vertices[:, 1:, 1:]
    d = vertices[:, 1:, :-1]
    return torch.stack((
        _cross(b - a, d - a),
        _cross(b - a, c - b),
        _cross(c - d, c - b),
        _cross(c - d, d - a),
    ), dim=-1)


def q1_dyadic_refine(vertices: torch.Tensor) -> torch.Tensor:
    """Represent exactly the same Q1 function on four child cells per cell."""
    if vertices.ndim != 4 or vertices.shape[-1] != 2 or min(vertices.shape[1:3]) < 2:
        raise ValueError("vertices must have shape (batch,rows>=2,columns>=2,2)")
    batch, rows, columns, _ = vertices.shape
    fine = vertices.new_empty((batch, 2 * rows - 1, 2 * columns - 1, 2))
    fine[:, ::2, ::2] = vertices
    fine[:, ::2, 1::2] = (vertices[:, :, :-1] + vertices[:, :, 1:]) / 2
    fine[:, 1::2, ::2] = (vertices[:, :-1] + vertices[:, 1:]) / 2
    fine[:, 1::2, 1::2] = (
        vertices[:, :-1, :-1] + vertices[:, :-1, 1:]
        + vertices[:, 1:, :-1] + vertices[:, 1:, 1:]
    ) / 4
    return fine


@torch.no_grad()
def validate_q1_map(
    vertices: torch.Tensor, boundary_reference: torch.Tensor, *,
    chunk_rows: int = 256,
) -> dict[str, bool | int | float]:
    """Audit actual coordinates, all Q1 corners, and an exact boundary.

    This is a numerical check of the supplied tensor, not a rounding-proof
    certificate. The caller must independently know that the reference
    boundary is a simple orientation-preserving polygon. Read a saved map
    back before invoking this when making an exported-output claim.
    """
    if vertices.shape != boundary_reference.shape or vertices.ndim != 4 or vertices.shape[-1] != 2:
        raise ValueError("vertices and boundary_reference must have matching (B,R,C,2) shape")
    if vertices.dtype != boundary_reference.dtype or vertices.device != boundary_reference.device:
        raise ValueError("vertices and boundary_reference must match dtype/device")
    if min(vertices.shape[1:3]) < 2 or chunk_rows < 1:
        raise ValueError("need at least one cell and positive chunk_rows")

    rows = vertices.shape[1]
    nonfinite_coordinates = 0
    for start in range(0, rows, chunk_rows):
        nonfinite_coordinates += int((~torch.isfinite(vertices[:, start:start + chunk_rows])).sum())

    boundary_differences = (
        vertices[:, 0] - boundary_reference[:, 0],
        vertices[:, -1] - boundary_reference[:, -1],
        vertices[:, :, 0] - boundary_reference[:, :, 0],
        vertices[:, :, -1] - boundary_reference[:, :, -1],
    )
    boundary_max_error = max(
        float(value.abs().amax()) if bool(torch.isfinite(value).all()) else float("inf")
        for value in boundary_differences
    )
    nonpositive = 0
    nonfinite_corners = 0
    corner_min = float("inf")
    for start in range(0, rows - 1, chunk_rows):
        corners = q1_corner_determinants(
            vertices[:, start:min(start + chunk_rows + 1, rows)],
        )
        finite = torch.isfinite(corners)
        nonfinite_corners += int((~finite).sum())
        nonpositive += int((finite & (corners <= 0)).sum())
        if bool(finite.any()):
            corner_min = min(corner_min, float(corners[finite].amin()))
    valid = (
        nonfinite_coordinates == 0 and nonfinite_corners == 0
        and nonpositive == 0 and boundary_max_error == 0.0
        and bool(torch.isfinite(boundary_reference).all())
    )
    return {
        "valid": valid,
        "corner_min": corner_min,
        "nonpositive_corners": nonpositive,
        "nonfinite_corners": nonfinite_corners,
        "nonfinite_coordinates": nonfinite_coordinates,
        "boundary_max_error": boundary_max_error,
    }


def _q1_incident_opposite_offsets(side: int) -> torch.Tensor:
    """The 12 oriented corner triangles containing a strict interior vertex."""
    center = (1, 1)
    offsets: list[tuple[int, int]] = []
    for row in (0, 1):
        for column in (0, 1):
            a = (row, column)
            b = (row, column + 1)
            c = (row + 1, column + 1)
            d = (row + 1, column)
            for triangle in ((a, b, d), (a, b, c), (d, b, c), (a, c, d)):
                if center not in triangle:
                    continue
                index = triangle.index(center)
                start = triangle[(index + 1) % 3]
                end = triangle[(index + 2) % 3]
                offsets.append((
                    (start[0] - 1) * side + start[1] - 1,
                    (end[0] - 1) * side + end[1] - 1,
                ))
    if len(offsets) != 12:
        raise AssertionError("every strict interior vertex has 12 Q1 corner constraints")
    return torch.tensor(offsets, dtype=torch.long)


class SafeColoredQ1Relaxation(SafeColoredVertexRelaxation):
    """Four-color F1-D: protect all Q1 corner determinants on affected cells.

    Exact-arithmetic guarantee assumes the input already has positive corner
    determinants and a fixed, injective boundary. The inherited radial step
    uses a separate explicit safe scale for each active interior vertex.
    ``minimum_jacobian`` is an optional normalized source-cell floor; the
    floor guarantee additionally assumes the input meets that floor.
    """

    def __init__(
        self, side: int, *, safety_fraction: float = 0.75,
        raw_span: float = 2.0, minimum_jacobian: float | None = 0.05,
        checkpoint_colors: bool = False,
    ) -> None:
        if minimum_jacobian is not None and not 0.0 < minimum_jacobian < 1.0:
            raise ValueError("minimum_jacobian must be in (0,1) or None")
        super().__init__(
            side, safety_fraction=safety_fraction, motion_mode="radial",
            raw_span=raw_span, index_mode="generated",
            checkpoint_colors=checkpoint_colors,
        )
        self.minimum_jacobian = minimum_jacobian
        self._opposite_offsets = _q1_incident_opposite_offsets(side)

    def forward(
        self, base: torch.Tensor, logits: torch.Tensor, *,
        area_floor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if area_floor is None and self.minimum_jacobian is not None:
            area_floor = base.new_full(
                (base.shape[0],), self.minimum_jacobian / (self.side - 1) ** 2,
            )
        return super().forward(base, logits, area_floor=area_floor)


class SafePatchQ1Pass(SafePatchFieldPass):
    """F2-D: simultaneous patch motion with all four corner constraints.

    Patch perimeter vertices remain fixed during this pass, so only cells
    inside a patch can change. A shared scale for each nonconflicting patch
    bounds the complete quadratic determinant path, not just its endpoint.
    The exact-arithmetic guarantee assumes strictly positive input corners
    and a fixed bijective outer boundary. An optional normalized area floor
    is preserved only if the input already meets that floor.
    """

    def forward(self, base: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        side, cells = self.side, self.patch_cells
        if base.ndim != 4 or base.shape[1:] != (side, side, 2):
            raise ValueError("base must have shape (batch,side,side,2)")
        batch = base.shape[0]
        full_shape = (batch, side - 2, side - 2, 2)
        compact_shape = (batch, self.interior_ids.numel(), 2)
        if logits.shape not in (full_shape, compact_shape):
            raise ValueError("logits need a full interior field or active patch interior array")
        if logits.dtype != base.dtype or logits.device != base.device:
            raise ValueError("base and logits must match dtype/device")

        current = base.reshape(batch, side * side, 2)
        patch = current[:, self.patch_ids]
        selected = (
            logits.reshape(batch, -1, 2)[:, self.latent_ids]
            if logits.shape == full_shape else logits
        )
        raw = self.raw_span * cells / (side - 1) * torch.tanh(selected).reshape(
            batch, self.patch_ids.shape[0], cells - 1, cells - 1, 2,
        )
        displacement = torch.zeros_like(patch)
        displacement[:, :, 1:-1, 1:-1] = raw

        a, b = patch[:, :, :-1, :-1], patch[:, :, :-1, 1:]
        c, d = patch[:, :, 1:, 1:], patch[:, :, 1:, :-1]
        da, db = displacement[:, :, :-1, :-1], displacement[:, :, :-1, 1:]
        dc, dd = displacement[:, :, 1:, 1:], displacement[:, :, 1:, :-1]

        def triangle_terms(
            p: torch.Tensor, q: torch.Tensor, r: torch.Tensor,
            dp: torch.Tensor, dq: torch.Tensor, dr: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            edge, other = q - p, r - p
            edge_delta, other_delta = dq - dp, dr - dp
            area = _cross(edge, other)
            linear = _cross(edge_delta, other) + _cross(edge, other_delta)
            quadratic = _cross(edge_delta, other_delta)
            adverse_bound = (-linear).clamp_min(0) + (-quadratic).clamp_min(0)
            return area, adverse_bound

        corner_terms = (
            triangle_terms(a, b, d, da, db, dd),
            triangle_terms(a, b, c, da, db, dc),
            triangle_terms(d, b, c, dd, db, dc),
            triangle_terms(a, c, d, da, dc, dd),
        )
        areas = torch.stack([terms[0] for terms in corner_terms], dim=-1).reshape(
            batch, self.patch_ids.shape[0], -1,
        )
        bounds = torch.stack([terms[1] for terms in corner_terms], dim=-1).reshape_as(areas)
        allowance = self.safety_fraction * areas
        if self.minimum_jacobian is not None:
            floor = self.minimum_jacobian / (side - 1) ** 2
            allowance = torch.minimum(allowance, (areas - floor).clamp_min(0))
        guard = math.sqrt(torch.finfo(base.dtype).eps) / (side - 1) ** 2
        quotient = allowance / torch.maximum(
            torch.maximum(bounds, allowance), bounds.new_tensor(guard),
        )
        scales = torch.where(bounds > 0, quotient, torch.ones_like(bounds)).amin(dim=-1)
        updated = patch[:, :, 1:-1, 1:-1] + scales[:, :, None, None, None] * raw
        return current.index_copy(1, self.interior_ids, updated.reshape(batch, -1, 2)).reshape(
            batch, side, side, 2,
        )


class StaggeredPatchQ1Layer(torch.nn.Module):
    """Four sequential nonconflicting F2-D passes on one shared Q1 grid."""

    def __init__(self, side: int, patch_cells: int = 8, **kwargs: float | None) -> None:
        super().__init__()
        if patch_cells % 2:
            raise ValueError("patch_cells must be even for the staggered schedule")
        shift = patch_cells // 2
        # With one patch across the entire domain there are no interior
        # seams; shifted full-size patches do not exist or need to be run.
        offsets = (
            ((0, 0),) if side - 1 == patch_cells else
            ((0, 0), (shift, 0), (0, shift), (shift, shift))
        )
        self.passes = torch.nn.ModuleList(
            SafePatchQ1Pass(
                side, patch_cells, offset_row=row, offset_column=column,
                **kwargs,
            )
            for row, column in offsets
        )

    def forward(self, base: torch.Tensor, logits: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(logits) != len(self.passes):
            raise ValueError("one latent field per patch pass is required")
        current = base
        for patch_pass, field in zip(self.passes, logits):
            current = patch_pass(current, field)
        return current
