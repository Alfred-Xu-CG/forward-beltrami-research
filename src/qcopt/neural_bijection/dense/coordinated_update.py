"""Shared-direction feasible updates of a declared Q1 vertex table.

Four-corner positivity also supports either fixed P1 diagonal, but Q1 and
P1 queries remain different functions. Safety is for this vertex table only.
The scale is part of autograd; maxima/minima are differentiable almost
everywhere. Finite-precision validation rejects a failed candidate rather
than pretending that exact-arithmetic strictness certifies rounded output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .digital_q1 import q1_corner_determinants


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def single_direction_corner_change(
    vertices: torch.Tensor, amplitude: torch.Tensor,
    direction: torch.Tensor | tuple[float, float],
) -> torch.Tensor:
    """Exact linear change of all SW, SE, NE, NW corner determinants.

    Amplitudes are physical displacements along one common direction and
    have shape (B,R,C). This identity is valid for arbitrary current geometry
    and arbitrary amplitudes; it does not itself imply positivity.
    """
    e = torch.as_tensor(direction, dtype=vertices.dtype, device=vertices.device)
    if e.shape != (2,) or amplitude.shape != vertices.shape[:-1]:
        raise ValueError("one common (2,) direction and (B,R,C) amplitudes required")
    a, b = vertices[:, :-1, :-1], vertices[:, :-1, 1:]
    c, d = vertices[:, 1:, 1:], vertices[:, 1:, :-1]
    ua, ub = amplitude[:, :-1, :-1], amplitude[:, :-1, 1:]
    uc, ud = amplitude[:, 1:, 1:], amplitude[:, 1:, :-1]

    def change(p, q, r, up, uq, ur):
        return (uq - up) * _cross(e, r - p) + (ur - up) * _cross(q - p, e)

    return torch.stack((
        change(a, b, d, ua, ub, ud),
        change(a, b, c, ua, ub, uc),
        change(d, b, c, ud, ub, uc),
        change(a, c, d, ua, uc, ud),
    ), dim=-1)


def interpolate_proposal(coefficients: torch.Tensor, shape: tuple[int, int], *,
                         boundary: str = "none") -> torch.Tensor:
    """Interpolate raw physical amplitudes, never an accepted map.

    Constants retain the same physical displacement at every output level;
    align_corners matches the vertex-grid convention. Coefficient gradients
    still aggregate their support, so equal optimizer rates need calibration.
    """
    if coefficients.ndim != 3 or min(shape) < 2 or boundary not in ("none", "fixed"):
        raise ValueError("coefficients need (B,R,C); output dimensions must be >=2")
    if boundary == "fixed":
        # A coarse boundary zero gives a coarse-width transition. Masking only
        # fine output vertices can instead create a one-cell discontinuity.
        mask = torch.ones_like(coefficients)
        mask[:, 0] = mask[:, -1] = 0
        mask[:, :, 0] = mask[:, :, -1] = 0
        coefficients = coefficients * mask
    return F.interpolate(coefficients[:, None], size=shape, mode="bilinear",
                         align_corners=True)[:, 0]


@dataclass
class CoordinatedUpdateResult:
    vertices: torch.Tensor
    amplitude: torch.Tensor
    scale: torch.Tensor
    gauge: torch.Tensor
    alpha_max: torch.Tensor
    normalized_margin_min: torch.Tensor


class CoordinatedQ1Update(torch.nn.Module):
    """One full-grid single-direction update with radial or analytic entry.

    ``proposal`` is a (B,R,C) amplitude field in physical units. Every corner
    is protected, including outside the nonzero support. ``reference`` gives
    positive source corner determinants; omitted means the unit rectangle.
    ``minimum_jacobian`` is an extra reference-normalized margin, not a
    necessary condition for a homeomorphism. Sliding also preserves boundary
    gaps above ``minimum_boundary_gap`` times their reference gaps.

    Fixed boundary accepts arbitrary current boundary geometry (global
    injectivity remains an anchor assumption). Sliding requires an ordered
    axis-aligned rectangular anchor and horizontal/vertical directions.
    ``validate=False`` skips input/output numerical checks for inner trials;
    callers must validate accepted/exported vertices independently.
    """

    def __init__(self, direction: tuple[float, float] = (1.0, 0.0), *,
                 mode: str = "radial", boundary: str = "fixed",
                 minimum_jacobian: float = 0.001,
                 minimum_boundary_gap: float = 0.0, theta: float = 0.95) -> None:
        super().__init__()
        if mode not in ("radial", "analytic") or boundary not in ("fixed", "sliding"):
            raise ValueError("mode radial/analytic and boundary fixed/sliding required")
        if len(direction) != 2 or not all(math.isfinite(v) for v in direction):
            raise ValueError("direction must be finite and nonzero")
        length = math.hypot(*direction)
        if length == 0:
            raise ValueError("direction must be nonzero")
        if boundary == "sliding" and direction[0] != 0 and direction[1] != 0:
            raise ValueError("sliding requires an axis direction")
        if not math.isfinite(minimum_jacobian) or minimum_jacobian < 0:
            raise ValueError("minimum_jacobian must be finite and nonnegative")
        if not math.isfinite(minimum_boundary_gap) or minimum_boundary_gap < 0:
            raise ValueError("minimum_boundary_gap must be finite and nonnegative")
        if not math.isfinite(theta) or not 0 < theta < 1:
            raise ValueError("theta must be in (0,1)")
        self.direction = (direction[0] / length, direction[1] / length)
        self.mode, self.boundary = mode, boundary
        self.minimum_jacobian = minimum_jacobian
        self.minimum_boundary_gap, self.theta = minimum_boundary_gap, theta

    def _mask(self, proposal: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(proposal)
        if self.boundary == "fixed" or self.direction[0] == 0:
            mask[:, 0] = 0
            mask[:, -1] = 0
        if self.boundary == "fixed" or self.direction[1] == 0:
            mask[:, :, 0] = 0
            mask[:, :, -1] = 0
        return proposal * mask

    def _constraints(self, vertices, proposal, reference):
        qref = q1_corner_determinants(reference)
        corners = q1_corner_determinants(vertices) / qref
        changes = single_direction_corner_change(vertices, proposal, self.direction) / qref
        slack = (corners - self.minimum_jacobian).flatten(1)
        delta = changes.flatten(1)
        if self.boundary == "sliding":
            edge_current = (vertices[:, 0], vertices[:, -1], vertices[:, :, 0], vertices[:, :, -1])
            edge_raw = (proposal[:, 0], proposal[:, -1], proposal[:, :, 0], proposal[:, :, -1])
            edge_ref = (reference[:, 0], reference[:, -1], reference[:, :, 0], reference[:, :, -1])
            for edge, raw, ref, axis in zip(edge_current, edge_raw, edge_ref, (0, 0, 1, 1)):
                gaps_ref = ref[:, 1:, axis] - ref[:, :-1, axis]
                slack = torch.cat((slack, (edge[:, 1:, axis] - edge[:, :-1, axis]) /
                                   gaps_ref - self.minimum_boundary_gap), dim=1)
                delta = torch.cat((delta, self.direction[axis] *
                                   (raw[:, 1:] - raw[:, :-1]) / gaps_ref), dim=1)
        return slack, delta, qref

    def forward(self, vertices: torch.Tensor, proposal: torch.Tensor, *,
                reference: torch.Tensor | None = None,
                alpha_trial: float | torch.Tensor = 1.0,
                validate: bool = True) -> CoordinatedUpdateResult:
        if (vertices.ndim != 4 or vertices.shape[-1] != 2 or vertices.shape[0] < 1
                or min(vertices.shape[1:3]) < 2 or proposal.shape != vertices.shape[:-1]):
            raise ValueError("vertices (B,R>=2,C>=2,2) and proposal (B,R,C) required")
        if vertices.dtype not in (torch.float32, torch.float64):
            raise ValueError("vertices must use float32 or float64")
        if proposal.dtype != vertices.dtype or proposal.device != vertices.device:
            raise ValueError("vertices and proposal must match dtype/device")
        current = vertices.to(torch.float64)
        raw = self._mask(proposal.to(torch.float64))
        if reference is None:
            rows, columns = vertices.shape[1:3]
            y, x = torch.meshgrid(torch.linspace(0, 1, rows, dtype=torch.float64, device=vertices.device),
                                  torch.linspace(0, 1, columns, dtype=torch.float64, device=vertices.device),
                                  indexing="ij")
            reference64 = torch.stack((x, y), dim=-1)[None]
        else:
            if reference.shape[1:] != vertices.shape[1:] or reference.shape[0] not in (1, vertices.shape[0]):
                raise ValueError("reference must have matching grid and batch 1 or B")
            reference64 = reference.to(device=vertices.device, dtype=torch.float64)
        slack, delta, qref = self._constraints(current, raw, reference64)
        if validate:
            if not bool(torch.isfinite(current).all() and torch.isfinite(raw).all()
                        and torch.isfinite(slack).all() and torch.isfinite(delta).all()
                        and (qref > 0).all() and (slack > 0).all()):
                raise ValueError("finite inputs and strictly positive reference/current margins required")
            if self.boundary == "sliding":
                if not self._rectangle(current):
                    raise ValueError("sliding anchor needs ordered axis-aligned rectangular boundary")
        adverse = (-delta).clamp_min(0)
        gauge = (adverse / slack).amax(dim=1)
        if validate and not bool(torch.isfinite(gauge).all()):
            raise ValueError("proposal gauge overflowed finite-precision safety arithmetic")
        nonzero_gauge = gauge > 0
        diagnostic_denominator = torch.where(nonzero_gauge, gauge, torch.ones_like(gauge))
        alpha_max = torch.where(nonzero_gauge, 1 / diagnostic_denominator,
                                torch.full_like(gauge, float("inf")))
        if self.mode == "radial":
            scale = 1 / (1 + gauge)
        else:
            trial = torch.as_tensor(alpha_trial, device=vertices.device, dtype=torch.float64)
            if trial.ndim > 1 or (trial.ndim == 1 and trial.shape != (vertices.shape[0],)):
                raise ValueError("alpha_trial must be scalar or shape (B,)")
            if validate and not bool(torch.isfinite(trial).all() and (trial >= 0).all()):
                raise ValueError("alpha_trial must be finite and nonnegative")
            # Exactly min(trial, theta*alpha_max), without evaluating a huge
            # inactive reciprocal on the scale's backward path. Even a finite
            # alpha_max~1e305 has an overflowing derivative; 0*inf can poison
            # the VJP although the trial branch should have derivative zero.
            bound_active = gauge * trial > self.theta
            active_denominator = torch.where(bound_active, gauge, torch.ones_like(gauge))
            scale = torch.where(bound_active, self.theta / active_denominator, trial)
        amplitude64 = raw * scale[:, None, None]
        e = current.new_tensor(self.direction)
        candidate = (current + amplitude64[..., None] * e).to(vertices.dtype)
        # Evaluate the actual rounded coordinates; do not use linear prediction
        # as a substitute for a saved-output topology check.
        output_slack, _, _ = self._constraints(candidate.to(torch.float64), torch.zeros_like(raw), reference64)
        margin = output_slack.amin(dim=1)
        if validate and not bool(torch.isfinite(candidate).all() and (margin > 0).all()):
            raise RuntimeError("rounded candidate failed strict margins; reject this proposal")
        return CoordinatedUpdateResult(candidate, amplitude64.to(vertices.dtype), scale, gauge,
                                       alpha_max, margin)

    @staticmethod
    def _rectangle(vertices: torch.Tensor) -> bool:
        x0, x1 = vertices[:, 0, 0, 0], vertices[:, 0, -1, 0]
        y0, y1 = vertices[:, 0, 0, 1], vertices[:, -1, 0, 1]
        return bool((x0 < x1).all() and (y0 < y1).all()
                    and (vertices[:, 0, :, 1] == y0[:, None]).all()
                    and (vertices[:, -1, :, 1] == y1[:, None]).all()
                    and (vertices[:, :, 0, 0] == x0[:, None]).all()
                    and (vertices[:, :, -1, 0] == x1[:, None]).all())
