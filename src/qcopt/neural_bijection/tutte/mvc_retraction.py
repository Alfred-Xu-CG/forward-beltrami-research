"""Covariance logit lifts and decoder-mediated geometric retractions.

The public construction in this module never accepts ``Y + alpha * d`` as a
map.  It lifts the requested vertex tangent into supported directed logits and
then calls an existing Route-I Tutte decoder, whose ordinary solver and
positive-face checks remain the acceptance authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch

from ...forward.tutte_directed_implicit import DirectedTutteSystem
from ...mesh import TriMesh
from .mvc import MeanValueCoordinateEncoder


@dataclass(frozen=True)
class CovarianceLiftResult:
    """A weighted-minimum-norm solution of the differentiated row equations.

    Supported slots of ``delta_logits`` minimize
    ``sum_ij probabilities_ij * delta_logits_ij**2`` subject to the complete
    frozen barycentric row equations for the supplied global vertex direction.
    This is a right inverse of the decoder differential when ``control`` is a
    decoder state satisfying barycentric equilibrium.  For an arbitrary
    off-equilibrium diagnostic state, it solves the centered frozen-row
    equation but is not described as a decoder differential.  Padding slots
    are exactly zero.
    """

    delta_logits: torch.Tensor
    probabilities: torch.Tensor
    covariance: torch.Tensor
    condition_number: torch.Tensor
    barycentric_rhs: torch.Tensor
    linearized_residual: torch.Tensor
    equilibrium_residual: torch.Tensor


@dataclass(frozen=True)
class CovarianceRetractionResult:
    """Base state, lifted parameters, and the newly decoded accepted state."""

    base_control: torch.Tensor
    canonical_logits: torch.Tensor
    canonical_probabilities: torch.Tensor
    canonical_boundary: torch.Tensor
    updated_control: torch.Tensor
    updated_logits: torch.Tensor
    updated_boundary: torch.Tensor
    lift: CovarianceLiftResult


def _global_neighbor_vertices(system: DirectedTutteSystem) -> np.ndarray:
    """Convert the system's interior/boundary-local slots to global vertices."""

    vertices = np.full_like(system.neighbors, -1)
    rows, slots = np.nonzero(system.valid_mask)
    represented = system.neighbors[rows, slots]
    is_boundary = system.neighbor_is_boundary[rows, slots]
    vertices[rows[is_boundary], slots[is_boundary]] = system.loop[represented[is_boundary]]
    vertices[rows[~is_boundary], slots[~is_boundary]] = system.interior[
        represented[~is_boundary]
    ]
    return vertices


def _validate_float_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have float32 or float64 dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _as_optional_batch(
    name: str,
    value: torch.Tensor,
    trailing_shape: tuple[int, ...],
) -> tuple[torch.Tensor, bool]:
    expected_ndim = len(trailing_shape)
    if value.ndim not in (expected_ndim, expected_ndim + 1):
        raise ValueError(f"{name} must have trailing shape {trailing_shape} and optional batch")
    if tuple(value.shape[-expected_ndim:]) != trailing_shape:
        raise ValueError(f"{name} must have trailing shape {trailing_shape} and optional batch")
    unbatched = value.ndim == expected_ndim
    return (value.unsqueeze(0) if unbatched else value), unbatched


def _broadcast_batch(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    batch = max(value.shape[0] for value in values)
    if batch < 1 or any(value.shape[0] not in (1, batch) for value in values):
        raise ValueError("batch dimensions must match or be singleton")
    return tuple(value.expand((batch,) + tuple(value.shape[1:])) for value in values)


def _squeeze_lift(result: CovarianceLiftResult) -> CovarianceLiftResult:
    return CovarianceLiftResult(
        delta_logits=result.delta_logits[0],
        probabilities=result.probabilities[0],
        covariance=result.covariance[0],
        condition_number=result.condition_number[0],
        barycentric_rhs=result.barycentric_rhs[0],
        linearized_residual=result.linearized_residual[0],
        equilibrium_residual=result.equilibrium_residual[0],
    )


def _covariance_lift_batched(
    system: DirectedTutteSystem,
    probabilities: torch.Tensor,
    control: torch.Tensor,
    direction: torch.Tensor,
    *,
    max_covariance_condition: float,
) -> CovarianceLiftResult:
    """Core calculation on already validated and batch-broadcast tensors."""

    batch = probabilities.shape[0]
    if system.n_rows == 0:
        empty_rows = probabilities.new_empty((batch, 0, 2))
        return CovarianceLiftResult(
            delta_logits=torch.zeros_like(probabilities),
            probabilities=probabilities,
            covariance=probabilities.new_empty((batch, 0, 2, 2)),
            condition_number=probabilities.new_empty((batch, 0)),
            barycentric_rhs=empty_rows,
            linearized_residual=empty_rows.clone(),
            equilibrium_residual=empty_rows.clone(),
        )

    mask = torch.as_tensor(system.valid_mask, dtype=torch.bool, device=probabilities.device)
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite")
    if bool(torch.any(probabilities[:, mask] <= 0.0)) or bool(
        torch.any(probabilities[:, ~mask] != 0.0)
    ):
        raise ValueError("supported probabilities must be positive and padding probabilities zero")
    row_sum = probabilities.sum(dim=-1)
    normalization_tolerance = 256.0 * torch.finfo(probabilities.dtype).eps
    if not bool(torch.all(torch.abs(row_sum - 1.0) <= normalization_tolerance)):
        raise ValueError("supported probabilities must sum to one in every row")

    global_neighbors_numpy = _global_neighbor_vertices(system)
    safe_neighbors_numpy = np.where(system.valid_mask, global_neighbors_numpy, 0)
    global_neighbors = torch.as_tensor(
        safe_neighbors_numpy, dtype=torch.int64, device=probabilities.device
    )
    flat_neighbors = global_neighbors.reshape(-1)
    neighbor_control = control.index_select(1, flat_neighbors).reshape(
        batch, system.n_rows, system.max_degree, 2
    )
    neighbor_direction = direction.index_select(1, flat_neighbors).reshape_as(neighbor_control)
    interior_index = torch.as_tensor(
        system.interior, dtype=torch.int64, device=probabilities.device
    )
    interior_control = control.index_select(1, interior_index)
    interior_direction = direction.index_select(1, interior_index)

    weighted_neighbor = torch.sum(probabilities[..., None] * neighbor_control, dim=2)
    centered = neighbor_control - weighted_neighbor[..., None, :]
    covariance = torch.einsum(
        "bijd,bij,bije->bide", centered, probabilities, centered
    )
    eigenvalues = torch.linalg.eigvalsh(covariance)
    smallest, largest = eigenvalues[..., 0], eigenvalues[..., 1]
    representable_limit = 1.0 / (64.0 * torch.finfo(probabilities.dtype).eps)
    allowed_condition = min(float(max_covariance_condition), representable_limit)
    condition = largest / smallest
    rank_failure = (
        (~torch.isfinite(eigenvalues).all(dim=-1))
        | (smallest <= 0.0)
        | (largest <= torch.finfo(probabilities.dtype).tiny)
    )
    conditioning_failure = (~torch.isfinite(condition)) | (condition > allowed_condition)
    if bool(torch.any(rank_failure)):
        raise ValueError("local covariance is rank deficient")
    if bool(torch.any(conditioning_failure)):
        raise ValueError(
            f"local covariance condition exceeds the fail-closed limit {allowed_condition:.6g}"
        )

    weighted_neighbor_direction = torch.sum(
        probabilities[..., None] * neighbor_direction, dim=2
    )
    barycentric_rhs = interior_direction - weighted_neighbor_direction
    dual = torch.linalg.solve(covariance, barycentric_rhs[..., None]).squeeze(-1)
    delta_logits = torch.einsum("bijd,bid->bij", centered, dual)
    delta_logits = torch.where(mask, delta_logits, torch.zeros_like(delta_logits))
    mean_delta = torch.sum(probabilities * delta_logits, dim=-1, keepdim=True)
    delta_probabilities = probabilities * (delta_logits - mean_delta)
    recovered_rhs = torch.sum(delta_probabilities[..., None] * neighbor_control, dim=2)
    linearized_residual = recovered_rhs - barycentric_rhs
    equilibrium_residual = weighted_neighbor - interior_control
    if not all(
        bool(torch.isfinite(value).all())
        for value in (delta_logits, barycentric_rhs, linearized_residual, equilibrium_residual)
    ):
        raise ValueError("covariance lift produced a nonfinite result")
    return CovarianceLiftResult(
        delta_logits=delta_logits,
        probabilities=probabilities,
        covariance=covariance,
        condition_number=condition,
        barycentric_rhs=barycentric_rhs,
        linearized_residual=linearized_residual,
        equilibrium_residual=equilibrium_residual,
    )


def covariance_logit_lift(
    system: DirectedTutteSystem,
    logits: torch.Tensor,
    control: torch.Tensor,
    direction: torch.Tensor,
    *,
    max_covariance_condition: float = 1.0e8,
) -> CovarianceLiftResult:
    r"""Lift a full vertex tangent to directed row-softmax logits.

    For an interior row, let ``p`` be its represented probabilities and let
    ``q = sum_j p_j Y_j``.  Differentiating

    ``Y_i = sum_j p_ij Y_j``

    gives ``b_i = d_i - sum_j p_ij d_j = sum_j delta_p_ij Y_j``.  The softmax
    derivative makes the exact finite-precision row matrix
    ``p_ij (Y_j - q_i)``.  Consequently

    ``delta_l_ij = (Y_j-q_i)^T C_i^{-1} b_i`` and
    ``C_i = sum_j p_ij (Y_j-q_i)(Y_j-q_i)^T``.

    At an exact barycentric solution ``q_i=Y_i``; this is precisely the plan's
    formula with ``r_ij=Y_j-Y_i``.  Centering at represented ``q_i`` preserves
    the actual softmax differential when a numerical solve has a small residual.
    The construction is globally coupled through every neighbor direction in
    ``b_i``, although the constrained weighted minimization separates by rows.
    At a decoder state this is a decoder-Jacobian right inverse.  The function
    also admits off-equilibrium inputs for diagnostics; there it solves the
    displayed centered frozen-row equation only.
    """

    if not isinstance(system, DirectedTutteSystem):
        raise TypeError("system must be a DirectedTutteSystem")
    if (
        not isinstance(max_covariance_condition, Real)
        or not np.isfinite(max_covariance_condition)
        or max_covariance_condition <= 1.0
    ):
        raise ValueError("max_covariance_condition must be finite and greater than one")
    for name, value in (("logits", logits), ("control", control), ("direction", direction)):
        _validate_float_tensor(name, value)
    if control.dtype != logits.dtype or direction.dtype != logits.dtype:
        raise TypeError("logits, control, and direction must share dtype")
    if control.device != logits.device or direction.device != logits.device:
        raise ValueError("logits, control, and direction must share device")

    z, z_unbatched = _as_optional_batch(
        "logits", logits, (system.n_rows, system.max_degree)
    )
    y, y_unbatched = _as_optional_batch("control", control, (system.n_vertices, 2))
    d, d_unbatched = _as_optional_batch("direction", direction, (system.n_vertices, 2))
    z, y, d = _broadcast_batch(z, y, d)
    unbatched = z_unbatched and y_unbatched and d_unbatched
    if system.n_rows:
        mask = torch.as_tensor(system.valid_mask, dtype=torch.bool, device=z.device)
        probabilities = torch.softmax(z.masked_fill(~mask, -torch.inf), dim=-1)
    else:
        probabilities = torch.zeros_like(z)
    result = _covariance_lift_batched(
        system,
        probabilities,
        y,
        d,
        max_covariance_condition=float(max_covariance_condition),
    )
    return _squeeze_lift(result) if unbatched else result


class CovarianceLogitLift(torch.nn.Module):
    """System-bound lift with the MVC-compatible ``(Y, p, d)`` API.

    ``current_map`` and ``direction`` have trailing shape ``(V, 2)``;
    ``probabilities`` has the encoder/decoder padded shape ``(I, D)``.  Each
    argument may additionally have one leading batch dimension, with singleton
    batches broadcast by :func:`covariance_logit_lift`.
    """

    def __init__(
        self,
        system: DirectedTutteSystem,
        *,
        max_covariance_condition: float = 1.0e8,
    ) -> None:
        super().__init__()
        if not isinstance(system, DirectedTutteSystem):
            raise TypeError("system must be a DirectedTutteSystem")
        if (
            not isinstance(max_covariance_condition, Real)
            or not np.isfinite(max_covariance_condition)
            or max_covariance_condition <= 1.0
        ):
            raise ValueError("max_covariance_condition must be finite and greater than one")
        self.system = system
        self.max_covariance_condition = float(max_covariance_condition)

    def forward(
        self,
        current_map: torch.Tensor,
        probabilities: torch.Tensor,
        direction: torch.Tensor,
    ) -> CovarianceLiftResult:
        for name, value in (
            ("current_map", current_map),
            ("probabilities", probabilities),
            ("direction", direction),
        ):
            _validate_float_tensor(name, value)
        if current_map.dtype != probabilities.dtype or direction.dtype != probabilities.dtype:
            raise TypeError("current_map, probabilities, and direction must share dtype")
        if current_map.device != probabilities.device or direction.device != probabilities.device:
            raise ValueError("current_map, probabilities, and direction must share device")
        p, p_unbatched = _as_optional_batch(
            "probabilities",
            probabilities,
            (self.system.n_rows, self.system.max_degree),
        )
        y, y_unbatched = _as_optional_batch(
            "current_map", current_map, (self.system.n_vertices, 2)
        )
        d, d_unbatched = _as_optional_batch(
            "direction", direction, (self.system.n_vertices, 2)
        )
        p, y, d = _broadcast_batch(p, y, d)
        result = _covariance_lift_batched(
            self.system,
            p,
            y,
            d,
            max_covariance_condition=self.max_covariance_condition,
        )
        return _squeeze_lift(result) if p_unbatched and y_unbatched and d_unbatched else result


class CovarianceTutteRetraction(torch.nn.Module):
    """Convert a vertex tangent into a newly decoded hard-valid Tutte map.

    ``decoder`` is an existing Route-I control decoder with the contract
    ``decoder(logits, boundary) -> control`` and a ``DirectedTutteSystem`` in
    ``decoder.system``.  Fixed boundary is the default and rejects, rather than
    silently discards, any nonzero requested boundary tangent.  With
    ``allow_boundary_motion=True``, the represented boundary follows
    ``boundary + alpha * direction[loop]`` and is still validated by the
    wrapped decoder before an updated state is returned.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        encoder: torch.nn.Module | None = None,
        allow_boundary_motion: bool = False,
        max_covariance_condition: float = 1.0e8,
    ) -> None:
        super().__init__()
        if not isinstance(decoder, torch.nn.Module) or not hasattr(decoder, "system"):
            raise TypeError("decoder must be a Route-I torch module exposing system")
        if not isinstance(decoder.system, DirectedTutteSystem):
            raise TypeError("decoder.system must be a DirectedTutteSystem")
        if not isinstance(allow_boundary_motion, bool):
            raise TypeError("allow_boundary_motion must be bool")
        if (
            not isinstance(max_covariance_condition, Real)
            or not np.isfinite(max_covariance_condition)
            or max_covariance_condition <= 1.0
        ):
            raise ValueError("max_covariance_condition must be finite and greater than one")
        if encoder is None:
            source_mesh = TriMesh(decoder.system.source_vertices, decoder.system.faces)
            encoder = MeanValueCoordinateEncoder(source_mesh)
        if not isinstance(encoder, torch.nn.Module) or not hasattr(encoder, "system"):
            raise TypeError("encoder must be a torch module exposing system")
        encoder_system = encoder.system
        if not isinstance(encoder_system, DirectedTutteSystem) or any(
            not np.array_equal(getattr(encoder_system, name), getattr(decoder.system, name))
            for name in (
                "loop",
                "interior",
                "neighbors",
                "neighbor_is_boundary",
                "valid_mask",
                "faces",
                "source_vertices",
            )
        ):
            raise ValueError("encoder and decoder must use the same fixed Tutte system")
        self.decoder = decoder
        self.encoder = encoder
        self.covariance_lift = CovarianceLogitLift(
            decoder.system,
            max_covariance_condition=max_covariance_condition,
        )
        self.allow_boundary_motion = allow_boundary_motion
        self.max_covariance_condition = float(max_covariance_condition)

    @property
    def system(self) -> DirectedTutteSystem:
        return self.decoder.system

    def forward(
        self,
        logits: torch.Tensor,
        boundary: torch.Tensor,
        direction: torch.Tensor,
        alpha: float | torch.Tensor,
    ) -> CovarianceRetractionResult:
        for name, value in (("logits", logits), ("boundary", boundary), ("direction", direction)):
            _validate_float_tensor(name, value)
        if boundary.dtype != logits.dtype or direction.dtype != logits.dtype:
            raise TypeError("logits, boundary, and direction must share dtype")
        if boundary.device != logits.device or direction.device != logits.device:
            raise ValueError("logits, boundary, and direction must share device")

        system = self.system
        z, z_unbatched = _as_optional_batch(
            "logits", logits, (system.n_rows, system.max_degree)
        )
        b, b_unbatched = _as_optional_batch("boundary", boundary, (len(system.loop), 2))
        d, d_unbatched = _as_optional_batch("direction", direction, (system.n_vertices, 2))
        z, b, d = _broadcast_batch(z, b, d)
        unbatched = z_unbatched and b_unbatched and d_unbatched

        if isinstance(alpha, torch.Tensor):
            if alpha.ndim != 0:
                raise ValueError("alpha must be a scalar")
            if alpha.dtype != z.dtype or alpha.device != z.device:
                raise TypeError("tensor alpha must share logits dtype and device")
            if not bool(torch.isfinite(alpha)):
                raise ValueError("alpha must be finite")
            represented_alpha = alpha
        elif isinstance(alpha, Real) and np.isfinite(alpha):
            represented_alpha = z.new_tensor(float(alpha))
        else:
            raise TypeError("alpha must be a finite real scalar or scalar tensor")

        base_control = self.decoder(z, b)
        if base_control.ndim != 3 or tuple(base_control.shape[1:]) != (system.n_vertices, 2):
            raise ValueError("decoder returned an unexpected control shape")
        canonical = self.encoder(base_control)
        canonical_logits = canonical.logits
        canonical_probabilities = canonical.probabilities
        canonical_boundary = canonical.boundary
        boundary_index = torch.as_tensor(system.loop, dtype=torch.int64, device=z.device)
        boundary_direction = d.index_select(1, boundary_index)
        if not self.allow_boundary_motion and bool(torch.any(boundary_direction != 0.0)):
            raise ValueError("fixed boundary retraction requires an exactly zero boundary tangent")

        lift = self.covariance_lift(
            base_control,
            canonical_probabilities,
            d,
        )
        updated_logits = canonical_logits + represented_alpha * lift.delta_logits
        updated_boundary = (
            canonical_boundary + represented_alpha * boundary_direction
            if self.allow_boundary_motion
            else canonical_boundary
        )
        # This call, not the Euler candidate, is the only accepted updated map.
        updated_control = self.decoder(updated_logits, updated_boundary)

        if not unbatched:
            return CovarianceRetractionResult(
                base_control=base_control,
                canonical_logits=canonical_logits,
                canonical_probabilities=canonical_probabilities,
                canonical_boundary=canonical_boundary,
                updated_control=updated_control,
                updated_logits=updated_logits,
                updated_boundary=updated_boundary,
                lift=lift,
            )
        return CovarianceRetractionResult(
            base_control=base_control[0],
            canonical_logits=canonical_logits[0],
            canonical_probabilities=canonical_probabilities[0],
            canonical_boundary=canonical_boundary[0],
            updated_control=updated_control[0],
            updated_logits=updated_logits[0],
            updated_boundary=updated_boundary[0],
            lift=_squeeze_lift(lift),
        )


__all__ = [
    "CovarianceLiftResult",
    "CovarianceLogitLift",
    "CovarianceRetractionResult",
    "CovarianceTutteRetraction",
    "covariance_logit_lift",
]
