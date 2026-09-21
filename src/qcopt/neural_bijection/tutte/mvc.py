"""Differentiable mean-value-coordinate canonicalization for Tutte maps.

This module is an encoder/gauge choice for an existing positive directed
Tutte decoder.  It is not a second hard-bijective decoder.  The fixed mesh
supplies an oriented cyclic one-ring for every interior vertex; returned
logits are scattered back to the sorted, padded row layout used by the Route-I
directed solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ...forward.tutte_directed_implicit import DirectedTutteSystem
from ...mesh import TriMesh


@dataclass(frozen=True)
class MVCDiagnostics:
    """Per-interior-row geometric diagnostics in the returned tensor dtype."""

    minimum_edge_length: torch.Tensor
    minimum_angle_sine: torch.Tensor
    winding_error: torch.Tensor
    barycentric_residual: torch.Tensor
    covariance_condition: torch.Tensor


@dataclass(frozen=True)
class MVCEncodeResult:
    """Canonical row logits, represented probabilities, and unchanged boundary."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    boundary: torch.Tensor
    diagnostics: MVCDiagnostics


@dataclass(frozen=True)
class MVCCanonicalizationResult:
    """Decoded control map and its mean-value-coordinate representation."""

    control: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    boundary: torch.Tensor
    diagnostics: MVCDiagnostics


def _first_diagnostics(diagnostics: MVCDiagnostics) -> MVCDiagnostics:
    return MVCDiagnostics(
        minimum_edge_length=diagnostics.minimum_edge_length[0],
        minimum_angle_sine=diagnostics.minimum_angle_sine[0],
        winding_error=diagnostics.winding_error[0],
        barycentric_residual=diagnostics.barycentric_residual[0],
        covariance_condition=diagnostics.covariance_condition[0],
    )


def _global_neighbor_vertices(system: DirectedTutteSystem) -> np.ndarray:
    neighbors = np.full_like(system.neighbors, -1)
    interior_mask = system.valid_mask & ~system.neighbor_is_boundary
    boundary_mask = system.valid_mask & system.neighbor_is_boundary
    neighbors[interior_mask] = system.interior[system.neighbors[interior_mask]]
    neighbors[boundary_mask] = system.loop[system.neighbors[boundary_mask]]
    return neighbors


def _oriented_cyclic_slots(
    mesh: TriMesh,
    system: DirectedTutteSystem,
    global_neighbors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cyclic-to-sorted slots and cyclic next/previous positions.

    For a positively oriented face ``(v,a,b)``, the ray to ``a`` precedes the
    ray to ``b`` counterclockwise around ``v``.  The same rule after cyclically
    rotating a face gives one successor arc per incident triangle.  The disk
    validator in :class:`DirectedTutteSystem` has already checked that each
    interior link is one simple cycle; the checks below keep this precomputation
    fail-closed if that contract ever changes.
    """

    cyclic_slots = np.full_like(system.neighbors, -1)
    next_positions = np.zeros_like(system.neighbors)
    previous_positions = np.zeros_like(system.neighbors)
    incident_faces: list[list[np.ndarray]] = [[] for _ in range(system.n_vertices)]
    for face in mesh.faces:
        for vertex in face:
            incident_faces[int(vertex)].append(face)

    for row, vertex in enumerate(system.interior.tolist()):
        supported = global_neighbors[row, system.valid_mask[row]].tolist()
        supported_set = set(int(item) for item in supported)
        successor: dict[int, int] = {}
        predecessor: dict[int, int] = {}
        for face in incident_faces[vertex]:
            position = int(np.flatnonzero(face == vertex)[0])
            first = int(face[(position + 1) % 3])
            second = int(face[(position - 1) % 3])
            if first in successor and successor[first] != second:
                raise ValueError("interior one-ring has inconsistent oriented successors")
            if second in predecessor and predecessor[second] != first:
                raise ValueError("interior one-ring has inconsistent oriented predecessors")
            successor[first] = second
            predecessor[second] = first
        if set(successor) != supported_set or set(predecessor) != supported_set:
            raise ValueError("interior one-ring is not one oriented neighbor cycle")

        degree = len(supported)
        start = min(supported)
        cycle: list[int] = []
        current = start
        for _ in range(degree):
            if current in cycle:
                raise ValueError("interior one-ring closes before visiting every neighbor")
            cycle.append(current)
            current = successor[current]
        if current != start or set(cycle) != supported_set:
            raise ValueError("interior one-ring does not close as one cycle")

        sorted_slot = {int(neighbor): slot for slot, neighbor in enumerate(global_neighbors[row]) if neighbor >= 0}
        cyclic_slots[row, :degree] = [sorted_slot[neighbor] for neighbor in cycle]
        next_positions[row, :degree] = np.roll(np.arange(degree), -1)
        previous_positions[row, :degree] = np.roll(np.arange(degree), 1)
    return cyclic_slots, next_positions, previous_positions


class MeanValueCoordinateEncoder(torch.nn.Module):
    """Encode an admitted planar P1 map by canonical positive MVC row logits.

    Input has shape ``(V,2)`` or ``(batch,V,2)``.  The output row layout is
    exactly ``DirectedTutteSystem.neighbors``: supported columns are sorted by
    global vertex id and shorter rows have finite zero padding.  The MVC
    formula itself is evaluated in the oriented cyclic order induced by the
    fixed source triangulation.

    The admitted numerical domain is deliberately narrower than all finite
    vertex arrays.  Every interior edge must be resolved, every consecutive
    cyclic angle must be strictly counterclockwise and separated from 0/pi,
    the rays must wind once, the MVC denominator must be positive and finite,
    and the represented row covariance must have numerical rank two.  These
    screens are floating-point checks, not exact predicates.  Inside that
    domain the operations are ordinary differentiable Torch operations.
    """

    def __init__(
        self,
        mesh: TriMesh,
        *,
        edge_relative_tolerance: float | None = None,
        angle_sine_tolerance: float | None = None,
        winding_tolerance: float | None = None,
        covariance_rank_tolerance: float | None = None,
    ) -> None:
        super().__init__()
        for name, value, upper in (
            ("edge_relative_tolerance", edge_relative_tolerance, None),
            ("angle_sine_tolerance", angle_sine_tolerance, 1.0),
            ("winding_tolerance", winding_tolerance, math.pi),
            ("covariance_rank_tolerance", covariance_rank_tolerance, 0.25),
        ):
            if value is not None and (
                not math.isfinite(value) or value < 0.0 or (upper is not None and value >= upper)
            ):
                qualifier = f" and less than {upper}" if upper is not None else ""
                raise ValueError(f"{name} must be finite and nonnegative{qualifier}")

        self.system = DirectedTutteSystem.from_mesh(mesh)
        self.edge_relative_tolerance = edge_relative_tolerance
        self.angle_sine_tolerance = angle_sine_tolerance
        self.winding_tolerance = winding_tolerance
        self.covariance_rank_tolerance = covariance_rank_tolerance

        global_neighbors = _global_neighbor_vertices(self.system)
        cyclic_slots, next_positions, previous_positions = _oriented_cyclic_slots(
            mesh, self.system, global_neighbors
        )
        cyclic_valid = cyclic_slots >= 0
        cyclic_slot_safe = np.where(cyclic_valid, cyclic_slots, 0)
        cyclic_neighbors = np.zeros_like(cyclic_slots)
        if self.system.n_rows:
            cyclic_neighbors[cyclic_valid] = global_neighbors[
                np.broadcast_to(np.arange(self.system.n_rows)[:, None], cyclic_slots.shape)[cyclic_valid],
                cyclic_slot_safe[cyclic_valid],
            ]

        for name, values in (
            ("_interior", self.system.interior),
            ("_loop", self.system.loop),
            ("_neighbor_vertices", global_neighbors),
            ("_valid", self.system.valid_mask),
            ("_cyclic_slots", cyclic_slots),
            ("_cyclic_slot_safe", cyclic_slot_safe),
            ("_cyclic_neighbors", cyclic_neighbors),
            ("_cyclic_valid", cyclic_valid),
            ("_next_positions", next_positions),
            ("_previous_positions", previous_positions),
        ):
            self.register_buffer(name, torch.from_numpy(np.array(values, copy=True)), persistent=False)

    @property
    def valid_mask(self) -> np.ndarray:
        return self._valid.detach().cpu().numpy().copy()

    @property
    def cyclic_slots(self) -> np.ndarray:
        return self._cyclic_slots.detach().cpu().numpy().copy()

    @property
    def neighbor_vertices(self) -> np.ndarray:
        return self._neighbor_vertices.detach().cpu().numpy().copy()

    @property
    def interior_vertices(self) -> np.ndarray:
        return self._interior.detach().cpu().numpy().copy()

    def _dtype_tolerances(self, dtype: torch.dtype) -> tuple[float, float, float, float]:
        epsilon = torch.finfo(dtype).eps
        edge = self.edge_relative_tolerance
        angle = self.angle_sine_tolerance
        winding = self.winding_tolerance
        rank = self.covariance_rank_tolerance
        return (
            64.0 * epsilon if edge is None else edge,
            64.0 * epsilon if angle is None else angle,
            1024.0 * epsilon * max(self.system.max_degree, 1) if winding is None else winding,
            256.0 * epsilon if rank is None else rank,
        )

    def forward(self, vertices: torch.Tensor) -> MVCEncodeResult:
        if not isinstance(vertices, torch.Tensor):
            raise TypeError("vertices must be a torch tensor")
        if vertices.dtype not in (torch.float32, torch.float64):
            raise TypeError("vertices must have float32 or float64 dtype")
        trailing = (self.system.n_vertices, 2)
        if vertices.ndim not in (2, 3) or tuple(vertices.shape[-2:]) != trailing:
            raise ValueError(f"vertices must have trailing shape {trailing} and optional batch")
        if vertices.ndim == 3 and vertices.shape[0] == 0:
            raise ValueError("vertices batch must be nonempty")
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("vertices must be finite")
        if self._valid.device != vertices.device:
            raise ValueError("move MeanValueCoordinateEncoder to the input device before use")

        unbatched = vertices.ndim == 2
        values = vertices.unsqueeze(0) if unbatched else vertices
        boundary = values.index_select(1, self._loop)
        batch = values.shape[0]
        if self.system.n_rows == 0:
            empty = values.new_empty((batch, 0, 0))
            empty_rows = values.new_empty((batch, 0))
            diagnostics = MVCDiagnostics(
                minimum_edge_length=empty_rows,
                minimum_angle_sine=empty_rows.clone(),
                winding_error=empty_rows.clone(),
                barycentric_residual=empty_rows.clone(),
                covariance_condition=empty_rows.clone(),
            )
            result = MVCEncodeResult(empty, empty.clone(), boundary, diagnostics)
            if unbatched:
                return MVCEncodeResult(
                    result.logits[0],
                    result.probabilities[0],
                    result.boundary[0],
                    _first_diagnostics(diagnostics),
                )
            return result

        centers = values.index_select(1, self._interior)
        flat_neighbors = self._cyclic_neighbors.reshape(-1)
        neighbors = values.index_select(1, flat_neighbors).reshape(
            batch, self.system.n_rows, self.system.max_degree, 2
        )
        rays = neighbors - centers.unsqueeze(2)
        lengths = torch.linalg.vector_norm(rays, dim=-1)
        valid = self._cyclic_valid.unsqueeze(0)
        safe_lengths = torch.where(valid, lengths, torch.ones_like(lengths))

        edge_relative, angle_tolerance, winding_tolerance, rank_tolerance = self._dtype_tolerances(
            vertices.dtype
        )
        coordinate_span = (
            values.amax(dim=1) - values.amin(dim=1)
        ).amax(dim=-1).detach()
        edge_threshold = edge_relative * coordinate_span[:, None, None]
        if bool(torch.any(valid & (lengths <= edge_threshold))):
            raise ValueError("every supported interior one-ring edge must have resolved positive length")

        gather_index = self._next_positions.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, 2)
        next_rays = torch.gather(rays, 2, gather_index)
        units = rays / safe_lengths.unsqueeze(-1)
        next_lengths = torch.gather(safe_lengths, 2, self._next_positions.unsqueeze(0).expand(batch, -1, -1))
        next_units = next_rays / next_lengths.unsqueeze(-1)
        sine = units[..., 0] * next_units[..., 1] - units[..., 1] * next_units[..., 0]
        cosine = (units * next_units).sum(dim=-1)
        angles = torch.atan2(sine, cosine)
        if bool(torch.any(valid & ((sine <= angle_tolerance) | ~torch.isfinite(angles)))):
            raise ValueError(
                "cyclic one-ring angles must be finite, strictly counterclockwise, and separated from 0 and pi"
            )
        winding = torch.where(valid, angles, torch.zeros_like(angles)).sum(dim=-1)
        if bool(torch.any(torch.abs(winding - 2.0 * math.pi) > winding_tolerance)):
            raise ValueError("cyclic one-ring rays must have positive orientation and winding number one")

        half_tangent = torch.where(valid, torch.tan(0.5 * angles), torch.zeros_like(angles))
        previous_tangent = torch.gather(
            half_tangent,
            2,
            self._previous_positions.unsqueeze(0).expand(batch, -1, -1),
        )
        unnormalized = torch.where(
            valid,
            (previous_tangent + half_tangent) / safe_lengths,
            torch.zeros_like(lengths),
        )
        if bool(torch.any(valid & ((unnormalized <= 0.0) | ~torch.isfinite(unnormalized)))):
            raise ValueError("unnormalized MVC weights must be finite and strictly positive")
        denominator = unnormalized.sum(dim=-1, keepdim=True)
        if bool(
            torch.any(
                ~torch.isfinite(denominator)
                | (denominator <= torch.finfo(vertices.dtype).tiny)
            )
        ):
            raise ValueError("MVC row denominator must be finite and strictly positive")
        cyclic_probabilities = unnormalized / denominator

        covariance_xx = (cyclic_probabilities * rays[..., 0].square()).sum(dim=-1)
        covariance_xy = (cyclic_probabilities * rays[..., 0] * rays[..., 1]).sum(dim=-1)
        covariance_yy = (cyclic_probabilities * rays[..., 1].square()).sum(dim=-1)
        trace = covariance_xx + covariance_yy
        determinant = covariance_xx * covariance_yy - covariance_xy.square()
        relative_determinant = determinant / trace.square()
        if bool(
            torch.any(
                ~torch.isfinite(relative_determinant)
                | (trace <= 0.0)
                | (relative_determinant <= rank_tolerance)
            )
        ):
            raise ValueError("MVC row covariance must have resolved numerical rank two")

        log_weights = torch.where(valid, torch.log(unnormalized), torch.zeros_like(unnormalized))
        degree = valid.sum(dim=-1, keepdim=True)
        mean_log_weight = log_weights.sum(dim=-1, keepdim=True) / degree
        cyclic_logits = torch.where(valid, log_weights - mean_log_weight, torch.zeros_like(log_weights))

        slot_index = self._cyclic_slot_safe.unsqueeze(0).expand(batch, -1, -1)
        logits = torch.zeros_like(cyclic_logits).scatter_add(2, slot_index, cyclic_logits)
        probabilities = torch.zeros_like(cyclic_probabilities).scatter_add(
            2, slot_index, cyclic_probabilities
        )
        if not bool(torch.isfinite(logits).all()) or not bool(
            torch.all(probabilities[:, self._valid] > 0.0)
        ):
            raise ValueError("canonical MVC logits and supported probabilities must remain finite and positive")

        infinity = torch.full_like(lengths, torch.inf)
        minimum_edge_length = torch.where(valid, lengths, infinity).amin(dim=-1)
        minimum_angle_sine = torch.where(valid, sine, infinity).amin(dim=-1)
        winding_error = torch.abs(winding - 2.0 * math.pi)
        barycentric_residual = torch.linalg.vector_norm(
            (cyclic_probabilities.unsqueeze(-1) * rays).sum(dim=-2), dim=-1
        )
        spectral_discriminant = torch.sqrt(
            torch.clamp_min(
                (covariance_xx - covariance_yy).square() + 4.0 * covariance_xy.square(),
                0.0,
            )
        )
        maximum_eigenvalue = 0.5 * (trace + spectral_discriminant)
        covariance_condition = maximum_eigenvalue.square() / determinant
        diagnostics = MVCDiagnostics(
            minimum_edge_length=minimum_edge_length,
            minimum_angle_sine=minimum_angle_sine,
            winding_error=winding_error,
            barycentric_residual=barycentric_residual,
            covariance_condition=covariance_condition,
        )
        result = MVCEncodeResult(logits, probabilities, boundary, diagnostics)
        if unbatched:
            return MVCEncodeResult(
                result.logits[0],
                result.probabilities[0],
                result.boundary[0],
                _first_diagnostics(diagnostics),
            )
        return result


class MVCCanonicalizationLayer(torch.nn.Module):
    """Compose an existing fixed-mesh Tutte solve with the MVC encoder.

    The returned control map is the solver output.  Re-decoding the returned
    logits and boundary reproduces that map only when the encoder's explicit
    local-domain checks pass and up to floating solve/roundoff error.  This
    class does not assert that the original latent row was unique.
    """

    def __init__(self, mesh: TriMesh, solver: torch.nn.Module, **encoder_options) -> None:
        super().__init__()
        if not hasattr(solver, "system"):
            raise TypeError("solver must expose its fixed mesh through a system member")
        system = solver.system
        if (
            system.n_vertices != mesh.n_vertices
            or not np.array_equal(system.faces, mesh.faces)
            or not np.array_equal(system.source_vertices, mesh.vertices)
        ):
            raise ValueError("solver fixed mesh does not match canonicalization mesh")
        self.solver = solver
        self.encoder = MeanValueCoordinateEncoder(mesh, **encoder_options)

    def forward(self, logits: torch.Tensor, boundary: torch.Tensor) -> MVCCanonicalizationResult:
        control = self.solver(logits, boundary)
        encoded = self.encoder(control)
        return MVCCanonicalizationResult(
            control=control,
            logits=encoded.logits,
            probabilities=encoded.probabilities,
            boundary=encoded.boundary,
            diagnostics=encoded.diagnostics,
        )


__all__ = [
    "MVCEncodeResult",
    "MVCDiagnostics",
    "MVCCanonicalizationLayer",
    "MVCCanonicalizationResult",
    "MeanValueCoordinateEncoder",
]
