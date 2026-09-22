"""Positive fixed-planar direction fits and a hard-valid symmetric Tutte layer.

This module deliberately separates three objects which are easy to conflate:

* a *local direction moment* ``sum_k c_k d_k d_k.T``;
* a shared-edge graph Laplacian assembled after adjacent local predictions are
  averaged; and
* the Beltrami coefficient of the P1 map returned by the Dirichlet solve.

An accurate fit of the first object is not a theorem about the latter two.
The hard topology statement comes only from strictly positive conductances,
the validated planar disk, the convex ordered boundary, and the independent
P1 checks in :class:`MatrixFreeSymmetricTutteLayer`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.optimize import nnls
import torch
from torch import nn
import torch.nn.functional as torch_functional

from ...forward.tutte_directed_implicit import DirectedTutteSystem
from ...mesh import TriMesh, structured_rectangle
from .symmetric import MatrixFreeSymmetricTutteLayer


_SQRT_TWO = math.sqrt(2.0)


@dataclass(frozen=True)
class PositivePlanarGraph:
    """One explicitly embedded triangular disk and its projective directions.

    Directions are unoriented and therefore stored in ``[0, pi)``.  The edge
    arrays are lexicographically sorted.  ``active_edges`` exactly follows the
    symmetric Tutte convention: boundary--boundary edges are omitted because
    they do not enter a fixed-boundary Dirichlet solve.
    """

    name: str
    mesh: TriMesh
    direction_angles: np.ndarray
    all_edges: np.ndarray
    edge_direction_indices: np.ndarray
    edge_faces: np.ndarray
    active_edges: np.ndarray
    active_edge_direction_indices: np.ndarray
    active_edge_faces: np.ndarray


@dataclass(frozen=True)
class PlanarGraphAudit:
    proper_crossing_count: int
    collinear_overlap_count: int
    boundary_loop_count: int
    euler_characteristic: int
    disk_topology: bool
    dividing_edge_count: int
    vertex_count: int
    edge_count: int
    face_count: int


@dataclass(frozen=True)
class PositiveTensorFit:
    """Strict lower-bound NNLS fit of one local direction moment.

    ``local_operator_spectral_relative_error`` is the relative matrix two-norm
    error of the local continuum symbol.  It is not an error for the globally
    assembled shared-edge operator and not a Schur-complement error.
    """

    conductances: np.ndarray
    logits: np.ndarray
    fitted_tensor: np.ndarray
    frobenius_relative_error: float
    local_operator_spectral_relative_error: float
    residual_norm: float
    minimum_conductance: float
    minimum_excess: float


def _validate_cells(cells_per_side: int) -> int:
    if isinstance(cells_per_side, bool) or not isinstance(cells_per_side, int):
        raise TypeError("cells_per_side must be an integer")
    if cells_per_side < 2:
        raise ValueError("cells_per_side must be at least two")
    return cells_per_side


def _grid_vertices(cells: int) -> np.ndarray:
    coordinates = np.linspace(0.0, 1.0, cells + 1, dtype=np.float64)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def _edge_data(mesh: TriMesh) -> tuple[np.ndarray, np.ndarray]:
    incidence: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(mesh.faces.tolist()):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            incidence.setdefault(edge, []).append(face_index)
    edges = np.asarray(sorted(incidence), dtype=np.int64)
    edge_faces = np.full((len(edges), 2), -1, dtype=np.int64)
    for position, edge in enumerate(map(tuple, edges.tolist())):
        faces = incidence[edge]
        if len(faces) not in (1, 2):
            raise RuntimeError("triangular disk edge must have one or two incident faces")
        edge_faces[position, : len(faces)] = faces
    return edges, edge_faces


def _projective_angle(vector: np.ndarray) -> float:
    angle = math.atan2(float(vector[1]), float(vector[0])) % math.pi
    if math.isclose(angle, math.pi, rel_tol=0.0, abs_tol=2.0e-14):
        return 0.0
    return angle


def _projective_distance(first: float, second: float) -> float:
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def _make_graph(name: str, mesh: TriMesh, declared_directions: np.ndarray) -> PositivePlanarGraph:
    directions = np.asarray(declared_directions, dtype=np.float64)
    if directions.ndim != 1 or len(directions) == 0:
        raise ValueError("declared directions must be a nonempty vector")
    directions = np.mod(directions, math.pi)
    directions.sort()
    if np.min(np.diff(directions)) <= 1.0e-12:
        raise ValueError("declared projective directions must be distinct")
    edges, edge_faces = _edge_data(mesh)
    edge_directions: list[int] = []
    for first, second in edges:
        angle = _projective_angle(mesh.vertices[second] - mesh.vertices[first])
        distances = np.asarray([_projective_distance(angle, candidate) for candidate in directions])
        match = int(np.argmin(distances))
        if distances[match] > 2.0e-12:
            raise RuntimeError(
                f"edge direction {math.degrees(angle):.12g} degrees is not declared"
            )
        edge_directions.append(match)
    edge_direction_indices = np.asarray(edge_directions, dtype=np.int64)
    boundary = set(mesh.boundary_loops[0].tolist())
    active_positions = np.asarray(
        [
            position
            for position, (first, second) in enumerate(edges.tolist())
            if not (first in boundary and second in boundary)
        ],
        dtype=np.int64,
    )
    graph = PositivePlanarGraph(
        name=name,
        mesh=mesh,
        direction_angles=directions,
        all_edges=edges,
        edge_direction_indices=edge_direction_indices,
        edge_faces=edge_faces,
        active_edges=edges[active_positions],
        active_edge_direction_indices=edge_direction_indices[active_positions],
        active_edge_faces=edge_faces[active_positions],
    )
    for values in (
        graph.direction_angles,
        graph.all_edges,
        graph.edge_direction_indices,
        graph.edge_faces,
        graph.active_edges,
        graph.active_edge_direction_indices,
        graph.active_edge_faces,
    ):
        values.setflags(write=False)
    return graph


def build_standard_square_graph(cells_per_side: int) -> PositivePlanarGraph:
    """Consistent SW--NE diagonal grid with directions 0, 45 and 90 degrees."""

    cells = _validate_cells(cells_per_side)
    return _make_graph(
        "standard_sw_ne",
        structured_rectangle(cells, cells),
        np.deg2rad(np.asarray([0.0, 45.0, 90.0])),
    )


def build_center_split_square_graph(cells_per_side: int) -> PositivePlanarGraph:
    """Split every square by its center and four corner spokes."""

    cells = _validate_cells(cells_per_side)
    vertices = _grid_vertices(cells).tolist()
    faces: list[tuple[int, int, int]] = []
    stride = cells + 1
    for j in range(cells):
        for i in range(cells):
            v00 = j * stride + i
            v10 = v00 + 1
            v01 = v00 + stride
            v11 = v01 + 1
            center = len(vertices)
            vertices.append([(i + 0.5) / cells, (j + 0.5) / cells])
            faces.extend(
                (
                    (v00, v10, center),
                    (v10, v11, center),
                    (v11, v01, center),
                    (v01, v00, center),
                )
            )
    mesh = TriMesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    return _make_graph(
        "center_split",
        mesh,
        np.deg2rad(np.asarray([0.0, 45.0, 90.0, 135.0])),
    )


def build_stellar_square_graph(cells_per_side: int) -> PositivePlanarGraph:
    """Stellarly refine each SW--NE parent triangle by its own centroid."""

    cells = _validate_cells(cells_per_side)
    parent = structured_rectangle(cells, cells)
    vertices = parent.vertices.tolist()
    faces: list[tuple[int, int, int]] = []
    for first, second, third in parent.faces.tolist():
        centroid = len(vertices)
        vertices.append(
            ((parent.vertices[first] + parent.vertices[second] + parent.vertices[third]) / 3.0).tolist()
        )
        faces.extend(
            (
                (first, second, centroid),
                (second, third, centroid),
                (third, first, centroid),
            )
        )
    mesh = TriMesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    return _make_graph(
        "stellar_triangle_centroids",
        mesh,
        np.asarray(
            [
                0.0,
                math.atan(0.5),
                math.pi / 4.0,
                math.atan(2.0),
                math.pi / 2.0,
                3.0 * math.pi / 4.0,
            ],
            dtype=np.float64,
        ),
    )


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    a = second - first
    b = third - first
    return float(a[0] * b[1] - a[1] * b[0])


def _positive_interval_overlap(a0: float, a1: float, b0: float, b1: float, tolerance: float) -> bool:
    return min(max(a0, a1), max(b0, b1)) - max(min(a0, a1), min(b0, b1)) > tolerance


def _point_on_segment(first: np.ndarray, second: np.ndarray, point: np.ndarray, tolerance: float) -> bool:
    return (
        abs(_orientation(first, second, point)) <= tolerance
        and min(first[0], second[0]) - tolerance <= point[0] <= max(first[0], second[0]) + tolerance
        and min(first[1], second[1]) - tolerance <= point[1] <= max(first[1], second[1]) + tolerance
    )


def audit_positive_planar_graph(graph: PositivePlanarGraph) -> PlanarGraphAudit:
    """Independently screen nonincident edge crossings and disk combinatorics."""

    vertices = graph.mesh.vertices
    edges = graph.all_edges
    tolerance = 128.0 * np.finfo(np.float64).eps
    crossings = 0
    overlaps = 0
    for left in range(len(edges)):
        a_index, b_index = map(int, edges[left])
        a, b = vertices[a_index], vertices[b_index]
        for right in range(left + 1, len(edges)):
            c_index, d_index = map(int, edges[right])
            if len({a_index, b_index, c_index, d_index}) < 4:
                continue
            c, d = vertices[c_index], vertices[d_index]
            ab_c = _orientation(a, b, c)
            ab_d = _orientation(a, b, d)
            cd_a = _orientation(c, d, a)
            cd_b = _orientation(c, d, b)
            if ab_c * ab_d < -(tolerance**2) and cd_a * cd_b < -(tolerance**2):
                crossings += 1
                continue
            if max(abs(ab_c), abs(ab_d), abs(cd_a), abs(cd_b)) <= tolerance:
                axis = 0 if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else 1
                if _positive_interval_overlap(a[axis], b[axis], c[axis], d[axis], tolerance):
                    overlaps += 1
                continue
            if any(
                (
                    _point_on_segment(a, b, c, tolerance),
                    _point_on_segment(a, b, d, tolerance),
                    _point_on_segment(c, d, a, tolerance),
                    _point_on_segment(c, d, b, tolerance),
                )
            ):
                crossings += 1
    loop_count = len(graph.mesh.boundary_loops)
    euler = graph.mesh.n_vertices - len(edges) + graph.mesh.n_faces
    system = DirectedTutteSystem.from_mesh(graph.mesh)
    disk = loop_count == 1 and euler == 1 and crossings == 0 and overlaps == 0
    return PlanarGraphAudit(
        proper_crossing_count=crossings,
        collinear_overlap_count=overlaps,
        boundary_loop_count=loop_count,
        euler_characteristic=euler,
        disk_topology=disk,
        dividing_edge_count=len(system.dividing_edges),
        vertex_count=graph.mesh.n_vertices,
        edge_count=len(edges),
        face_count=graph.mesh.n_faces,
    )


def beltrami_tensor(mu: complex | np.ndarray) -> np.ndarray:
    """Return the determinant-one conductivity tensor associated with ``mu``."""

    values = np.asarray(mu, dtype=np.complex128)
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError("mu must be finite")
    squared = values.real * values.real + values.imag * values.imag
    if np.any(squared >= 1.0):
        raise ValueError("mu must lie strictly inside the unit disk")
    denominator = 1.0 - squared
    result = np.empty(values.shape + (2, 2), dtype=np.float64)
    result[..., 0, 0] = (1.0 - 2.0 * values.real + squared) / denominator
    result[..., 0, 1] = -2.0 * values.imag / denominator
    result[..., 1, 0] = result[..., 0, 1]
    result[..., 1, 1] = (1.0 + 2.0 * values.real + squared) / denominator
    return result


def direction_design_matrix(direction_angles: np.ndarray) -> np.ndarray:
    """Frobenius-isometric design for ``sum c d d^T``.

    Symmetric matrices are vectorized as ``[A00, sqrt(2) A01, A11]`` so the
    Euclidean residual is exactly the Frobenius residual.
    """

    angles = np.asarray(direction_angles, dtype=np.float64)
    if angles.ndim != 1 or len(angles) == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("direction_angles must be a finite nonempty vector")
    cosine = np.cos(angles)
    sine = np.sin(angles)
    return np.vstack((cosine * cosine, _SQRT_TWO * cosine * sine, sine * sine))


def _symmetric_vector(tensor: np.ndarray) -> np.ndarray:
    return np.asarray([tensor[0, 0], _SQRT_TWO * tensor[0, 1], tensor[1, 1]])


def _vector_symmetric(vector: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[vector[0], vector[1] / _SQRT_TWO], [vector[1] / _SQRT_TWO, vector[2]]],
        dtype=np.float64,
    )


def _inverse_softplus_numpy(values: np.ndarray) -> np.ndarray:
    return values + np.log(-np.expm1(-values))


def fit_direction_tensor_nnls(
    tensor: np.ndarray,
    direction_angles: np.ndarray,
    *,
    minimum_conductance: float = 1.0e-6,
    minimum_excess: float = 1.0e-10,
) -> PositiveTensorFit:
    """Fit one SPD tensor by deterministic lower-bound NNLS.

    The actual lower bound is ``minimum_conductance + minimum_excess``.  The
    positive excess keeps the corresponding softplus logits finite; the
    returned logits realize the NNLS coefficients exactly up to floating point.
    """

    target = np.asarray(tensor, dtype=np.float64)
    if target.shape != (2, 2) or not np.all(np.isfinite(target)):
        raise ValueError("tensor must be a finite 2 by 2 matrix")
    if not np.allclose(target, target.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("tensor must be symmetric")
    if np.min(np.linalg.eigvalsh(target)) <= 0.0:
        raise ValueError("tensor must be positive definite")
    for name, value in (
        ("minimum_conductance", minimum_conductance),
        ("minimum_excess", minimum_excess),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    design = direction_design_matrix(direction_angles)
    target_vector = _symmetric_vector(target)
    lower = minimum_conductance + minimum_excess
    offset = np.full(design.shape[1], lower, dtype=np.float64)
    free, _ = nnls(design, target_vector - design @ offset)
    conductances = offset + free
    fitted_vector = design @ conductances
    fitted = _vector_symmetric(fitted_vector)
    difference = fitted - target
    target_frobenius = np.linalg.norm(target, ord="fro")
    target_spectral = np.linalg.norm(target, ord=2)
    logits = _inverse_softplus_numpy(conductances - minimum_conductance)
    for values in (conductances, logits, fitted):
        values.setflags(write=False)
    return PositiveTensorFit(
        conductances=conductances,
        logits=logits,
        fitted_tensor=fitted,
        frobenius_relative_error=float(np.linalg.norm(difference, ord="fro") / target_frobenius),
        local_operator_spectral_relative_error=float(
            np.linalg.norm(difference, ord=2) / target_spectral
        ),
        residual_norm=float(np.linalg.norm(fitted_vector - target_vector)),
        minimum_conductance=float(minimum_conductance),
        minimum_excess=float(minimum_excess),
    )


def inverse_softplus_conductances(
    conductances: torch.Tensor, minimum_conductance: float
) -> torch.Tensor:
    """Finite logits whose ``minimum + softplus`` equals ``conductances``."""

    if not isinstance(conductances, torch.Tensor):
        raise TypeError("conductances must be a torch tensor")
    if not math.isfinite(minimum_conductance) or minimum_conductance <= 0.0:
        raise ValueError("minimum_conductance must be finite and positive")
    excess = conductances - minimum_conductance
    if not bool(torch.isfinite(excess).all()) or bool(torch.any(excess <= 0.0)):
        raise ValueError("conductances must be finite and strictly above the minimum")
    return excess + torch.log(-torch.expm1(-excess))


class LearnedPositiveDirectionMap(nn.Module):
    """Small local MLP ``features(A) -> logits -> positive coefficients``."""

    def __init__(
        self,
        direction_angles: np.ndarray,
        *,
        hidden_features: int = 24,
        minimum_conductance: float = 1.0e-6,
        seed: int = 20260922,
    ) -> None:
        super().__init__()
        if isinstance(hidden_features, bool) or hidden_features < 1:
            raise ValueError("hidden_features must be positive")
        if not math.isfinite(minimum_conductance) or minimum_conductance <= 0.0:
            raise ValueError("minimum_conductance must be finite and positive")
        angles = np.array(direction_angles, dtype=np.float64, copy=True)
        design = direction_design_matrix(angles)
        outer_products = np.stack(
            [
                np.outer([math.cos(angle), math.sin(angle)], [math.cos(angle), math.sin(angle)])
                for angle in angles
            ]
        )
        self.minimum_conductance = float(minimum_conductance)
        self.register_buffer("direction_angles", torch.as_tensor(angles), persistent=True)
        self.register_buffer("design_matrix", torch.as_tensor(design), persistent=True)
        self.register_buffer("direction_outer_products", torch.as_tensor(outer_products), persistent=True)
        # The local fork makes construction reproducible without changing the
        # caller's random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(3, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, len(angles)),
            )

    @staticmethod
    def tensor_features(tensors: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensors, torch.Tensor):
            raise TypeError("tensors must be a torch tensor")
        if tensors.dtype not in (torch.float32, torch.float64):
            raise TypeError("tensors must be float32 or float64")
        if tensors.ndim < 2 or tensors.shape[-2:] != (2, 2):
            raise ValueError("tensors must have trailing shape (2,2)")
        if not bool(torch.isfinite(tensors).all()):
            raise ValueError("tensors must be finite")
        symmetry_tolerance = 64.0 * torch.finfo(tensors.dtype).eps
        if bool(torch.any(torch.abs(tensors[..., 0, 1] - tensors[..., 1, 0]) > symmetry_tolerance)):
            raise ValueError("tensors must be symmetric")
        trace = tensors[..., 0, 0] + tensors[..., 1, 1]
        determinant = tensors[..., 0, 0] * tensors[..., 1, 1] - tensors[..., 0, 1].square()
        if bool(torch.any(trace <= 0.0)) or bool(torch.any(determinant <= 0.0)):
            raise ValueError("tensors must be positive definite")
        return torch.stack(
            (
                torch.log(trace),
                (tensors[..., 0, 0] - tensors[..., 1, 1]) / trace,
                2.0 * tensors[..., 0, 1] / trace,
            ),
            dim=-1,
        )

    def forward(self, tensors: torch.Tensor) -> torch.Tensor:
        logits = self.network(self.tensor_features(tensors))
        values = self.minimum_conductance + torch_functional.softplus(logits)
        if not bool(torch.isfinite(values).all()) or bool(torch.any(values <= 0.0)):
            raise RuntimeError("learned conductances must remain finite and strictly positive")
        return values

    def fitted_tensors(self, conductances: torch.Tensor) -> torch.Tensor:
        if conductances.shape[-1] != len(self.direction_angles):
            raise ValueError("conductance direction dimension does not match the projector")
        outer = self.direction_outer_products.to(
            device=conductances.device, dtype=conductances.dtype
        )
        return torch.einsum("...m,mij->...ij", conductances, outer)


def aggregate_face_direction_conductances(
    graph: PositivePlanarGraph, face_direction_conductances: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Average adjacent face predictions onto shared active graph edges.

    This positive aggregation defines the global student graph.  It is not the
    solution of a global operator-projection problem.
    """

    expected = (graph.mesh.n_faces, len(graph.direction_angles))
    if isinstance(face_direction_conductances, torch.Tensor):
        values = face_direction_conductances
        if tuple(values.shape[-2:]) != expected:
            raise ValueError(f"face conductances must have trailing shape {expected}")
        if not bool(torch.isfinite(values).all()) or bool(torch.any(values <= 0.0)):
            raise ValueError("face conductances must be finite and strictly positive")
        device = values.device
        faces = torch.tensor(
            np.array(graph.active_edge_faces, copy=True), dtype=torch.int64, device=device
        )
        directions = torch.tensor(
            np.array(graph.active_edge_direction_indices, copy=True),
            dtype=torch.int64,
            device=device,
        )
        first = values[..., faces[:, 0], directions]
        has_second = faces[:, 1] >= 0
        safe_second = torch.where(has_second, faces[:, 1], faces[:, 0])
        second = values[..., safe_second, directions]
        return torch.where(has_second, 0.5 * (first + second), first)
    values = np.asarray(face_direction_conductances, dtype=np.float64)
    if tuple(values.shape[-2:]) != expected:
        raise ValueError(f"face conductances must have trailing shape {expected}")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("face conductances must be finite and strictly positive")
    faces = graph.active_edge_faces
    directions = graph.active_edge_direction_indices
    first = values[..., faces[:, 0], directions]
    safe_second = np.where(faces[:, 1] >= 0, faces[:, 1], faces[:, 0])
    second = values[..., safe_second, directions]
    return np.where(faces[:, 1] >= 0, 0.5 * (first + second), first)


class PositiveHodgeTutteLayer(nn.Module):
    """Local positive tensor map, shared-edge assembly, and hard Tutte solve.

    A successful return has passed the symmetric layer's fixed-disk,
    boundary/dividing-edge, and strict-positive-face checks.  No Euler proposal
    and no fold repair can be returned by this wrapper.
    """

    def __init__(
        self,
        graph: PositivePlanarGraph,
        projector: LearnedPositiveDirectionMap,
        *,
        minimum_conductance: float | None = None,
        relative_tolerance: float | None = None,
        absolute_tolerance: float = 0.0,
        max_iterations: int = 1000,
    ) -> None:
        super().__init__()
        if not isinstance(projector, LearnedPositiveDirectionMap):
            raise TypeError("projector must be a LearnedPositiveDirectionMap")
        angles = projector.direction_angles.detach().cpu().numpy()
        if angles.shape != graph.direction_angles.shape or not np.allclose(
            angles, graph.direction_angles, rtol=0.0, atol=1.0e-14
        ):
            raise ValueError("projector directions do not match the planar graph")
        floor = projector.minimum_conductance if minimum_conductance is None else minimum_conductance
        if not math.isclose(floor, projector.minimum_conductance, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("projector and solver minimum conductance must match exactly")
        self.graph = graph
        self.projector = projector
        self.solver = MatrixFreeSymmetricTutteLayer(
            graph.mesh,
            minimum_conductance=floor,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            max_iterations=max_iterations,
        )
        if not np.array_equal(self.solver.active_edges, graph.active_edges):
            raise RuntimeError("graph and symmetric solver active-edge order disagree")
        self.system = self.solver.system
        self.last_topology_certified: bool | None = None
        self.last_edge_conductances: torch.Tensor | None = None

    def project_edge_conductances(self, face_tensors: torch.Tensor) -> torch.Tensor:
        face_values = self.projector(face_tensors)
        edges = aggregate_face_direction_conductances(self.graph, face_values)
        assert isinstance(edges, torch.Tensor)
        return edges

    def forward(self, face_tensors: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        self.last_topology_certified = False
        edge_values = self.project_edge_conductances(face_tensors)
        logits = inverse_softplus_conductances(
            edge_values, self.solver.minimum_conductance
        )
        mapped = self.solver(logits, boundary)
        self.last_edge_conductances = edge_values.detach().clone()
        self.last_topology_certified = True
        return mapped


def edge_conductance_gradient(
    solver: MatrixFreeSymmetricTutteLayer,
    control: np.ndarray,
    adjoint_interior: np.ndarray,
) -> np.ndarray:
    """Evaluate ``-(B0 lambda)_e dot (B0 x)_e`` on active edges.

    Boundary adjoint values are zero because the identity is for a fixed
    Dirichlet boundary.  ``adjoint_interior`` solves ``L lambda=dLoss/dx``.
    """

    mapped = np.asarray(control, dtype=np.float64)
    adjoint_values = np.asarray(adjoint_interior, dtype=np.float64)
    if mapped.ndim != 2 or mapped.shape[0] != solver.system.n_vertices:
        raise ValueError("control must have shape (n_vertices, channels)")
    if adjoint_values.shape != (solver.n_interior, mapped.shape[1]):
        raise ValueError("adjoint_interior has the wrong shape")
    if not np.all(np.isfinite(mapped)) or not np.all(np.isfinite(adjoint_values)):
        raise ValueError("control and adjoint must be finite")
    full_adjoint = np.zeros_like(mapped)
    full_adjoint[solver.interior_vertices] = adjoint_values
    first = solver.active_edges[:, 0]
    second = solver.active_edges[:, 1]
    primal_difference = mapped[first] - mapped[second]
    adjoint_difference = full_adjoint[first] - full_adjoint[second]
    return -np.sum(primal_difference * adjoint_difference, axis=1)


__all__ = [
    "LearnedPositiveDirectionMap",
    "PlanarGraphAudit",
    "PositiveHodgeTutteLayer",
    "PositivePlanarGraph",
    "PositiveTensorFit",
    "aggregate_face_direction_conductances",
    "audit_positive_planar_graph",
    "beltrami_tensor",
    "build_center_split_square_graph",
    "build_standard_square_graph",
    "build_stellar_square_graph",
    "direction_design_matrix",
    "edge_conductance_gradient",
    "fit_direction_tensor_nnls",
    "inverse_softplus_conductances",
]
