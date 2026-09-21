"""Implicit differentiable decoder for directed positive-row Tutte systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

from ..mesh import TriMesh


@dataclass
class DirectedTutteSystem:
    """Fixed graph data for a row-stochastic directed barycentric solve."""

    loop: np.ndarray
    interior: np.ndarray
    neighbors: np.ndarray
    neighbor_is_boundary: np.ndarray
    valid_mask: np.ndarray
    faces: np.ndarray
    n_vertices: int
    max_degree: int

    @property
    def n_rows(self) -> int:
        return int(len(self.interior))

    @classmethod
    def from_mesh(cls, mesh: TriMesh) -> "DirectedTutteSystem":
        if len(mesh.boundary_loops) != 1:
            raise ValueError("directed Tutte embedding requires exactly one boundary loop")
        used_vertices = np.unique(mesh.faces)
        if not np.array_equal(used_vertices, np.arange(mesh.n_vertices, dtype=np.int64)):
            raise ValueError("directed Tutte embedding requires every vertex to belong to a face")
        loop = np.asarray(mesh.boundary_loops[0], dtype=np.int64)
        boundary_set = set(loop.tolist())
        interior = np.asarray([v for v in range(mesh.n_vertices) if v not in boundary_set], dtype=np.int64)
        index = {int(v): i for i, v in enumerate(interior.tolist())}
        boundary_index = {int(v): i for i, v in enumerate(loop.tolist())}
        adjacency = [set() for _ in range(mesh.n_vertices)]
        for a, b, c in mesh.faces.tolist():
            adjacency[a].update((b, c))
            adjacency[b].update((a, c))
            adjacency[c].update((a, b))
        degrees = [len(adjacency[int(v)]) for v in interior]
        max_degree = max(degrees, default=0)
        neighbors = np.full((len(interior), max_degree), -1, dtype=np.int64)
        is_boundary = np.zeros_like(neighbors, dtype=bool)
        valid = np.zeros_like(neighbors, dtype=bool)
        for row, vertex in enumerate(interior.tolist()):
            for col, neighbor in enumerate(sorted(adjacency[int(vertex)])):
                if neighbor in boundary_index:
                    neighbors[row, col] = boundary_index[neighbor]
                    is_boundary[row, col] = True
                else:
                    neighbors[row, col] = index[neighbor]
                valid[row, col] = True
        return cls(
            loop,
            interior,
            neighbors,
            is_boundary,
            valid,
            np.asarray(mesh.faces, dtype=np.int64),
            mesh.n_vertices,
            max_degree,
        )

    def _probabilities(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (self.n_rows, self.max_degree):
            raise ValueError("logits must have shape (interior_vertices, max_degree)")
        if not np.all(np.isfinite(values)):
            raise ValueError("logits must be finite")
        masked = np.where(self.valid_mask, values, -np.inf)
        shifted = masked - np.max(masked, axis=1, keepdims=True)
        weights = np.exp(shifted)
        weights[~self.valid_mask] = 0.0
        if np.any(weights[self.valid_mask] == 0.0):
            raise ValueError("a supported softmax probability underflowed to zero")
        denominator = np.sum(weights, axis=1, keepdims=True)
        return weights / denominator

    def _assemble(self, probabilities: np.ndarray):
        matrix = lil_matrix((self.n_rows, self.n_rows), dtype=np.float64)
        coupling = lil_matrix((self.n_rows, len(self.loop)), dtype=np.float64)
        for row in range(self.n_rows):
            matrix[row, row] = 1.0
            for col in np.flatnonzero(self.valid_mask[row]):
                weight = float(probabilities[row, col])
                neighbor = int(self.neighbors[row, col])
                if self.neighbor_is_boundary[row, col]:
                    coupling[row, neighbor] += weight
                else:
                    matrix[row, neighbor] -= weight
        return matrix.tocsr(), coupling.tocsr()


def _validate_weakly_convex_boundary(boundary: np.ndarray, tolerance: float = 1e-12) -> None:
    """Validate a positively oriented convex cycle, allowing side subdivisions."""

    points = np.asarray(boundary, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("target boundary must contain at least three 2D points")
    if not np.all(np.isfinite(points)):
        raise ValueError("target boundary must be finite")
    edges = np.roll(points, -1, axis=0) - points
    if np.any(np.sum(edges * edges, axis=1) <= tolerance * tolerance):
        raise ValueError("target boundary has a repeated consecutive vertex")
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        delta1 = b - a
        delta2 = c - a
        return float(delta1[0] * delta2[1] - delta1[1] * delta2[0])

    def on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
        lower = np.minimum(a, b) - tolerance
        upper = np.maximum(a, b) + tolerance
        return bool(abs(orient(a, b, p)) <= tolerance and np.all(p >= lower) and np.all(p <= upper))

    def intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        ab_c, ab_d = orient(a, b, c), orient(a, b, d)
        cd_a, cd_b = orient(c, d, a), orient(c, d, b)
        if ((ab_c > tolerance and ab_d < -tolerance) or (ab_c < -tolerance and ab_d > tolerance)) and ((cd_a > tolerance and cd_b < -tolerance) or (cd_a < -tolerance and cd_b > tolerance)):
            return True
        return on_segment(a, b, c) or on_segment(a, b, d) or on_segment(c, d, a) or on_segment(c, d, b)

    n_points = points.shape[0]
    for first in range(n_points):
        for second in range(first + 1, n_points):
            if second in (first, (first + 1) % n_points) or first in ((second + 1) % n_points,):
                continue
            if intersects(points[first], points[(first + 1) % n_points], points[second], points[(second + 1) % n_points]):
                raise ValueError("target boundary must be a simple polygon")
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    signed_area_twice = float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)))
    if signed_area_twice <= tolerance or float(np.min(turns)) < -tolerance:
        raise ValueError("target boundary must be a positively oriented weakly convex polygon")
    if int(np.count_nonzero(turns > tolerance)) < 3:
        raise ValueError("target boundary is degenerate rather than strictly convex at its corners")


def _validate_strictly_positive_faces(
    faces: np.ndarray,
    mapped: np.ndarray,
) -> None:
    """Numerically certify strict orientation on every original face."""
    triangles = mapped[faces]
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    twice_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    scale = max(float(np.ptp(mapped[:, 0])), float(np.ptp(mapped[:, 1])), 1.0)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale * scale
    if np.any(twice_area <= tolerance):
        raise ValueError("directed Tutte output does not have strictly positive face areas")


def rectangle_boundary_from_logits(
    logits: torch.Tensor,
    *,
    width: float | torch.Tensor = 1.0,
    height: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Parameterize a strictly ordered rectangular boundary by side logits.

    ``logits`` has shape ``(4, n_edges_per_side)``.  Each row is converted by
    a softmax into positive segment lengths whose sum is respectively
    ``width``, ``height``, ``width`` and ``height``.  The returned vertices are
    ordered counter-clockwise, with the final closing segment implicit.  Thus
    this layer can be used before :func:`directed_tutte_embedding_torch_implicit`
    without ever supplying a self-intersecting or non-monotone boundary.
    It controls boundary sampling and rectangular aspect ratio, not arbitrary
    convex-polygon shape.
    """
    if logits.ndim != 2 or logits.shape[0] != 4 or logits.shape[1] < 1:
        raise ValueError("logits must have shape (4, n_edges_per_side), n_edges_per_side >= 1")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    width_tensor = _positive_scalar_parameter(width, logits, "width")
    height_tensor = _positive_scalar_parameter(height, logits, "height")
    lengths = torch.stack(
        (
            torch.softmax(logits[0], dim=0) * width_tensor,
            torch.softmax(logits[1], dim=0) * height_tensor,
            torch.softmax(logits[2], dim=0) * width_tensor,
            torch.softmax(logits[3], dim=0) * height_tensor,
        )
    )
    directions = torch.as_tensor(
        ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
        dtype=logits.dtype,
        device=logits.device,
    )
    current = torch.zeros(2, dtype=logits.dtype, device=logits.device)
    vertices = []
    for side in range(4):
        for segment in lengths[side]:
            vertices.append(current)
            current = current + directions[side] * segment
    return torch.stack(vertices, dim=0)


def _positive_scalar_parameter(
    value: float | torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or not torch.is_floating_point(value):
            raise ValueError(f"{name} must be a floating-point scalar tensor")
        scalar = value.to(device=reference.device, dtype=reference.dtype)
    else:
        scalar = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if not bool(torch.isfinite(scalar).item()) or not bool((scalar > 0.0).item()):
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def rectangle_boundary_from_modulus_logits(
    logits: torch.Tensor,
    modulus_logit: torch.Tensor,
    *,
    width: float | torch.Tensor = 1.0,
    min_height: float = 1e-6,
) -> torch.Tensor:
    """Create the ordered rectangle boundary with a learnable positive height."""
    if not isinstance(modulus_logit, torch.Tensor) or modulus_logit.ndim != 0:
        raise ValueError("modulus_logit must be a scalar tensor")
    if not torch.is_floating_point(modulus_logit) or not torch.isfinite(modulus_logit).item():
        raise ValueError("modulus_logit must be finite and floating point")
    if min_height <= 0.0 or not np.isfinite(min_height):
        raise ValueError("min_height must be finite and positive")
    modulus_logit = modulus_logit.to(device=logits.device, dtype=logits.dtype)
    height = float(min_height) + torch.nn.functional.softplus(modulus_logit)
    return rectangle_boundary_from_logits(logits, width=width, height=height)


class _DirectedTutteImplicitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target_boundary: torch.Tensor, logits: torch.Tensor, system: DirectedTutteSystem):
        if target_boundary.ndim != 2 or target_boundary.shape[1] != 2:
            raise ValueError("target_boundary must have shape (boundary_vertices, 2)")
        if logits.ndim != 2 or logits.shape != (system.n_rows, system.max_degree):
            raise ValueError("logits has an incompatible shape")
        if not torch.is_floating_point(target_boundary) or not torch.is_floating_point(logits):
            raise ValueError("target_boundary and logits must be floating point")
        boundary = target_boundary.detach().cpu().numpy().astype(np.float64, copy=False)
        _validate_weakly_convex_boundary(boundary)
        logit_values = logits.detach().cpu().numpy().astype(np.float64, copy=False)
        if boundary.shape[0] != len(system.loop):
            raise ValueError("target_boundary has the wrong number of vertices")
        if system.n_rows == 0:
            ctx.empty = True
            ctx.logit_dtype = logits.dtype
            ctx.logit_device = logits.device
            ctx.loop = system.loop
            output = np.empty((system.n_vertices, 2), dtype=np.float64)
            output[system.loop] = boundary
            _validate_strictly_positive_faces(system.faces, output)
            return torch.as_tensor(
                output,
                dtype=target_boundary.dtype,
                device=target_boundary.device,
            )
        ctx.empty = False
        probabilities = system._probabilities(logit_values)
        matrix, coupling = system._assemble(probabilities)
        solve = factorized(matrix.tocsc())
        solve_transpose = factorized(matrix.T.tocsc())
        interior_values = np.asarray(solve(coupling @ boundary), dtype=np.float64)
        output = np.empty((system.n_vertices, 2), dtype=np.float64)
        output[system.loop] = boundary
        output[system.interior] = interior_values
        _validate_strictly_positive_faces(system.faces, output)
        ctx.system = system
        ctx.device = target_boundary.device
        ctx.boundary_dtype = target_boundary.dtype
        ctx.logit_dtype = logits.dtype
        ctx.probabilities = probabilities
        ctx.boundary = boundary
        ctx.interior_values = interior_values
        ctx.coupling = coupling
        ctx.solve_transpose = solve_transpose
        return torch.as_tensor(output, dtype=target_boundary.dtype, device=target_boundary.device)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        if ctx.empty:
            loop = torch.as_tensor(ctx.loop.copy(), device=gradient.device)
            return gradient[loop], torch.zeros((0, 0), dtype=ctx.logit_dtype, device=ctx.logit_device), None
        system: DirectedTutteSystem = ctx.system
        grad = gradient.detach().cpu().numpy().astype(np.float64, copy=False)
        adjoint = np.asarray(ctx.solve_transpose(grad[system.interior]), dtype=np.float64)
        boundary_gradient = grad[system.loop] + np.asarray(ctx.coupling.T @ adjoint, dtype=np.float64)
        logits_gradient = np.zeros_like(ctx.probabilities)
        for row in range(system.n_rows):
            x_row = ctx.interior_values[row]
            adjoint_row = adjoint[row]
            for col in np.flatnonzero(system.valid_mask[row]):
                if system.neighbor_is_boundary[row, col]:
                    neighbor_value = ctx.boundary[system.neighbors[row, col]]
                else:
                    neighbor_value = ctx.interior_values[system.neighbors[row, col]]
                logits_gradient[row, col] = ctx.probabilities[row, col] * float(
                    np.dot(adjoint_row, neighbor_value - x_row)
                )
        return (
            torch.as_tensor(boundary_gradient, dtype=ctx.boundary_dtype, device=ctx.device),
            torch.as_tensor(logits_gradient, dtype=ctx.logit_dtype, device=ctx.device),
            None,
        )


def directed_tutte_embedding_torch_implicit(
    mesh: TriMesh,
    target_boundary: torch.Tensor,
    logits: torch.Tensor,
    system: DirectedTutteSystem | None = None,
) -> torch.Tensor:
    """Decode a directed positive-row Tutte system with sparse implicit VJPs."""
    if system is None:
        system = DirectedTutteSystem.from_mesh(mesh)
    if system.n_vertices != mesh.n_vertices:
        raise ValueError("system was built for a different mesh")
    if target_boundary.device != logits.device:
        raise ValueError("target_boundary and logits must be on the same device")
    return _DirectedTutteImplicitFunction.apply(target_boundary, logits, system)
