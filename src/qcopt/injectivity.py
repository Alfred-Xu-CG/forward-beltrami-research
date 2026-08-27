"""Independent floating-point injectivity audit for PL disk maps."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi

import numpy as np

from .beltrami import face_jacobians
from .mesh import TriMesh


@dataclass(frozen=True)
class InjectivityReport:
    certified: bool
    flipped_faces: tuple[int, ...]
    boundary_intersections: tuple[tuple[int, int], ...]
    bad_branch_vertices: tuple[int, ...]
    rectangle_side_violations: tuple[str, ...]
    minimum_signed_area_ratio: float
    boundary_orientation_ok: bool
    topology_ok: bool


def audit_injectivity(
    mesh: TriMesh,
    uv: np.ndarray,
    *,
    rectangle: bool = False,
    tolerance: float = 1e-12,
) -> InjectivityReport:
    """Audit a disk map without using any training energy as evidence.

    The result is a numerical discrete certificate, not an exact-predicate proof.
    """

    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape != (mesh.n_vertices, 2) or not np.all(np.isfinite(uv)):
        raise ValueError("uv must be a finite (mesh.n_vertices, 2) array")
    target_scale = max(float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1])), 1.0)
    length_tol = tolerance * target_scale
    area_tol = tolerance * target_scale * target_scale

    determinants = np.linalg.det(face_jacobians(mesh, uv))
    flipped = tuple(np.flatnonzero(determinants <= area_tol).astype(int).tolist())
    minimum_ratio = float(np.min(determinants))

    topology_ok = len(mesh.boundary_loops) == 1
    intersections: tuple[tuple[int, int], ...] = ()
    boundary_orientation_ok = False
    boundary_vertices: set[int] = set()
    if topology_ok:
        loop = mesh.boundary_loops[0]
        boundary_vertices = set(loop.tolist())
        polygon = uv[loop]
        twice_area = float(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
        boundary_orientation_ok = twice_area > area_tol
        intersections = _boundary_intersections(polygon, length_tol)

    bad_branches = _bad_branch_vertices(
        mesh, uv, boundary_vertices, length_tol, tolerance
    )
    rectangle_violations = (
        _rectangle_violations(mesh, uv, length_tol) if rectangle else ()
    )
    certified = bool(
        topology_ok
        and not flipped
        and not intersections
        and boundary_orientation_ok
        and not bad_branches
        and not rectangle_violations
    )
    return InjectivityReport(
        certified=certified,
        flipped_faces=flipped,
        boundary_intersections=intersections,
        bad_branch_vertices=bad_branches,
        rectangle_side_violations=rectangle_violations,
        minimum_signed_area_ratio=minimum_ratio,
        boundary_orientation_ok=boundary_orientation_ok,
        topology_ok=topology_ok,
    )


def _boundary_intersections(
    polygon: np.ndarray, tolerance: float
) -> tuple[tuple[int, int], ...]:
    count = len(polygon)
    collisions: list[tuple[int, int]] = []
    for first in range(count):
        a0, a1 = polygon[first], polygon[(first + 1) % count]
        for second in range(first + 1, count):
            if second == first or second == (first + 1) % count:
                continue
            if first == (second + 1) % count:
                continue
            b0, b1 = polygon[second], polygon[(second + 1) % count]
            if _segments_intersect(a0, a1, b0, b1, tolerance):
                collisions.append((first, second))
    return tuple(collisions)


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: float,
) -> bool:
    if (
        max(a[0], b[0]) + tolerance < min(c[0], d[0])
        or max(c[0], d[0]) + tolerance < min(a[0], b[0])
        or max(a[1], b[1]) + tolerance < min(c[1], d[1])
        or max(c[1], d[1]) + tolerance < min(a[1], b[1])
    ):
        return False

    def orient(p, q, r):
        delta1, delta2 = q - p, r - p
        return float(delta1[0] * delta2[1] - delta1[1] * delta2[0])

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    strict = (o1 > tolerance and o2 < -tolerance or o1 < -tolerance and o2 > tolerance) and (
        o3 > tolerance and o4 < -tolerance or o3 < -tolerance and o4 > tolerance
    )
    if strict:
        return True

    def on_segment(p, q, r):
        return (
            min(p[0], q[0]) - tolerance <= r[0] <= max(p[0], q[0]) + tolerance
            and min(p[1], q[1]) - tolerance <= r[1] <= max(p[1], q[1]) + tolerance
        )

    return bool(
        (abs(o1) <= tolerance and on_segment(a, b, c))
        or (abs(o2) <= tolerance and on_segment(a, b, d))
        or (abs(o3) <= tolerance and on_segment(c, d, a))
        or (abs(o4) <= tolerance and on_segment(c, d, b))
    )


def _bad_branch_vertices(
    mesh: TriMesh,
    uv: np.ndarray,
    boundary_vertices: set[int],
    length_tolerance: float,
    winding_tolerance: float,
) -> tuple[int, ...]:
    successors: list[dict[int, int]] = [dict() for _ in range(mesh.n_vertices)]
    for a, b, c in mesh.faces.tolist():
        for center, first, second in ((a, b, c), (b, c, a), (c, a, b)):
            prior = successors[center].get(first)
            if prior is not None and prior != second:
                successors[center][first] = -1
            else:
                successors[center][first] = second

    bad: list[int] = []
    for center in range(mesh.n_vertices):
        if center in boundary_vertices:
            continue
        relation = successors[center]
        if not relation or -1 in relation.values():
            bad.append(center)
            continue
        start = min(relation)
        ordered = [start]
        current = relation[start]
        while current != start and len(ordered) <= len(relation):
            if current not in relation or current in ordered:
                break
            ordered.append(current)
            current = relation[current]
        if current != start or len(ordered) != len(relation):
            bad.append(center)
            continue
        rays = uv[np.asarray(ordered)] - uv[center]
        norms = np.linalg.norm(rays, axis=1)
        if np.any(norms <= length_tolerance):
            bad.append(center)
            continue
        angle_sum = 0.0
        for index, ray in enumerate(rays):
            next_ray = rays[(index + 1) % len(rays)]
            cross = ray[0] * next_ray[1] - ray[1] * next_ray[0]
            dot = float(ray @ next_ray)
            angle_sum += atan2(float(cross), dot)
        if abs(angle_sum - 2.0 * pi) > max(1e-8, 100.0 * winding_tolerance):
            bad.append(center)
    return tuple(bad)


def _rectangle_violations(
    mesh: TriMesh, uv: np.ndarray, tolerance: float
) -> tuple[str, ...]:
    source = mesh.vertices
    x_min, y_min = source.min(axis=0)
    x_max, y_max = source.max(axis=0)
    source_scale = max(x_max - x_min, y_max - y_min, 1.0)
    source_tol = 1e-12 * source_scale
    boundary = set(mesh.boundary_loops[0].tolist()) if mesh.boundary_loops else set()
    side_specs = (
        ("left", 0, x_min, 1),
        ("right", 0, x_max, 1),
        ("bottom", 1, y_min, 0),
        ("top", 1, y_max, 0),
    )
    violations: list[str] = []
    for name, normal_axis, normal_value, tangent_axis in side_specs:
        vertices = np.asarray(
            [
                vertex
                for vertex in boundary
                if abs(source[vertex, normal_axis] - normal_value) <= source_tol
            ],
            dtype=np.int64,
        )
        if len(vertices) < 2:
            violations.append(f"{name}-missing")
            continue
        if np.any(np.abs(uv[vertices, normal_axis] - normal_value) > tolerance):
            violations.append(f"{name}-membership")
        order = np.argsort(source[vertices, tangent_axis])
        tangents = uv[vertices[order], tangent_axis]
        if np.any(np.diff(tangents) <= tolerance):
            violations.append(f"{name}-order")
    return tuple(violations)
