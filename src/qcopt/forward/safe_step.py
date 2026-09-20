"""Determinant-margin safe steps for explicit piecewise-linear residual updates."""

from __future__ import annotations

import numpy as np

from ..beltrami import face_jacobians
from ..mesh import TriMesh


def maximum_safe_step(
    mesh: TriMesh,
    uv: np.ndarray,
    delta: np.ndarray,
    *,
    min_det_margin: float = 1e-10,
) -> float:
    """Return a conservative first determinant-crossing step over all faces."""

    if min_det_margin <= 0.0 or not np.isfinite(min_det_margin):
        raise ValueError("min_det_margin must be positive and finite")
    value, _ = maximum_safe_step_with_active_face(
        mesh, uv, delta, min_det_margin=min_det_margin
    )
    return value


def maximum_safe_step_with_active_face(
    mesh: TriMesh,
    uv: np.ndarray,
    delta: np.ndarray,
    *,
    min_det_margin: float = 1e-10,
) -> tuple[float, int]:
    """Return the safe step and the face attaining the first root.

    The active face is useful for a piecewise-smooth custom VJP. At ties the
    first face in mesh order is returned; callers should treat such ties as a
    nondifferentiable active-set boundary.
    """

    if min_det_margin <= 0.0 or not np.isfinite(min_det_margin):
        raise ValueError("min_det_margin must be positive and finite")
    uv = _validate_map(mesh, uv, "uv")
    delta = _validate_map(mesh, delta, "delta")
    base = face_jacobians(mesh, uv)
    direction = face_jacobians(mesh, delta)
    limits: list[tuple[float, int]] = []
    for face_index, (a_mat, b_mat) in enumerate(zip(base, direction)):
        c = float(np.linalg.det(a_mat))
        if c <= min_det_margin:
            raise ValueError("current map is at or below the determinant margin")
        a = float(np.linalg.det(b_mat))
        b = float(
            a_mat[0, 0] * b_mat[1, 1]
            + b_mat[0, 0] * a_mat[1, 1]
            - a_mat[0, 1] * b_mat[1, 0]
            - b_mat[0, 1] * a_mat[1, 0]
        )
        constant = c - min_det_margin
        if abs(a) <= 1e-15:
            if b < 0.0:
                limits.append((-constant / b, face_index))
            continue
        discriminant = b * b - 4.0 * a * constant
        if discriminant < 0.0:
            continue
        root = np.sqrt(max(discriminant, 0.0))
        candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
        positive = [value for value in candidates if value > 0.0 and np.isfinite(value)]
        if positive:
            limits.append((min(positive), face_index))
    if not limits:
        return float("inf"), -1
    return min(limits, key=lambda item: item[0])


def certified_residual_step(
    mesh: TriMesh,
    uv: np.ndarray,
    delta: np.ndarray,
    *,
    min_det_margin: float = 1e-10,
    safety: float = 0.99,
) -> tuple[np.ndarray, float]:
    """Apply a residual with a determinant-certified scalar step."""

    if not 0.0 < safety < 1.0:
        raise ValueError("safety must lie strictly between zero and one")
    uv = _validate_map(mesh, uv, "uv")
    delta = _validate_map(mesh, delta, "delta")
    bound = maximum_safe_step(
        mesh, uv, delta, min_det_margin=min_det_margin
    )
    step = safety * bound if np.isfinite(bound) else 1.0
    return np.ascontiguousarray(uv + step * delta), float(step)


def certified_step_toward_target(
    mesh: TriMesh,
    uv: np.ndarray,
    target: np.ndarray,
    *,
    min_det_margin: float = 1e-10,
    safety: float = 0.99,
) -> tuple[np.ndarray, float]:
    """Advance along the straight residual toward ``target`` without overshoot.

    Unlike :func:`certified_residual_step`, this helper caps the scalar at one,
    because the direction is the full remaining displacement.  It exposes a
    fundamental limitation of straight residual continuation: two valid maps
    can be separated by a singular linear homotopy, in which case repeated
    safe steps approach the barrier but cannot reach the target.
    """

    if not 0.0 < safety < 1.0:
        raise ValueError("safety must lie strictly between zero and one")
    current = _validate_map(mesh, uv, "uv")
    destination = _validate_map(mesh, target, "target")
    delta = destination - current
    bound = maximum_safe_step(
        mesh, current, delta, min_det_margin=min_det_margin
    )
    if not np.isfinite(bound) or bound >= 1.0:
        step = 1.0
    else:
        step = safety * bound
    return np.ascontiguousarray(current + step * delta), float(step)


def maximum_safe_step_directional_derivative(
    mesh: TriMesh,
    uv: np.ndarray,
    delta: np.ndarray,
    d_uv: np.ndarray,
    d_delta: np.ndarray,
    *,
    min_det_margin: float = 1e-10,
) -> float:
    """Differentiate the active-root step within a fixed active set.

    At a tie between faces this derivative is not defined; callers should
    detect active-face changes and use a nonsmooth/KKT treatment instead.
    """

    uv = _validate_map(mesh, uv, "uv")
    delta = _validate_map(mesh, delta, "delta")
    d_uv = _validate_map(mesh, d_uv, "d_uv")
    d_delta = _validate_map(mesh, d_delta, "d_delta")
    step, active = maximum_safe_step_with_active_face(
        mesh, uv, delta, min_det_margin=min_det_margin
    )
    if active < 0 or not np.isfinite(step):
        return 0.0
    base = face_jacobians(mesh, uv)[active]
    direction = face_jacobians(mesh, delta)[active]
    d_base = face_jacobians(mesh, d_uv)[active]
    d_direction = face_jacobians(mesh, d_delta)[active]

    def det_derivative(matrix: np.ndarray, variation: np.ndarray) -> float:
        return float(
            variation[0, 0] * matrix[1, 1]
            + matrix[0, 0] * variation[1, 1]
            - variation[0, 1] * matrix[1, 0]
            - matrix[0, 1] * variation[1, 0]
        )

    a = float(np.linalg.det(direction))
    b = float(
        base[0, 0] * direction[1, 1]
        + direction[0, 0] * base[1, 1]
        - base[0, 1] * direction[1, 0]
        - direction[0, 1] * base[1, 0]
    )
    da = det_derivative(direction, d_direction)
    db = float(
        d_base[0, 0] * direction[1, 1]
        + base[0, 0] * d_direction[1, 1]
        + d_direction[0, 0] * base[1, 1]
        + direction[0, 0] * d_base[1, 1]
        - d_base[0, 1] * direction[1, 0]
        - base[0, 1] * d_direction[1, 0]
        - d_direction[0, 1] * base[1, 0]
        - direction[0, 1] * d_base[1, 0]
    )
    dc = det_derivative(base, d_base)
    denominator = 2.0 * a * step + b
    if abs(denominator) < 1e-14:
        raise ValueError("active determinant root is multiple or ill-conditioned")
    return float(-(step * step * da + step * db + dc) / denominator)


def _validate_map(mesh: TriMesh, values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (mesh.n_vertices, 2) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite (n_vertices, 2) array")
    return values
