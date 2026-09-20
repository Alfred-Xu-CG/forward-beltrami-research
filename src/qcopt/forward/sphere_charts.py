"""Numerically stable north/south stereographic chart helpers."""

from __future__ import annotations

import numpy as np


def stereographic_forward(points: np.ndarray, *, pole: str) -> np.ndarray:
    """Project unit-sphere points from ``pole`` to a complex chart."""

    values = _validate_points(points)
    if pole not in {"north", "south"}:
        raise ValueError("pole must be 'north' or 'south'")
    denominator = 1.0 - values[:, 2] if pole == "north" else 1.0 + values[:, 2]
    if np.any(denominator <= 1e-14):
        raise ValueError("points too close to the projection pole")
    numerator = (
        values[:, 0] + 1j * values[:, 1]
        if pole == "north"
        else values[:, 0] - 1j * values[:, 1]
    )
    return numerator / denominator


def stereographic_inverse(values: np.ndarray, *, pole: str) -> np.ndarray:
    """Map complex chart coordinates back to the unit sphere."""

    chart = np.asarray(values, dtype=np.complex128)
    if chart.ndim != 1 or not np.all(np.isfinite(chart)):
        raise ValueError("chart values must be a finite one-dimensional array")
    if pole not in {"north", "south"}:
        raise ValueError("pole must be 'north' or 'south'")
    radius = np.abs(chart) ** 2
    result = np.empty((chart.size, 3), dtype=np.float64)
    result[:, 0] = 2.0 * chart.real / (1.0 + radius)
    result[:, 1] = 2.0 * chart.imag / (1.0 + radius)
    if pole == "south":
        result[:, 1] *= -1.0
    result[:, 2] = (radius - 1.0) / (1.0 + radius)
    if pole == "south":
        result[:, 2] *= -1.0
    return np.ascontiguousarray(result)


def stereographic_transition(values: np.ndarray, *, from_pole: str, to_pole: str) -> np.ndarray:
    """Transition an overlap coordinate between the north and south charts.

    With the orientation-compatible south coordinate used here, the overlap
    transition is the holomorphic inversion ``w = 1/z``.  Keeping this as a
    named primitive makes the atlas convention explicit at call sites.
    """

    chart = _validate_chart(values)
    _validate_poles(from_pole, to_pole)
    if from_pole == to_pole:
        return chart.copy()
    if np.any(np.abs(chart) <= 1e-14):
        raise ValueError("chart values too close to the transition pole")
    return np.ascontiguousarray(1.0 / chart)


def stereographic_transition_velocity(
    values: np.ndarray,
    velocity: np.ndarray,
    *,
    from_pole: str,
    to_pole: str,
) -> np.ndarray:
    """Push a chart velocity through the north/south overlap transition.

    For ``w=1/z`` the chain rule gives ``wdot=-zdot/z**2``.  The operation is
    pointwise and therefore suitable for both analytic flows and autodiff
    cotangent checks; it intentionally rejects points too close to the chart
    pole where the transition is numerically ill-conditioned.
    """

    chart = _validate_chart(values)
    tangent = _validate_chart(velocity)
    if chart.shape != tangent.shape:
        raise ValueError("values and velocity must have the same shape")
    _validate_poles(from_pole, to_pole)
    if from_pole == to_pole:
        return tangent.copy()
    if np.any(np.abs(chart) <= 1e-14):
        raise ValueError("chart values too close to the transition pole")
    return np.ascontiguousarray(-tangent / (chart**2))


def _validate_chart(values: np.ndarray) -> np.ndarray:
    chart = np.asarray(values, dtype=np.complex128)
    if chart.ndim != 1 or not np.all(np.isfinite(chart)):
        raise ValueError("chart values must be a finite one-dimensional array")
    return chart


def _validate_poles(from_pole: str, to_pole: str) -> None:
    if from_pole not in {"north", "south"} or to_pole not in {"north", "south"}:
        raise ValueError("from_pole and to_pole must be 'north' or 'south'")


def _validate_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("points must have shape (n, 3) and be finite")
    if np.max(np.abs(np.linalg.norm(values, axis=1) - 1.0)) > 1e-10:
        raise ValueError("points must lie on the unit sphere")
    return values
