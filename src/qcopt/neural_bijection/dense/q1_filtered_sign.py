"""Filtered exact signs of Q1 corners on *saved binary* map coordinates.

Each vectorized binary64 interval operation expands one representable value
in both directions. Under IEEE correctly rounded NumPy float64 arithmetic,
the interval encloses the exact result of that operation on the input binary
coordinates. Ambiguous corners use rational arithmetic on the exact binary
values. No approximate positive value is silently accepted as a certificate.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np


def _outward_difference(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rounded = x - y
    return np.nextafter(rounded, -np.inf), np.nextafter(rounded, np.inf)


def _outward_product(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    a, b = first
    c, d = second
    candidates = (a * c, a * d, b * c, b * d)
    lower = np.minimum.reduce(candidates)
    upper = np.maximum.reduce(candidates)
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def _orientation_interval(
    p: np.ndarray, q: np.ndarray, r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ex = _outward_difference(q[..., 0], p[..., 0])
    ey = _outward_difference(q[..., 1], p[..., 1])
    fx = _outward_difference(r[..., 0], p[..., 0])
    fy = _outward_difference(r[..., 1], p[..., 1])
    product1 = _outward_product(ex, fy)
    product2 = _outward_product(ey, fx)
    lower = np.nextafter(product1[0] - product2[1], -np.inf)
    upper = np.nextafter(product1[1] - product2[0], np.inf)
    return lower, upper


def _exact_orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> int:
    def exact(value: np.float64) -> Fraction:
        return Fraction.from_float(float(value))

    px, py = exact(p[0]), exact(p[1])
    qx, qy = exact(q[0]), exact(q[1])
    rx, ry = exact(r[0]), exact(r[1])
    determinant = (qx - px) * (ry - py) - (qy - py) * (rx - px)
    return (determinant > 0) - (determinant < 0)


def _ordered_axis_aligned_rectangle(reference: np.ndarray) -> bool:
    """Verify a simple, counterclockwise rectangle boundary in stored values."""
    x0 = reference[:, 0, 0, 0]
    x1 = reference[:, 0, -1, 0]
    y0 = reference[:, 0, 0, 1]
    y1 = reference[:, -1, 0, 1]
    return bool(
        np.all(x0 < x1) and np.all(y0 < y1)
        and np.all(reference[:, 0, :, 1] == y0[:, None])
        and np.all(reference[:, -1, :, 1] == y1[:, None])
        and np.all(reference[:, :, 0, 0] == x0[:, None])
        and np.all(reference[:, :, -1, 0] == x1[:, None])
        and np.all(reference[:, 0, 1:, 0] > reference[:, 0, :-1, 0])
        and np.all(reference[:, -1, 1:, 0] > reference[:, -1, :-1, 0])
        and np.all(reference[:, 1:, 0, 1] > reference[:, :-1, 0, 1])
        and np.all(reference[:, 1:, -1, 1] > reference[:, :-1, -1, 1])
    )


def certify_q1_binary_map(
    path: str | Path, *, chunk_rows: int = 64,
) -> dict[str, bool | int | str]:
    """Check a saved ``.npz`` Q1 map with an exact-sign fallback.

    The file must contain ``vertices`` and ``boundary_reference`` arrays of
    identical float32/float64 shape ``(batch,rows,columns,2)``. The stored
    reference boundary must itself be a simple, ordered, axis-aligned
    rectangle; this intentionally narrow condition makes ``valid`` a global
    Q1-homeomorphism certificate rather than merely a local-sign check.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    with np.load(path, allow_pickle=False) as archive:
        vertices = archive["vertices"]
        reference = archive["boundary_reference"]
    if (
        vertices.shape != reference.shape or vertices.ndim != 4
        or vertices.shape[0] < 1
        or vertices.shape[-1] != 2 or min(vertices.shape[1:3]) < 2
        or vertices.dtype != reference.dtype
        or vertices.dtype not in (np.dtype("float32"), np.dtype("float64"))
    ):
        raise ValueError("matching float32/float64 (nonempty batch,R>=2,C>=2,2) arrays required")
    boundary_exact = all(np.array_equal(actual, expected) for actual, expected in (
        (vertices[:, 0], reference[:, 0]),
        (vertices[:, -1], reference[:, -1]),
        (vertices[:, :, 0], reference[:, :, 0]),
        (vertices[:, :, -1], reference[:, :, -1]),
    ))
    nonfinite = int(np.count_nonzero(~np.isfinite(vertices)))
    nonfinite += int(np.count_nonzero(~np.isfinite(reference)))
    total = vertices.shape[0] * (vertices.shape[1] - 1) * (vertices.shape[2] - 1) * 4
    if nonfinite:
        return {
            "valid": False, "corners": total, "positive_corners": 0,
            "nonpositive_corners": 0, "exact_fallback_corners": 0,
            "nonfinite_coordinates": nonfinite, "boundary_exact": boundary_exact,
            "boundary_ordered_rectangle": False,
            "dtype": str(vertices.dtype),
        }

    boundary_ordered_rectangle = _ordered_axis_aligned_rectangle(reference)

    positive = 0
    nonpositive = 0
    fallback_count = 0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for start in range(0, vertices.shape[1] - 1, chunk_rows):
            end = min(start + chunk_rows, vertices.shape[1] - 1)
            block = vertices[:, start:end + 1].astype(np.float64)
            a, b = block[:, :-1, :-1], block[:, :-1, 1:]
            c, d = block[:, 1:, 1:], block[:, 1:, :-1]
            for p, q, r in ((a, b, d), (a, b, c), (d, b, c), (a, c, d)):
                lower, upper = _orientation_interval(p, q, r)
                definitely_positive = lower > 0
                definitely_negative = upper < 0
                positive += int(np.count_nonzero(definitely_positive))
                nonpositive += int(np.count_nonzero(definitely_negative))
                uncertain = ~(definitely_positive | definitely_negative)
                if np.any(uncertain):
                    indices = np.argwhere(uncertain)
                    fallback_count += len(indices)
                    for index in indices:
                        key = tuple(index)
                        sign = _exact_orientation(p[key], q[key], r[key])
                        if sign > 0:
                            positive += 1
                        else:
                            nonpositive += 1
    if positive + nonpositive != total:
        raise AssertionError("every Q1 corner must be classified")
    return {
        "valid": bool(boundary_exact and boundary_ordered_rectangle and nonpositive == 0),
        "corners": total,
        "positive_corners": positive,
        "nonpositive_corners": nonpositive,
        "exact_fallback_corners": fallback_count,
        "nonfinite_coordinates": 0,
        "boundary_exact": boundary_exact,
        "boundary_ordered_rectangle": boundary_ordered_rectangle,
        "dtype": str(vertices.dtype),
    }
