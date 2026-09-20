"""Non-periodic Barnes--Hut treecode for the whole-plane Beurling kernel.

This is a transparent scattered-grid control, not a replacement for a
production FMM.  A quadtree stores complex source moments and evaluates
``-1/(pi*(z-w)^2)`` by an order-``p`` Taylor expansion on admissible far
clusters, while leaf clusters are summed directly.  No periodic wrapping or
uniform source grid is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Node:
    center: complex
    radius: float
    moments: np.ndarray
    indices: np.ndarray | None
    children: tuple[int, ...]


def treecode_beurling_apply(
    points: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    target_points: np.ndarray | None = None,
    theta: float = 0.5,
    order: int = 6,
    max_leaf: int = 32,
    max_depth: int = 32,
    near_radius: float | None = None,
) -> np.ndarray:
    """Apply the whole-plane Beurling kernel on scattered source/target sets.

    ``theta`` is the Barnes--Hut opening angle and ``order`` is the number of
    Taylor moments retained.  If ``target_points`` is omitted, coincident
    source terms are omitted at leaf evaluation, matching the direct
    principal-value diagnostic.  With disjoint targets the result is an
    ordinary quadrature approximation.
    """

    source = np.asarray(points, dtype=np.complex128)
    targets = source if target_points is None else np.asarray(target_points, dtype=np.complex128)
    density = np.asarray(values, dtype=np.complex128)
    area = np.asarray(weights, dtype=np.float64)
    if source.ndim != 1 or targets.ndim != 1 or density.shape != source.shape or area.shape != source.shape:
        raise ValueError("points, values, and weights must have matching 1D shapes")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(targets)) or not np.all(np.isfinite(density)) or not np.all(np.isfinite(area)):
        raise ValueError("inputs must be finite")
    if np.any(area <= 0.0):
        raise ValueError("weights must be positive")
    if not (0.0 < theta < 1.0) or order < 1 or max_leaf < 1 or max_depth < 1:
        raise ValueError("theta must lie in (0,1), order/max_leaf/max_depth must be positive")
    if near_radius is not None and (not np.isfinite(near_radius) or near_radius <= 0.0):
        raise ValueError("near_radius must be positive when supplied")
    if source.size == 0:
        return np.zeros_like(targets)
    nodes: list[_Node] = []
    charge = density * area

    def build(indices: np.ndarray, depth: int) -> int:
        xs, ys = source[indices].real, source[indices].imag
        center = complex(0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()))
        radius = float(0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()))
        radius = max(radius, np.finfo(float).eps)
        delta = source[indices] - center
        moments = np.asarray([np.sum(charge[indices] * delta**k) for k in range(order)], dtype=np.complex128)
        if len(indices) <= max_leaf or depth >= max_depth:
            nodes.append(_Node(center, radius, moments, np.asarray(indices, dtype=np.int64), ()))
            return len(nodes) - 1
        half = radius
        quadrant = (source[indices].real >= center.real).astype(np.int64) + 2 * (source[indices].imag >= center.imag).astype(np.int64)
        child_ids: list[int] = []
        for q in range(4):
            child_indices = indices[quadrant == q]
            if child_indices.size:
                child_ids.append(build(child_indices, depth + 1))
        if len(child_ids) <= 1:
            nodes.append(_Node(center, radius, moments, np.asarray(indices, dtype=np.int64), ()))
        else:
            nodes.append(_Node(center, half, moments, None, tuple(child_ids)))
        return len(nodes) - 1

    root = build(np.arange(source.size, dtype=np.int64), 0)
    result = np.zeros(targets.shape, dtype=np.complex128)
    same_source = target_points is None
    for target_index, target in enumerate(targets):
        total = 0.0j
        stack = [root]
        while stack:
            node = nodes[stack.pop()]
            delta = target - node.center
            distance = abs(delta)
            if node.indices is not None:
                offsets = target - source[node.indices]
                kernel = np.zeros(offsets.shape, dtype=np.complex128)
                valid = np.abs(offsets) > 0.0
                kernel[valid] = -1.0 / (np.pi * offsets[valid] ** 2)
                if same_source:
                    # The source/target arrays coincide by position in this
                    # mode; omit the matching principal-value self term.
                    matching = node.indices == target_index
                    kernel[matching] = 0.0
                total += np.sum(kernel * charge[node.indices])
            elif (
                (near_radius is None or distance - node.radius >= near_radius)
                and node.radius / max(distance, np.finfo(float).eps) < theta
            ):
                expansion = 0.0j
                for k, moment in enumerate(node.moments):
                    expansion += (k + 1.0) * moment / (delta ** (k + 2))
                total += -expansion / np.pi
            else:
                stack.extend(node.children)
        result[target_index] = total
    return np.ascontiguousarray(result)


def treecode_beurling_values_vjp(
    points: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    grad_output: np.ndarray,
    *,
    target_points: np.ndarray | None = None,
    theta: float = 0.5,
    order: int = 6,
    max_leaf: int = 32,
    max_depth: int = 32,
) -> np.ndarray:
    """Return the real-inner-product VJP with respect to complex source values.

    The VJP replays the same fixed Barnes--Hut admissibility decisions and
    accumulates the conjugate coefficient of each accepted leaf or multipole
    term.  It is therefore the exact transpose of this treecode approximation
    (up to floating-point order), rather than an unrelated swapped-tree
    approximation.
    """

    source = np.asarray(points, dtype=np.complex128)
    targets = source if target_points is None else np.asarray(target_points, dtype=np.complex128)
    area = np.asarray(weights, dtype=np.float64)
    cotangent = np.asarray(grad_output, dtype=np.complex128)
    if cotangent.shape != targets.shape or not np.all(np.isfinite(cotangent)):
        raise ValueError("grad_output must match target_points and be finite")
    if source.ndim != 1 or targets.ndim != 1 or area.shape != source.shape or not np.all(np.isfinite(source)) or not np.all(np.isfinite(targets)) or np.any(area <= 0.0):
        raise ValueError("points, target_points, and weights have invalid values")
    nodes: list[_Node] = []
    supports: dict[int, np.ndarray] = {}

    def build(indices: np.ndarray, depth: int) -> int:
        xs, ys = source[indices].real, source[indices].imag
        center = complex(0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()))
        radius = max(float(0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())), np.finfo(float).eps)
        zero_moments = np.zeros(order, dtype=np.complex128)
        if len(indices) <= max_leaf or depth >= max_depth:
            nodes.append(_Node(center, radius, zero_moments, np.asarray(indices, dtype=np.int64), ()))
            return len(nodes) - 1
        quadrant = (source[indices].real >= center.real).astype(np.int64) + 2 * (source[indices].imag >= center.imag).astype(np.int64)
        child_ids = [build(indices[quadrant == q], depth + 1) for q in range(4) if np.any(quadrant == q)]
        if len(child_ids) <= 1:
            nodes.append(_Node(center, radius, zero_moments, np.asarray(indices, dtype=np.int64), ()))
        else:
            nodes.append(_Node(center, radius, zero_moments, None, tuple(child_ids)))
            supports[len(nodes) - 1] = np.asarray(indices, dtype=np.int64)
        return len(nodes) - 1

    root = build(np.arange(source.size, dtype=np.int64), 0)
    gradient = np.zeros(source.shape, dtype=np.complex128)
    # Accumulate the adjoint of each accepted multipole moment while replaying
    # target traversals.  Expanding this once after the replay avoids the
    # previous O(targets * accepted-clusters * cluster-size) source loop.
    moment_adjoint = np.zeros((len(nodes), order), dtype=np.complex128)
    same_source = target_points is None
    for target_index, target in enumerate(targets):
        cotangent_value = cotangent[target_index]
        stack = [root]
        while stack:
            node_id = stack.pop()
            node = nodes[node_id]
            delta = target - node.center
            distance = abs(delta)
            if node.indices is not None:
                indices = node.indices
                offsets = target - source[indices]
                valid = np.abs(offsets) > 0.0
                if same_source:
                    valid &= indices != target_index
                contribution = np.zeros(indices.shape, dtype=np.complex128)
                contribution[valid] = np.conjugate(-1.0 / (np.pi * offsets[valid] ** 2)) * cotangent_value
                gradient[indices] += area[indices] * contribution
            elif node.radius / max(distance, np.finfo(float).eps) < theta:
                indices = supports[node_id]
                for k in range(order):
                    moment_adjoint[node_id, k] += (
                        np.conjugate(-((k + 1.0) / np.pi) / (delta ** (k + 2))) * cotangent_value
                    )
            else:
                stack.extend(node.children)
    # A node's moments are linear in the source values.  Replay that linear
    # map once per accepted node to obtain the exact transpose of the same
    # fixed-tree multipole approximation.
    for node_id, indices in supports.items():
        coeff = moment_adjoint[node_id]
        if not np.any(coeff):
            continue
        source_delta = np.conjugate(source[indices] - nodes[node_id].center)
        polynomial = np.zeros(indices.shape, dtype=np.complex128)
        for k in range(order - 1, -1, -1):
            polynomial = polynomial * source_delta + coeff[k]
        gradient[indices] += area[indices] * polynomial
    return np.ascontiguousarray(gradient)
