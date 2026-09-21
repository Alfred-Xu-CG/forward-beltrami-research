"""Precisely defined common metrics for Phase V P1 deformation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..beltrami import face_beltrami, face_jacobians
from ..injectivity import audit_injectivity
from ..mesh import TriMesh


@dataclass(frozen=True)
class P1MapMetrics:
    """Metrics evaluated on the represented vertex coordinates.

    ``minimum_signed_area`` is the smallest oriented target triangle area.
    ``minimum_area_ratio`` is the smallest determinant of the exact facewise
    P1 Jacobian, i.e. target/source oriented-area ratio.  The boundary gap is
    the smallest Euclidean length of an oriented boundary edge; it detects
    representationally repeated consecutive boundary vertices but is not by
    itself an order or global-injectivity proof.

    Map RMSE is ``sqrt(mean_i ||f_i-g_i||_2^2)`` and maximum map error is the
    maximum vertexwise Euclidean norm.  Beltrami errors use complex modulus per
    face.  Accuracy fields are NaN when no target is supplied.
    """

    flip_count: int
    minimum_signed_area: float
    minimum_area_ratio: float
    boundary_order_min_gap: float
    global_injectivity_certificate: bool
    map_rmse: float
    maximum_map_error: float
    mu_rmse: float
    maximum_mu_error: float


def compute_p1_map_metrics(
    mesh: TriMesh,
    mapped: np.ndarray,
    *,
    target: np.ndarray | None = None,
) -> P1MapMetrics:
    mapped = np.asarray(mapped, dtype=np.float64)
    if mapped.shape != (mesh.n_vertices, 2) or not np.all(np.isfinite(mapped)):
        raise ValueError("mapped must be a finite (mesh.n_vertices, 2) array")
    determinants = np.linalg.det(face_jacobians(mesh, mapped))
    signed_areas = mesh.areas * determinants
    loop = mesh.boundary_loops[0]
    boundary_edges = np.roll(mapped[loop], -1, axis=0) - mapped[loop]
    minimum_gap = float(np.min(np.linalg.norm(boundary_edges, axis=1)))
    report = audit_injectivity(mesh, mapped)

    map_rmse = maximum_map_error = mu_rmse = maximum_mu_error = float("nan")
    if target is not None:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != mapped.shape or not np.all(np.isfinite(target)):
            raise ValueError("target must be a finite array with the same shape as mapped")
        error_norm = np.linalg.norm(mapped - target, axis=1)
        map_rmse = float(np.sqrt(np.mean(error_norm**2)))
        maximum_map_error = float(np.max(error_norm))
        mu_error = np.abs(face_beltrami(mesh, mapped) - face_beltrami(mesh, target))
        mu_rmse = float(np.sqrt(np.mean(mu_error**2)))
        maximum_mu_error = float(np.max(mu_error))

    return P1MapMetrics(
        flip_count=int(np.count_nonzero(determinants <= 0.0)),
        minimum_signed_area=float(np.min(signed_areas)),
        minimum_area_ratio=float(np.min(determinants)),
        boundary_order_min_gap=minimum_gap,
        global_injectivity_certificate=report.certified,
        map_rmse=map_rmse,
        maximum_map_error=maximum_map_error,
        mu_rmse=mu_rmse,
        maximum_mu_error=maximum_mu_error,
    )
