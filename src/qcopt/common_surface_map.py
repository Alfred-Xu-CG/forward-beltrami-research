"""Sampling a certified overlay map onto the original source/target meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface_atlas import SurfacePoint
from .surface_locator import SurfaceLocator
from .surface_mesh import CommonRefinement, SurfaceMesh


@dataclass(frozen=True)
class SampledCommonMap:
    source_points: tuple[SurfacePoint, ...]
    target_points: tuple[SurfacePoint, ...]
    overlay_faces: np.ndarray
    overlay_barycentric: np.ndarray
    maximum_source_projection_error: float
    maximum_target_projection_error: float


def sample_common_refinement(
    common: CommonRefinement,
    source_original: SurfaceMesh,
    target_original: SurfaceMesh,
    *,
    maximum_samples: int,
) -> SampledCommonMap:
    """Locate common-overlay face centroids on the original surface meshes."""

    if maximum_samples < 1:
        raise ValueError("maximum_samples must be positive")
    genus = common.source.topology.genus
    if not (
        common.source.topology.closed
        and common.target.topology.closed
        and source_original.topology.closed
        and target_original.topology.closed
        and genus is not None
        and common.target.topology.genus == genus
        and source_original.topology.genus == genus
        and target_original.topology.genus == genus
    ):
        raise ValueError("common refinement and original surfaces must share closed topology/genus")
    if not common.source.has_same_connectivity(common.target):
        raise ValueError("overlay pair does not define common connectivity")
    count = min(maximum_samples, common.source.n_faces)
    faces = np.unique(
        np.linspace(0, common.source.n_faces - 1, count, dtype=np.int64)
    )
    barycentric = np.tile(np.full(3, 1.0 / 3.0), (len(faces), 1))
    return locate_overlay_samples(
        common,
        source_original,
        target_original,
        faces,
        barycentric,
    )


def locate_overlay_samples(
    common: CommonRefinement,
    source_original: SurfaceMesh,
    target_original: SurfaceMesh,
    overlay_faces: np.ndarray,
    overlay_barycentric: np.ndarray,
    *,
    source_locator: SurfaceLocator | None = None,
    target_locator: SurfaceLocator | None = None,
) -> SampledCommonMap:
    overlay_faces = np.asarray(overlay_faces, dtype=np.int64)
    overlay_barycentric = np.asarray(overlay_barycentric, dtype=np.float64)
    if overlay_faces.ndim != 1 or overlay_barycentric.shape != (len(overlay_faces), 3):
        raise ValueError("overlay faces/barycentric sample shapes do not agree")
    if len(overlay_faces) and (
        overlay_faces.min() < 0 or overlay_faces.max() >= common.source.n_faces
    ):
        raise ValueError("overlay face index out of range")
    source_xyz = np.vstack(
        [
            common.source.point_from_barycentric(int(face), barycentric)
            for face, barycentric in zip(overlay_faces, overlay_barycentric)
        ]
    )
    target_xyz = np.vstack(
        [
            common.target.point_from_barycentric(int(face), barycentric)
            for face, barycentric in zip(overlay_faces, overlay_barycentric)
        ]
    )
    source_locator = source_locator or SurfaceLocator(source_original)
    target_locator = target_locator or SurfaceLocator(target_original)
    source_batch = source_locator.locate_many(source_xyz)
    target_batch = target_locator.locate_many(target_xyz)
    faces_copy = overlay_faces.copy()
    barycentric_copy = overlay_barycentric.copy()
    faces_copy.setflags(write=False)
    barycentric_copy.setflags(write=False)
    return SampledCommonMap(
        source_batch.points,
        target_batch.points,
        faces_copy,
        barycentric_copy,
        source_batch.maximum_distance,
        target_batch.maximum_distance,
    )
