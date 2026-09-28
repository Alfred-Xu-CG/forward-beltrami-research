"""Coordinate and inversion fixtures for saved Q1 real-pair diagnostics."""

from __future__ import annotations

import numpy as np

from tools.digital_q1_real_eval import (
    invert_vertex_map_at_unit_point,
    load_effective_vertices,
    pixel_to_unit,
    unit_to_pixel,
)


def test_pixel_center_normalization_round_trip_and_anisotropic_sizes() -> None:
    sizes = (7470, 9890)
    point = np.array([127.25, 909.75])
    unit = pixel_to_unit(point, sizes)
    np.testing.assert_allclose(unit, (point + .5) / sizes, rtol=0, atol=0)
    np.testing.assert_allclose(unit_to_pixel(unit, sizes), point, rtol=0, atol=1e-12)


def test_inverse_of_bilinear_vertex_map_is_not_a_forward_lookup() -> None:
    axis = np.linspace(0, 1, 3)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((xx, yy), axis=-1)
    vertices[1, 1] += (.07, -.04)
    fixed = np.array([.27, .31])
    s, t = fixed * 2
    corners = vertices[:2, :2]
    moving = ((1 - s) * (1 - t) * corners[0, 0]
              + s * (1 - t) * corners[0, 1]
              + (1 - s) * t * corners[1, 0]
              + s * t * corners[1, 1])
    roots = invert_vertex_map_at_unit_point(vertices, moving)
    assert len(roots) == 1
    np.testing.assert_allclose(roots[0]["fixed_unit_xy"], fixed, atol=1e-12)
    assert roots[0]["residual_unit"] < 1e-12
    assert invert_vertex_map_at_unit_point(vertices, np.array([-0.1, .5])) == []


def test_affine_postmap_archive_is_evaluated_as_composition(tmp_path) -> None:
    axis = np.linspace(0, 1, 3, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    identity = np.stack((xx, yy), axis=-1)[None]
    matrix = np.array([[.94, .14], [-.07, .96]], dtype=np.float64)
    offset = np.array([-.03, .02], dtype=np.float64)
    path = tmp_path / "affine_residual.npz"
    np.savez_compressed(path, vertices=identity, boundary_reference=identity,
                        post_affine_matrix=matrix, post_affine_offset=offset)
    effective, report = load_effective_vertices(path)
    np.testing.assert_allclose(effective, identity[0] @ matrix.T + offset, atol=0, rtol=0)
    assert report["composite_representation_valid"]
    assert report["stored_affine_det_positive_exact"]
