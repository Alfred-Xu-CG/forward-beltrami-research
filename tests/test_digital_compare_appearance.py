"""DHR saved-field to common image-query coordinate fixtures."""

from __future__ import annotations

import numpy as np
import torch

from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers
from tools.digital_compare_appearance import (
    _appearance,
    dhr_map_at_fixed_pixel_centers,
    dhr_map_at_unit_queries,
    q1_query_from_saved_map,
)


def test_zero_field_with_same_sizes_is_unit_identity() -> None:
    field = np.zeros((2, 10, 10), dtype=np.float32)
    params = {"source_resample_ratio": .1, "target_resample_ratio": .1,
              "pad_1": [[0, 0], [0, 0]], "pad_2": [[0, 0], [0, 0]]}
    actual = dhr_map_at_fixed_pixel_centers(
        field, params, fixed_size=(100, 100), moving_size=(100, 100), image_side=8,
    )
    expected = fixed_pixel_centers(8, 8, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=0)


def test_constant_saved_field_is_scaled_physical_translation() -> None:
    field = np.zeros((2, 20, 20), dtype=np.float32)
    field[0] = .2  # Two supplied-JPEG pixels at scale .1.
    params = {"source_resample_ratio": .1, "target_resample_ratio": .1,
              "pad_1": [[0, 0], [0, 0]], "pad_2": [[0, 0], [0, 0]]}
    actual = dhr_map_at_fixed_pixel_centers(
        field, params, fixed_size=(100, 100), moving_size=(100, 100), image_side=8,
    )
    expected = fixed_pixel_centers(8, 8, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(actual[..., 0] - expected[..., 0],
                               torch.full_like(expected[..., 0], .02), atol=2e-7, rtol=0)
    torch.testing.assert_close(actual[..., 1], expected[..., 1], atol=2e-7, rtol=0)


def test_identical_image_has_zero_multiscale_correlation_loss() -> None:
    torch.manual_seed(3)
    image = torch.rand((1, 1, 32, 32))
    query = fixed_pixel_centers(32, 32, dtype=torch.float32, device=torch.device("cpu"))
    report = _appearance(image, image, query)
    assert abs(report["one_minus_ncc_area_8"]) < 2e-6
    assert abs(report["one_minus_ncc_area_4"]) < 2e-6


def test_dhr_field_can_be_queried_at_control_vertices() -> None:
    field = np.zeros((2, 10, 10), dtype=np.float32)
    field[1] = -.1
    params = {"source_resample_ratio": .1, "target_resample_ratio": .1,
              "pad_1": [[0, 0], [0, 0]], "pad_2": [[0, 0], [0, 0]]}
    axis = torch.linspace(.2, .8, 4)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    query = torch.stack((x, y), dim=-1)[None]
    mapped = dhr_map_at_unit_queries(
        field, params, fixed_size=(100, 100), moving_size=(100, 100), query=query,
    )
    torch.testing.assert_close(mapped[..., 0], query[..., 0], atol=2e-7, rtol=0)
    torch.testing.assert_close(mapped[..., 1] - query[..., 1],
                               torch.full_like(query[..., 1], -.01), atol=2e-7, rtol=0)


def test_saved_affine_postmap_is_applied_after_q1_interpolation(tmp_path) -> None:
    axis = np.linspace(0, 1, 3, dtype=np.float32)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    identity = np.stack((x, y), axis=-1)[None]
    matrix = np.array([[.94, .14], [-.07, .96]], dtype=np.float32)
    offset = np.array([-.03, .02], dtype=np.float32)
    path = tmp_path / "affine_q1.npz"
    np.savez_compressed(path, vertices=identity, boundary_reference=identity,
                        post_affine_matrix=matrix, post_affine_offset=offset)
    actual = q1_query_from_saved_map(path, image_side=8)
    query = fixed_pixel_centers(8, 8, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(actual, query @ torch.from_numpy(matrix.T) + torch.from_numpy(offset),
                               atol=2e-7, rtol=0)
