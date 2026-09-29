"""Native-pixel coordinate tiling uses a single certified P1 vertex table."""

import numpy as np
import pytest

from tools.digital_p1_native_tile_query import evaluate_native_tile, stream_roi


def _layout():
    return {
        "side": 32,
        "fixed": {"original_hw": [40, 50],
                  "effective_original_to_canvas_scale_xy": [.4, .5],
                  "padding_xy": [3, 2]},
        "moving": {"original_hw": [80, 100],
                   "effective_original_to_canvas_scale_xy": [.2, .25],
                   "padding_xy": [5, 4]},
    }


def test_native_tile_affine_manual_coordinates_and_different_partition():
    axis = np.linspace(0, 1, 9)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((.9 * xx + .03 * yy + .02,
                         -.02 * xx + 1.03 * yy + .01), axis=-1)
    layout = _layout()
    mapped, inside = evaluate_native_tile(
        vertices, layout["fixed"], layout["moving"], 32, 7, 11, 13, 15,
    )
    y, x = np.meshgrid(np.arange(11, 26), np.arange(7, 20), indexing="ij")
    unit_x = ((x + .5) * .4 + 3) / 32
    unit_y = ((y + .5) * .5 + 2) / 32
    expected_x = ((.9 * unit_x + .03 * unit_y + .02) * 32 - 5) / .2 - .5
    expected_y = ((-.02 * unit_x + 1.03 * unit_y + .01) * 32 - 4) / .25 - .5
    np.testing.assert_allclose(mapped[..., 0], expected_x, rtol=0, atol=2e-14)
    np.testing.assert_allclose(mapped[..., 1], expected_y, rtol=0, atol=2e-14)
    assert np.array_equal(inside, (mapped[..., 0] >= 0) & (mapped[..., 0] < 100)
                          & (mapped[..., 1] >= 0) & (mapped[..., 1] < 80))
    first = stream_roi(vertices, layout, roi_xywh=(7, 11, 13, 15), tile_side=4)
    second = stream_roi(vertices, layout, roi_xywh=(7, 11, 13, 15), tile_side=9)
    assert first["query_pixels"] == second["query_pixels"] == 195
    assert first["direct_tile_corner_checks_bitwise_equal"]
    assert second["direct_tile_corner_checks_bitwise_equal"]
    assert first["moving_original_inside_fraction"] == second["moving_original_inside_fraction"]
    np.testing.assert_allclose(first["moving_original_coordinate_min_xy"],
                               second["moving_original_coordinate_min_xy"], atol=0, rtol=0)
    np.testing.assert_allclose(first["moving_original_coordinate_max_xy"],
                               second["moving_original_coordinate_max_xy"], atol=0, rtol=0)


def test_native_tile_rejects_out_of_declared_original_slide():
    axis = np.linspace(0, 1, 3)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((xx, yy), axis=-1)
    with pytest.raises(ValueError, match="inside original fixed slide"):
        stream_roi(vertices, _layout(), roi_xywh=(49, 0, 2, 1), tile_side=4)


def test_nonaffine_p1_diagonal_and_every_pixel_partition_invariance():
    axis = np.linspace(0, 1, 9)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((xx + .03 * np.sin(np.pi * xx) * np.sin(np.pi * yy), yy), axis=-1)
    layout = _layout()
    entire, _ = evaluate_native_tile(
        vertices, layout["fixed"], layout["moving"], 32, 7, 11, 13, 15,
    )
    for tile_side in (1, 4, 9):
        assembled = np.empty_like(entire)
        for i in range(0, 15, tile_side):
            for j in range(0, 13, tile_side):
                th, tw = min(tile_side, 15 - i), min(tile_side, 13 - j)
                tile, _ = evaluate_native_tile(
                    vertices, layout["fixed"], layout["moving"], 32,
                    7 + j, 11 + i, tw, th,
                )
                assembled[i:i + th, j:j + tw] = tile
        assert np.array_equal(assembled, entire)

    # One query strictly above the diagonal: reconstruct the two P1
    # barycentric weights independently, with no call to p1_at_queries.
    p = np.array([7., 11.])
    q = ((p + .5) * np.array([.4, .5]) + np.array([3., 2.])) / 32
    scaled = q * 8
    cx, cy = np.floor(scaled).astype(int)
    s, t = scaled - np.array([cx, cy])
    a, b = vertices[cy, cx], vertices[cy, cx + 1]
    c, d = vertices[cy + 1, cx + 1], vertices[cy + 1, cx]
    if t <= s:
        expected_unit = (1 - s) * a + (s - t) * b + t * c
    else:
        expected_unit = (1 - t) * a + s * c + (t - s) * d
    expected = (expected_unit * 32 - np.array([5., 4.])) / np.array([.2, .25]) - .5
    np.testing.assert_allclose(entire[0, 0], expected, atol=2e-14, rtol=0)
    q1_unit = ((1 - s) * (1 - t) * a + s * (1 - t) * b
               + s * t * c + (1 - s) * t * d)
    assert np.linalg.norm(q1_unit - expected_unit) > 1e-6
