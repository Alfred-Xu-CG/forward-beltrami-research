import numpy as np

from tools.digital_birl_landmark_score import (
    canvas_unit_to_original_pixel, original_pixel_to_canvas_unit, q1_at_queries,
)


def test_layout_round_trip_and_identity_map() -> None:
    layout = {"effective_original_to_canvas_scale_xy": [.43, .57],
              "padding_xy": [12, 38]}
    points = np.array([[0, 0], [123.4, 56.7], [890, 502]], dtype=np.float64)
    mapped = original_pixel_to_canvas_unit(points, layout, 512)
    np.testing.assert_allclose(canvas_unit_to_original_pixel(mapped, layout, 512),
                               points, atol=1e-12)
    axis = np.arange(5, dtype=np.float64) / 4
    y, x = np.meshgrid(axis, axis, indexing="ij")
    identity = np.stack((x, y), axis=-1)
    queries = np.array([[.125, .25], [.33, .73], [1., 1.]])
    np.testing.assert_allclose(q1_at_queries(identity, queries), queries, atol=1e-15)
