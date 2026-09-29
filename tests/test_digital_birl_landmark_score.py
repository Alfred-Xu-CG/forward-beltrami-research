import numpy as np

from tools.digital_birl_landmark_score import (
    canvas_unit_to_original_pixel, original_pixel_to_canvas_unit,
    p1_at_queries, q1_at_queries,
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
    np.testing.assert_allclose(p1_at_queries(identity, queries), queries, atol=1e-15)


def test_fixed_diagonal_p1_and_q1_difference_formula() -> None:
    corners = np.array([
        [[.1, -.2], [1.1, .3]],
        [[-.1, .9], [1.4, 1.2]],
    ])
    query = np.array([[.8, .2], [.2, .8], [.5, .5], [1., 1.]])
    p1 = p1_at_queries(corners, query)
    q1 = q1_at_queries(corners, query)
    a, b, c, d = corners[0, 0], corners[0, 1], corners[1, 1], corners[1, 0]
    bend = a - b + c - d
    for index, (s, t) in enumerate(query):
        expected = ((1-s)*a + (s-t)*b + t*c if t <= s else
                    (1-t)*a + s*c + (t-s)*d)
        np.testing.assert_allclose(p1[index], expected, atol=1e-15)
        coefficient = t*(1-s) if t <= s else s*(1-t)
        np.testing.assert_allclose(q1[index] - p1[index], -coefficient*bend,
                                   atol=1e-15)
