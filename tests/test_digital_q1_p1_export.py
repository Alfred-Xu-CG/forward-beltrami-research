"""The stored safe vertex table has a separate fixed-diagonal P1 meaning."""

from __future__ import annotations

import numpy as np

from tools.digital_q1_p1_eval import (
    invert_p1_at_unit_point, p1_map_at_unit_queries, prepare_p1,
)


def test_p1_export_reproduces_affine_at_offgrid_queries_and_inverse() -> None:
    axis = np.linspace(0, 1, 5)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    base = np.stack((xx, yy), -1)
    matrix = np.array([[1.04, .09], [-.03, .98]])
    offset = np.array([.02, -.01])
    def scalar_affine(points: np.ndarray) -> np.ndarray:
        return np.stack((
            matrix[0, 0] * points[..., 0] + matrix[0, 1] * points[..., 1] + offset[0],
            matrix[1, 0] * points[..., 0] + matrix[1, 1] * points[..., 1] + offset[1],
        ), axis=-1)

    # Avoid a NumPy BLAS startup abort in the Windows PyTorch test process.
    vertices = scalar_affine(base)
    queries = np.array([[.11, .33], [.47, .47], [.76, .63], [.99, .02]])
    values = p1_map_at_unit_queries(vertices, queries)
    np.testing.assert_allclose(values, scalar_affine(queries), atol=1e-14)
    geometry = prepare_p1(vertices)
    for source, target in zip(queries, values):
        roots = invert_p1_at_unit_point(geometry, target)
        assert len(roots) == 1
        np.testing.assert_allclose(roots[0], source, atol=1e-12)


def test_p1_export_is_continuous_on_diagonal_of_nonaffine_quad() -> None:
    vertices = np.array([[[0., 0.], [1., .1]],
                         [[-.1, 1.], [1.1, 1.2]]])
    diagonal = np.array([[.25, .25], [.5, .5], [.75, .75]])
    expected = vertices[0, 0] + diagonal[:, :1] * (
        vertices[1, 1] - vertices[0, 0])
    np.testing.assert_allclose(p1_map_at_unit_queries(vertices, diagonal),
                               expected, atol=1e-14)
