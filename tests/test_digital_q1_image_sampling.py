"""Q1 map queries and image pixels use two explicitly different grids."""

from __future__ import annotations

import torch

from qcopt.neural_bijection.dense.q1_image_sampling import (
    fixed_pixel_centers,
    q1_map_at_pixel_centers,
    warp_moving_at_q1_map,
)


def _identity(rows: int, columns: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    y = torch.arange(rows, dtype=dtype) / (rows - 1)
    x = torch.arange(columns, dtype=dtype) / (columns - 1)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None]


def test_center_queries_are_not_mesh_vertices_or_image_endpoints() -> None:
    centers = fixed_pixel_centers(3, 5, dtype=torch.float64, device=torch.device("cpu"))
    torch.testing.assert_close(centers[0, 0, 0], torch.tensor([.1, 1 / 6], dtype=torch.float64))
    torch.testing.assert_close(centers[0, -1, -1], torch.tensor([.9, 5 / 6], dtype=torch.float64))
    vertex_map = _identity(5, 5)
    sampled = q1_map_at_pixel_centers(vertex_map, 3, 5)
    torch.testing.assert_close(sampled, centers, atol=1e-15, rtol=0)


def test_q1_query_is_exact_bilinear_interpolation_and_has_vertex_gradient() -> None:
    vertices = _identity(2, 2).requires_grad_()
    vertices = vertices + torch.tensor([[[[0., 0.], [.1, 0.]],
                                         [[0., -.1], [.2, .1]]]], dtype=torch.float64)
    queried = q1_map_at_pixel_centers(vertices, 2, 2)
    q = fixed_pixel_centers(2, 2, dtype=torch.float64, device=torch.device("cpu"))
    s, t = q[0, 0, 0]
    expected = ((1 - s) * (1 - t) * vertices[0, 0, 0]
                + s * (1 - t) * vertices[0, 0, 1]
                + s * t * vertices[0, 1, 1]
                + (1 - s) * t * vertices[0, 1, 0])
    torch.testing.assert_close(queried[0, 0, 0], expected, atol=1e-15, rtol=0)
    (grad,) = torch.autograd.grad(queried[..., 0].sum(), vertices)
    assert torch.isfinite(grad).all() and grad.abs().amax() > 0


def test_identity_q1_warp_preserves_a_pixel_center_image() -> None:
    moving = torch.arange(3 * 5, dtype=torch.float64).reshape(1, 1, 3, 5) / 15
    result = warp_moving_at_q1_map(moving, _identity(5, 5), height=3, width=5)
    torch.testing.assert_close(result, moving, atol=1e-15, rtol=0)
