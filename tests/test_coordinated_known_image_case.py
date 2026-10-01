import numpy as np
import pytest
import torch

from tools.coordinated_geometry_pilot import reference_grid, target_map
from tools.coordinated_known_image_case import (
    generate_fixed, generic_corner_ratio, held_out_map_metrics, sample_q1_queries,
)


def test_identity_pixel_centers_preserve_rectangular_texture():
    moving = torch.arange(99, dtype=torch.float64).reshape(1, 1, 9, 11)/100
    axis_y, axis_x = torch.linspace(0, 1, 5, dtype=torch.float64), torch.linspace(0, 1, 7, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
    vertices = torch.stack((xx, yy), -1)[None].double()
    assert torch.allclose(generate_fixed(moving, vertices), moving, atol=1e-14)


def test_independent_query_interpolator_recovers_affine_non_square_map():
    y, x = torch.meshgrid(torch.linspace(0, 1, 5, dtype=torch.float64),
                          torch.linspace(0, 1, 7, dtype=torch.float64), indexing="ij")
    vertices = torch.stack((2*x+3*y+.1, -x+4*y-.2), -1)[None]
    points = torch.tensor([[.13, .27], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    expected = torch.stack((2*points[:, 0]+3*points[:, 1]+.1,
                            -points[:, 0]+4*points[:, 1]-.2), -1)
    assert torch.allclose(sample_q1_queries(vertices, points), expected, atol=1e-14)


@pytest.mark.parametrize("name", ["wide_shear", "local_rotation", "coarse_fine"])
def test_actual_257_targets_have_all_corners_and_exact_boundary(name):
    reference = reference_grid(257)
    target = target_map(reference, name)
    assert generic_corner_ratio(target, reference) > .001
    for edge in ((slice(None), 0), (slice(None), -1)):
        assert torch.equal(target[edge], reference[edge])
    assert torch.equal(target[:, :, 0], reference[:, :, 0])
    assert torch.equal(target[:, :, -1], reference[:, :, -1])


def test_held_out_queries_are_evaluation_only_and_units_are_explicit():
    target = reference_grid(9)
    estimate = target+torch.tensor([.01, -.02], dtype=torch.float64)
    report = held_out_map_metrics(estimate, target, 512, count=31)
    assert report["euclidean_query_rmse_normalized"] == pytest.approx(np.sqrt(.0005))
    assert report["euclidean_query_rmse_canvas_pixels"] == pytest.approx(512*np.sqrt(.0005))
    assert report["query_count"] == 31
    assert report["queries_used_for_optimization"] is False
