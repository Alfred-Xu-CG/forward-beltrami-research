from __future__ import annotations

import pytest
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.benchmarks import (
    ANALYTIC_DEFORMATION_NAMES,
    IMAGE_NAMES,
    identity_query_grid,
    make_registration_pair,
    sample_analytic_deformation,
    synthetic_image,
    warp_image_backward,
)


def test_all_analytic_deformations_have_positive_bounds_and_certified_p1_maps() -> None:
    mesh = structured_rectangle(10, 10)  # 11 by 11 control vertices

    for name in ANALYTIC_DEFORMATION_NAMES:
        sample = sample_analytic_deformation(mesh, name)
        assert sample.name == name
        assert sample.vertices.shape == (mesh.n_vertices, 2)
        assert sample.continuous_determinant_lower_bound > 0.0
        assert sample.injectivity_report.certified, name
        assert sample.injectivity_report.minimum_signed_area_ratio > 0.0


def test_synthetic_images_are_deterministic_finite_nonconstant_and_normalized() -> None:
    for name in IMAGE_NAMES:
        first = synthetic_image(name, height=73, width=79, dtype=torch.float64)
        second = synthetic_image(name, height=73, width=79, dtype=torch.float64)
        assert first.shape == (1, 1, 73, 79)
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
        assert bool(torch.isfinite(first).all())
        assert float(first.min()) >= 0.0
        assert float(first.max()) <= 1.0
        assert float(first.std()) > 0.03


def test_backward_warp_identity_is_exact_at_align_corners_pixel_centers() -> None:
    image = synthetic_image("textured", height=57, width=61, dtype=torch.float64)
    grid = identity_query_grid(57, 61, dtype=torch.float64)

    warped = warp_image_backward(image, grid)

    torch.testing.assert_close(warped, image, atol=2e-14, rtol=0.0)


def test_registration_pair_uses_explicit_fixed_to_moving_backward_map() -> None:
    pair = make_registration_pair(
        "medical_phantom",
        deformation="boundary_sliding",
        height=64,
        width=64,
        dtype=torch.float64,
    )

    recomputed = warp_image_backward(pair.moving, pair.backward_map)
    torch.testing.assert_close(pair.fixed, recomputed, atol=0.0, rtol=0.0)
    assert float(torch.mean((pair.fixed - pair.moving).square())) > 1e-5
    assert pair.convention == "backward_map_fixed_to_moving"


def test_registration_pair_rejects_maps_that_leave_the_sampled_unit_square() -> None:
    with pytest.raises(ValueError, match="unit square"):
        make_registration_pair(
            "medical_phantom",
            deformation="shear",
            height=33,
            width=35,
            dtype=torch.float64,
        )


def test_backward_warp_rejects_nonfinite_images_and_coordinates() -> None:
    image = synthetic_image("smooth_blobs", height=17, width=19, dtype=torch.float64)
    grid = identity_query_grid(17, 19, dtype=torch.float64)
    bad_grid = grid.clone()
    bad_grid[4, 5, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        warp_image_backward(image, bad_grid)

    bad_image = image.clone()
    bad_image[0, 0, 3, 2] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        warp_image_backward(bad_image, grid)
