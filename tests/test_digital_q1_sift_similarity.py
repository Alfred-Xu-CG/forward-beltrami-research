"""Pixel-center similarity conversion and safe failure behavior."""

from __future__ import annotations

import numpy as np
import pytest

from tools.digital_q1_sift_similarity import (
    estimate_sift_similarity, pixel_affine_to_unit, save_similarity_q1,
)


def test_pixel_center_affine_conversion_commutes_with_coordinate_transform():
    pixel_matrix = np.array([[1.1, -.2, 7.], [.2, 1.1, -3.]])
    matrix, offset = pixel_affine_to_unit(pixel_matrix, 512)
    unit = np.array([.17, .63])
    pixel = 512 * unit - .5
    moving_pixel = pixel_matrix[:, :2] @ pixel + pixel_matrix[:, 2]
    np.testing.assert_allclose(matrix @ unit + offset, (moving_pixel + .5) / 512,
                               atol=1e-15)


def test_blank_pair_returns_identity_instead_of_unsafe_guess():
    image = np.zeros((32, 32), np.uint8)
    matrix, report = estimate_sift_similarity(image, image)
    np.testing.assert_array_equal(matrix, [[1, 0, 0], [0, 1, 0]])
    assert report["identity_fallback"] is True
    assert report["ransac_inliers"] == 0


def test_saved_similarity_is_valid_q1_and_reflection_is_rejected(tmp_path):
    safe = np.array([[.9, -.1, 3.], [.1, .9, -2.]])
    result = save_similarity_q1(tmp_path / "safe.npz", safe, image_side=32, control_side=9)
    assert result["saved_residual_valid"] is True
    assert result["saved_affine_det_float32"] > 0
    with pytest.raises(ValueError, match="positive finite affine"):
        save_similarity_q1(tmp_path / "bad.npz", np.array([[-1., 0., 0.], [0., 1., 0.]]),
                           image_side=32, control_side=9)
