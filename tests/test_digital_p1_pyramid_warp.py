import numpy as np
import pytest

from tools.digital_p1_pyramid_warp import warp_tiled, warped_tile


def _same_layout(side=20):
    image = {"original_hw": [side, side],
             "effective_original_to_canvas_scale_xy": [1., 1.],
             "padding_xy": [0, 0]}
    return {"side": side, "fixed": image, "moving": image}


def test_identity_pyramid_image_is_exact_and_partition_independent():
    axis = np.linspace(0, 1, 5)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((xx, yy), axis=-1)
    yy_img, xx_img = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    moving = np.stack((xx_img + yy_img, 2 * xx_img, 3 * yy_img), axis=-1).astype(np.uint8)
    tiled, _ = warp_tiled(vertices, _same_layout(), moving, (20, 20), 6)
    full, _ = warp_tiled(vertices, _same_layout(), moving, (20, 20), 20)
    assert np.array_equal(tiled, moving)
    assert np.array_equal(tiled, full)


def test_nonaffine_p1_warp_partition_independent():
    axis = np.linspace(0, 1, 5)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((xx + .02 * np.sin(np.pi * xx) * np.sin(np.pi * yy), yy), axis=-1)
    moving = np.arange(20 * 20 * 3, dtype=np.uint32).reshape(20, 20, 3).astype(np.uint8)
    full = warped_tile(vertices, _same_layout(), moving, (20, 20), 0, 0, 20, 20)
    for side in (1, 6, 11):
        tiled, _ = warp_tiled(vertices, _same_layout(), moving, (20, 20), side)
        assert np.array_equal(tiled, full)


def test_non_uint8_rgb_rejected_instead_of_silent_intensity_conversion():
    axis = np.linspace(0, 1, 5)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((xx, yy), axis=-1)
    with pytest.raises(ValueError, match="uint8 RGB"):
        warped_tile(vertices, _same_layout(), np.zeros((20, 20, 3), dtype=np.uint16),
                    (20, 20), 0, 0, 4, 4)


@pytest.mark.parametrize("level_side", [3, 7, 13])
def test_identity_at_noninteger_pyramid_scale_does_not_create_white_edge(level_side):
    axis = np.linspace(0, 1, 5)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((xx, yy), axis=-1)
    black = np.zeros((level_side, level_side, 3), dtype=np.uint8)
    output = warped_tile(vertices, _same_layout(20), black,
                         (level_side, level_side), 0, 0, level_side, level_side)
    assert np.array_equal(output, black)


def test_true_outside_sample_still_uses_white_constant_fill():
    axis = np.linspace(0, 1, 5)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((xx + .1, yy), axis=-1)
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    output = warped_tile(vertices, _same_layout(20), black, (20, 20), 0, 0, 20, 20)
    assert np.array_equal(output[:, :-2], black[:, :-2])
    assert np.all(output[:, -2:] == 255)
