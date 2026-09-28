import json

import numpy as np
from PIL import Image

from tools.digital_wsi_pair import make_identity, make_tissue_centroid_translation, save_transform


def test_identity_declares_source_target_frames(tmp_path):
    moving = tmp_path / "moving.jpg"
    fixed = tmp_path / "fixed.jpg"
    Image.new("RGB", (12, 8)).save(moving)
    Image.new("RGB", (10, 9)).save(fixed)

    transform = make_identity(moving, fixed)
    assert transform["moving_size_xy"] == [12, 8]
    assert transform["fixed_size_xy"] == [10, 9]
    assert transform["direction"] == "moving_to_fixed"
    assert transform["coordinate_frame"] == "top_left_origin_pixel_xy"
    np.testing.assert_array_equal(transform["matrix_moving_to_fixed_xy"], np.eye(3))
    assert transform["det_linear"] == 1.0

    output = tmp_path / "identity.json"
    save_transform(transform, output)
    assert json.loads(output.read_text())["moving_size_xy"] == [12, 8]


def test_identity_does_not_need_landmark_files(tmp_path):
    moving = tmp_path / "moving.jpg"
    fixed = tmp_path / "fixed.jpg"
    Image.new("RGB", (4, 3)).save(moving)
    Image.new("RGB", (5, 6)).save(fixed)
    assert make_identity(moving, fixed)["det_linear"] == 1.0


def test_tissue_translation_uses_images_only_and_preserves_orientation(tmp_path):
    moving = tmp_path / "moving.png"
    fixed = tmp_path / "fixed.png"
    source = np.full((40, 50, 3), 255, dtype=np.uint8)
    target = source.copy()
    source[10:20, 12:22] = 0
    target[14:24, 18:28] = 0
    Image.fromarray(source).save(moving)
    Image.fromarray(target).save(fixed)

    transform = make_tissue_centroid_translation(moving, fixed)
    np.testing.assert_allclose(
        transform["matrix_moving_to_fixed_xy"],
        [[1, 0, 6], [0, 1, 4], [0, 0, 1]],
        atol=1e-12,
    )
    assert transform["det_linear"] == 1.0
