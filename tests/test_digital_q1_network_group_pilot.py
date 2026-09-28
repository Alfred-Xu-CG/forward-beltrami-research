"""The multi-pair pilot must keep image-only teachers aligned with each pair."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tools.digital_q1_network_group_pilot import read_pair_batch


def test_read_pair_batch_keeps_order_and_group_specific_teachers(tmp_path):
    pairs = []
    for index in range(2):
        fixed = tmp_path / f"fixed_{index}.png"
        moving = tmp_path / f"moving_{index}.png"
        teacher_path = tmp_path / f"teacher_{index}.npz"
        Image.fromarray(np.full((20, 22), 20 + 70 * index, np.uint8)).save(fixed)
        Image.fromarray(np.full((21, 23), 30 + 70 * index, np.uint8)).save(moving)
        teacher = np.full((1, 5, 5, 2), index / 10, np.float32)
        np.savez(teacher_path, raw_teacher_vertices=teacher)
        pairs.append((fixed, moving, teacher_path))

    fixed, moving, teacher, metadata = read_pair_batch(pairs, image_side=16)
    assert fixed.shape == moving.shape == (2, 1, 16, 16)
    assert teacher.shape == (2, 5, 5, 2)
    assert teacher[0, 0, 0, 0].item() == pytest.approx(0)
    assert teacher[1, 0, 0, 0].item() == pytest.approx(.1)
    assert fixed[0].mean() > fixed[1].mean()  # Stain intensity is inverted.
    assert metadata[0]["fixed_size_xy"] == [22, 20]
    assert metadata[1]["moving_size_xy"] == [23, 21]


def test_read_pair_batch_rejects_single_group():
    with pytest.raises(ValueError, match="at least two pairs"):
        read_pair_batch([], image_side=16)
