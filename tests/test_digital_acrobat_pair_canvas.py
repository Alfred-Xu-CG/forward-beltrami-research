"""Physical-pixel-scale checks for paired ACROBAT development canvases."""

from __future__ import annotations

import pytest

from tools.digital_acrobat_pair_canvas import compute_layout, resolution_to_mpp


def test_resolution_conversion_cm_and_inch() -> None:
    assert resolution_to_mpp((10_000, 1), 3) == 1.0
    assert resolution_to_mpp((25_400, 1), 2) == 1.0
    with pytest.raises(ValueError, match="physical resolution"):
        resolution_to_mpp((10_000, 1), 1)


def test_pair_layout_keeps_one_physical_scale_and_aspect() -> None:
    layout = compute_layout([(500, 1000, 1.0, 1.0),
                             (400, 800, 1.0, 1.0)], side=100)
    assert layout[0]["resized_wh"] == [100, 50]
    assert layout[0]["padding_xy"] == [0, 25]
    assert layout[1]["resized_wh"] == [80, 40]
    assert layout[1]["padding_xy"] == [10, 30]
    assert layout[0]["canvas_mpp"] == layout[1]["canvas_mpp"] == 10.0


def test_layout_requires_positive_valid_physical_metadata() -> None:
    with pytest.raises(ValueError):
        compute_layout([(500, 1000, 0.0, 1.0)], side=100)
