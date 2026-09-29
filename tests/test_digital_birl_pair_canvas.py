from pathlib import Path

import pytest
from PIL import Image

from tools.digital_birl_pair_canvas import make_canvas


def test_canvas_preserves_common_scale_and_refuses_overwrite(tmp_path: Path) -> None:
    moving, fixed = tmp_path / "moving.jpg", tmp_path / "fixed.jpg"
    Image.new("RGB", (160, 120), "red").save(moving)
    Image.new("RGB", (200, 100), "blue").save(fixed)
    outputs = [tmp_path / name for name in ("moving.png", "fixed.png", "meta.json")]
    report = make_canvas(moving, fixed, *outputs, side=512)
    assert report["moving"]["resized_wh"] == [410, 307]
    assert report["fixed"]["resized_wh"] == [512, 256]
    assert report["moving"]["effective_original_to_canvas_scale_xy"][0] == pytest.approx(410 / 160)
    assert report["fixed"]["effective_original_to_canvas_scale_xy"][0] == pytest.approx(512 / 200)
    assert report["landmarks_used"] is False
    with pytest.raises(ValueError, match="nonexistent"):
        make_canvas(moving, fixed, *outputs, side=512)
