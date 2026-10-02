import json
import numpy as np
import pytest

from tools.coordinated_dhr_released_score import field_corner_diagnostics, score, summary


def test_all_corners_rectangular_chunked_affine_and_reflection():
    y, x = np.meshgrid(np.arange(11), np.arange(19), indexing="ij")
    field = np.stack((.2*x+.1*y+3, -.1*x+.3*y-2)).astype(np.float32)
    for chunk in (1, 3, 128):
        result = field_corner_diagnostics(field, chunk)
        assert result["checked_corner_count"] == 4*10*18
        assert result["nonpositive_corners"] == 0
        assert result["minimum_corner_ratio"] == pytest.approx(1.57, abs=2e-6)
        assert result["global_homeomorphism_certified"] is False
    reflection = np.stack((-2*x, np.zeros_like(y)))
    assert field_corner_diagnostics(reflection)["nonpositive_corners"] == 4*10*18


def test_finite_every_label_and_unfinished_prediction_rejected(tmp_path):
    assert summary([1., 3.], ["a", "b"])["mean"] == 2
    with pytest.raises(ValueError):
        summary([1., np.nan], ["a", "b"])
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"prediction_complete": False, "annotations_read": False}))
    with pytest.raises(ValueError, match="complete image-only"):
        score(path, tmp_path/"labels_do_not_exist", tmp_path)
