"""Frame-preserving supplied-affine initialization, without a real DHR fit."""

import argparse
import json
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools import coordinated_dhr_common as common


MATRIX = np.array([[0.93, 0.07], [-0.03, 1.04]], dtype=np.float32)
OFFSET = np.array([0.08, -0.06], dtype=np.float32)


def mock_run(tmp_path, monkeypatch, *, side=512, legacy=False,
             source_shape=None, target_shape=None, ratio=1, pad=None):
    """Exercise the actual override inside run(), using a tiny mock lifecycle."""
    captured = {}

    class MockPipeline:
        def __init__(self, params):
            captured["params"] = params
            captured["pipeline"] = self
            self.pre_source = torch.zeros(1, 1, *(source_shape or (side, side)))
            self.pre_target = torch.zeros(1, 1, *(target_shape or (side, side)))
            self.padding_params = {
                "initial_resample_ratio": ratio,
                "pad_1": ((0, 1 if pad == "pad_1" else 0), (0, 0)),
                "pad_2": ((0, 0), (1 if pad == "pad_2" else 0, 0)),
            }
            self.preprocessing_time = 0.0
            self.nonrigid_registration_time = 0.0

        def run_registration(self, moving, fixed, output):
            captured["paths"] = (moving, fixed, output)
            self.run_initial_registration()

    monkeypatch.setattr(
        common.deeperhistreg.direct_registration,
        "DeeperHistReg_FullResolution", MockPipeline,
    )
    archive = tmp_path / "supplied_affine.npz"
    np.savez(archive, post_affine_matrix=MATRIX, post_affine_offset=OFFSET)
    args = argparse.Namespace(
        fixed=tmp_path / "fixed.png", moving=tmp_path / "moving.png",
        affine=archive, output=tmp_path / "result", threads=1, device="cpu",
    )
    if not legacy:
        args.image_side = side
    common.run(args)
    captured["report"] = json.loads((args.output / "runtime.json").read_text())
    return captured


@pytest.mark.parametrize("side,legacy", [(512, True), (512, False), (1024, False)])
def test_supplied_affine_has_same_unit_square_frame(tmp_path, monkeypatch, side, legacy):
    data = mock_run(tmp_path, monkeypatch, side=side, legacy=legacy)
    pipeline = data["pipeline"]
    identity = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    # Independent expected map from pixel-center unit-square coordinates.
    base = F.affine_grid(identity, (1, 1, side, side), align_corners=False)
    unit = (base + 1) / 2
    expected_unit = unit @ torch.from_numpy(MATRIX).T + torch.from_numpy(OFFSET)
    actual_unit = (base + pipeline.initial_displacement_field + 1) / 2
    torch.testing.assert_close(actual_unit, expected_unit, rtol=0, atol=2e-7)
    assert pipeline.current_displacement_field is pipeline.initial_displacement_field
    assert pipeline.initial_registration_time == 0
    params = data["params"]
    assert params["preprocessing_params"]["initial_resolution"] == max(768, side)
    nonrigid = params["nonrigid_registration_params"]
    assert nonrigid["registration_size"] == side
    assert nonrigid["num_levels"] == nonrigid["used_levels"] == 5
    assert nonrigid["iterations"] == [30] * 5
    assert nonrigid["alphas"] == [1.5] * 5
    assert nonrigid["learning_rates"] == [.005, .0025, .0025, .0025, .0025]
    assert data["report"]["image_side"] == side
    assert data["report"]["initial_resample_ratio"] == 1
    assert "fixed/target" in data["report"]["affine_frame"]
    assert "supplied archive" in data["report"]["affine_provenance"]


@pytest.mark.parametrize("overrides,message", [
    ({"source_shape": (512, 513)}, "square shared canvases"),
    ({"target_shape": (511, 512)}, "square shared canvases"),
    ({"side": 1024, "source_shape": (512, 512)}, "square shared canvases"),
    ({"ratio": 2}, "initial resampling"),
    ({"pad": "pad_1"}, "unpadded shared input canvases"),
    ({"pad": "pad_2"}, "unpadded shared input canvases"),
])
def test_reject_changed_frame(tmp_path, monkeypatch, overrides, message):
    with pytest.raises(ValueError, match=message):
        mock_run(tmp_path, monkeypatch, **overrides)


@pytest.mark.parametrize("option,expected", [([], 512), (["--image-side", "1024"], 1024)])
def test_command_line_default_compatibility(monkeypatch, option, expected):
    captured = []
    monkeypatch.setattr(common, "run", captured.append)
    monkeypatch.setattr(sys, "argv", [
        "coordinated_dhr_common", "--fixed", "fixed.png", "--moving", "moving.png",
        "--affine", "affine.npz", "--output", "result", *option,
    ])
    common.main()
    assert captured[0].image_side == expected


@pytest.mark.parametrize("matrix,offset", [
    (np.array([[np.nan, 0.], [0., 1.]]), OFFSET),
    (np.array([[np.inf, 0.], [0., 1.]]), OFFSET),
    (MATRIX, np.array([np.nan, 0.])),
    (MATRIX, np.array([0., np.inf])),
    (np.array([[-1., 0.], [0., 1.]]), OFFSET),
])
def test_reject_invalid_affine_before_pipeline(tmp_path, monkeypatch, matrix, offset):
    def forbidden_pipeline(*args, **kwargs):
        pytest.fail("invalid affine must fail before constructing a pipeline")

    monkeypatch.setattr(
        common.deeperhistreg.direct_registration,
        "DeeperHistReg_FullResolution", forbidden_pipeline,
    )
    archive = tmp_path / "invalid_affine.npz"
    np.savez(archive, post_affine_matrix=matrix, post_affine_offset=offset)
    output = tmp_path / "result"
    args = argparse.Namespace(affine=archive, output=output, image_side=512)
    with pytest.raises(ValueError, match="positive affine required"):
        common.run(args)
    assert not output.exists()
