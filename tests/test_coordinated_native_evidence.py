import copy

import numpy as np
from PIL import Image
import pytest
import torch

pytest.importorskip("deeperhistreg")

from tools.coordinated_native_evidence import load_native_preprocessed_pair, native_evidence_parameters


def images(tmp_path, mode="RGB", side=32):
    rng = np.random.default_rng(61001)
    shape = (side, side) if mode == "L" else (side, side, 4 if mode == "RGBA" else 3)
    paths = []
    for name in ("fixed", "moving"):
        path = tmp_path / (name + ".png")
        Image.fromarray(rng.integers(8, 248, shape, dtype=np.uint8), mode).save(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_exact_native_pipeline_capture(tmp_path, mode):
    import deeperhistreg
    fixed, moving = images(tmp_path, mode)
    params = native_evidence_parameters()
    pipeline = deeperhistreg.direct_registration.DeeperHistReg_FullResolution(copy.deepcopy(params))
    pipeline.source_path, pipeline.target_path = str(moving), str(fixed)
    pipeline.load_images()
    pipeline.run_prepreprocessing()  # Only capture preprocessing, never registration.
    f, m, metadata = load_native_preprocessed_pair(fixed, moving, expected_side=32)
    torch.testing.assert_close(f, pipeline.pre_target, rtol=0, atol=0)
    torch.testing.assert_close(m, pipeline.pre_source, rtol=0, atol=0)
    assert f.shape == m.shape == (1, 1, 32, 32)
    assert f.dtype == m.dtype == torch.float32
    assert not f.requires_grad and not m.requires_grad
    assert metadata["coordinate_frame_unchanged"]
    assert metadata["initial_resample_ratio"] == 1
    assert metadata["initial_gaussian_kernel_size"] == 1
    assert metadata["original_modes"] == dict(fixed=mode, moving=mode)


def test_rgba_rejected_without_silent_alpha_conversion(tmp_path):
    fixed, moving = images(tmp_path, "RGBA")
    with pytest.raises(ValueError, match="RGBA"):
        load_native_preprocessed_pair(fixed, moving, expected_side=32)


def test_constant_native_normalization_not_silently_accepted(tmp_path):
    fixed, moving = images(tmp_path, "L")
    Image.fromarray(np.full((32, 32), 127, dtype=np.uint8)).save(fixed)
    with pytest.raises(ValueError, match="nonfinite"):
        load_native_preprocessed_pair(fixed, moving, expected_side=32)


def test_frame_changing_config_rejected(tmp_path):
    fixed, moving = images(tmp_path)
    config = native_evidence_parameters()
    config["loading_params"]["source_resample_ratio"] = .5
    with pytest.raises(ValueError, match="ratio"):
        load_native_preprocessed_pair(fixed, moving, config=config, expected_side=32)


def test_no_affine_or_registration_path_called(tmp_path, monkeypatch):
    import deeperhistreg
    fixed, moving = images(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("registration is outside frozen evidence extraction")
    cls = deeperhistreg.direct_registration.DeeperHistReg_FullResolution
    monkeypatch.setattr(cls, "run_registration", forbidden)
    monkeypatch.setattr(cls, "run_initial_registration", forbidden)
    monkeypatch.setattr(cls, "run_nonrigid_registration", forbidden)
    load_native_preprocessed_pair(fixed, moving, expected_side=32)
