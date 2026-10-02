"""Released reproduction keeps algorithm settings and records every attempt."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from tools.coordinated_dhr_released_baseline import PAIRS, configure, run


def preset():
    return dict(device="cuda:0", run_initial_registration=True, run_nonrigid_registration=True,
                loading_params=dict(loader="tiff", source_resample_ratio=.2, target_resample_ratio=.2, pad_value=255),
                saving_params=dict(final_saver="tiff", saver="pil"),
                preprocessing_params=dict(initial_resolution=2048, clahe=True, flip_intensity=True, save_results=True),
                initial_registration_params=dict(device="cuda:0", cuda=True, save_results=True,
                    initial_registration_function="multi_feature", registration_sizes=[150, 200], angle_step=180),
                nonrigid_registration_params=dict(device="cuda:0", save_results=True,
                    registration_size=2048, iterations=[100]*7, num_levels=7, used_levels=7,
                    learning_rates=[.005]+[.0025]*6, alphas=[1.5]*7))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_only_io_and_device_change(device, tmp_path):
    base = preset()
    original = copy.deepcopy(base)
    result = configure(base, device=device, output=tmp_path)
    assert base == original
    for section in ("preprocessing_params", "initial_registration_params", "nonrigid_registration_params"):
        algorithm = lambda value: {k: v for k, v in value.items() if k not in ("device", "cuda", "save_results")}
        assert algorithm(result[section]) == algorithm(base[section])
    assert result["run_initial_registration"] is True
    assert result["loading_params"]["source_resample_ratio"] == 1
    assert result["loading_params"]["target_resample_ratio"] == 1


@pytest.mark.parametrize("choice", ["fast", "standard"])
def test_native_pipeline_without_affine_and_failure_retention(tmp_path, choice):
    source = tmp_path / "source"
    for section in {s for pair in PAIRS for s in pair}:
        folder = source / str(section) / "images"
        folder.mkdir(parents=True)
        Image.new("RGB", (19+section, 21+section), (120, 100, 80)).save(folder / "image.tif")
    calls, selected = [], []
    class Pipeline:
        def __init__(self, params):
            self.params = params
        def run_registration(self, moving, fixed, output):
            calls.append((moving, fixed))
            assert self.params["initial_registration_params"]["initial_registration_function"] == "multi_feature"
            if len(calls) == 2:
                raise RuntimeError("deliberate native initializer failure")
            self.pre_source = torch.zeros(1, 1, 12, 10)
            self.source = torch.zeros(1, 3, 30, 29)
            self.current_displacement_field = torch.zeros(1, 12, 10, 2)
            self.initial_transform = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
            self.preprocessing_time = self.initial_registration_time = self.nonrigid_registration_time = .1
            self.total_registration_time = .3
            final = Path(output) / "released_dhr/Results_Final"
            final.mkdir(parents=True)
            (final / "displacement_field.mha").touch()
            (final / "postprocessing_params.json").write_text("{}")
    def get(name):
        selected.append(name)
        return preset()
    dhr = SimpleNamespace(configs=SimpleNamespace(default_initial_nonrigid_fast=lambda: get("fast"),
                        default_initial_nonrigid=lambda: get("standard")),
                        direct_registration=SimpleNamespace(DeeperHistReg_FullResolution=Pipeline))
    args = argparse.Namespace(source_data=source, output=tmp_path/"output", device="cpu", threads=1, preset=choice)
    report = run(args, dhr=dhr)
    assert selected == [choice]
    assert len(calls) == 3
    assert [r["status"] for r in report["rows"]] == ["ok", "failed", "ok"]
    assert report["successful_pairs"] == 2 and report["pair_denominator"] == 3
    assert report["annotations_read"] is False and report["prediction_complete"] is True
    assert report["rows"][0]["fixed_original_wh"] == [22, 24]
    assert json.loads((args.output/"predictions.json").read_text())["rows"] == report["rows"]
