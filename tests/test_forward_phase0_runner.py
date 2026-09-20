import json

import numpy as np

from qcopt.experiments.forward_phase0 import run_phase0


def test_phase0_runner_writes_reproducible_artifacts(tmp_path):
    output = tmp_path / "phase0"

    result = run_phase0(output, nx=32, ny=24, amplitude=0.08, seed=17)

    assert result["config"]["nx"] == 32
    assert result["config"]["ny"] == 24
    assert (output / "config.json").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "maps.npz").is_file()
    assert (output / "manifest.json").is_file()

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    maps = np.load(output / "maps.npz")
    assert config["seed"] == 17
    assert metrics["smooth_twist"]["min_det"] > 0.5
    assert maps["source"].shape == ((32 + 1) * (24 + 1), 2)
    assert maps["smooth_twist"].shape == maps["source"].shape
