import json

import numpy as np

from qcopt.experiments.beurling_boundary_audit import run_boundary_audit


def test_boundary_audit_records_periodic_image_error(tmp_path):
    output = tmp_path / "audit"

    result = run_boundary_audit(output, nx=64, ny=48, padding_factors=(2, 4))

    assert (output / "config.json").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "fields.npz").is_file()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    fields = np.load(output / "fields.npz")
    assert result["config"]["nx"] == 64
    assert metrics["padding_2"]["max_abs_difference"] > 1e-4
    assert metrics["padding_4"]["max_abs_difference"] > 1e-4
    assert fields["source"].shape == (48, 64)
