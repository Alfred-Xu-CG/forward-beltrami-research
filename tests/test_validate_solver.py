import json

from qcopt.experiments.validate_solver import run_solver_validation


def test_solver_validation_saves_residual_gradient_and_scaling_evidence(tmp_path):
    report = run_solver_validation(tmp_path, grid_sizes=(2, 3), seed=13, make_figure=True)
    assert report["acceptance"]["forward_residual_below_1e-10"]
    assert report["acceptance"]["adjoint_residual_below_1e-10"]
    assert report["acceptance"]["stable_gradient_relative_error_below_1e-5"]
    assert report["acceptance"]["manufactured_reconstruction_below_1e-8"]
    assert {item["solver"] for item in report["gradient_sweeps"]} == {
        "lbs",
        "lsqc_unweighted",
        "lsqc_weighted",
    }
    assert len(report["scaling"]) == 2
    for filename in ("validation.json", "scaling.csv", "gradient_sweep.png"):
        assert (tmp_path / filename).is_file()
    saved = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert saved["configuration"]["seed"] == 13
