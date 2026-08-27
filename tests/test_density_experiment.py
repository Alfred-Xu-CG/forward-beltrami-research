import json

from qcopt.experiments.density_equalizing import run_density_equalizing


def test_reduced_density_benchmark_saves_method_and_feasibility_evidence(tmp_path):
    metrics = run_density_equalizing(
        tmp_path,
        nx=3,
        ny=3,
        iterations=2,
        seed=9,
        target_names=("manufactured", "checkerboard"),
        make_figure=True,
    )
    assert len(metrics) == 10
    assert {item["method"] for item in metrics} == {
        "mu_lbs_fixed",
        "direct_none",
        "direct_lim_style",
        "direct_slim_style",
        "direct_amips_style",
    }
    assert {item["target"] for item in metrics} == {"manufactured", "checkerboard"}
    required = {
        "feasibility_status",
        "log_area_rmse",
        "relative_area_rmse",
        "flipped_faces",
        "certified",
        "elapsed_seconds",
    }
    assert all(required <= item.keys() for item in metrics)
    assert all(item["certified"] for item in metrics if item["method"] != "direct_none")
    for filename in ("metrics.json", "trajectories.csv", "maps.npz", "comparison.png"):
        assert (tmp_path / filename).is_file()
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["configuration"]["seed"] == 9
