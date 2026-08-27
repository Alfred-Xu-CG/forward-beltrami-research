import json

from qcopt.experiments.i_to_s import run_i_to_s


def test_reduced_i_to_s_benchmark_runs_all_methods_and_saves_evidence(tmp_path):
    metrics = run_i_to_s(
        tmp_path, nx=3, ny=3, iterations=2, seed=5, make_figure=True
    )
    methods = {item["method"] for item in metrics}
    assert methods == {
        "mu_lbs_fixed",
        "mu_lsqc_free",
        "direct_none",
        "direct_lim_style",
        "direct_slim_style",
        "direct_amips_style",
    }
    required = {
        "data_loss",
        "soft_dice",
        "flipped_faces",
        "minimum_signed_area_ratio",
        "max_qc_dilation",
        "certified",
        "elapsed_seconds",
        "iterations",
    }
    assert all(required <= item.keys() for item in metrics)
    assert all(item["certified"] for item in metrics if item["method"] != "direct_none")

    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "trajectories.csv").is_file()
    assert (tmp_path / "maps.npz").is_file()
    assert (tmp_path / "comparison.png").is_file()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["configuration"]["seed"] == 5
    assert len(saved["methods"]) == 6
