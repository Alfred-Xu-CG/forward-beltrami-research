import json

from qcopt.experiments.multichart_registration import run_multichart_registration


def test_multichart_experiment_quantifies_transition_aware_and_wrong_seams(tmp_path):
    metrics = run_multichart_registration(tmp_path, nx=3, ny=2, make_figure=True)
    assert metrics["transition_aware_seam_residual"] < 1e-10
    assert metrics["raw_coordinate_equality_physical_seam_residual"] > 0.1
    assert metrics["algebraic_residual"] < 1e-10
    assert metrics["max_angular_distortion"] >= 1.0
    assert metrics["landmark_rmse"] < 1e-8
    assert all(metrics["chart_certificates"].values())
    for filename in ("metrics.json", "maps.npz", "comparison.png"):
        assert (tmp_path / filename).is_file()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["compatibility_equation"] == "g_d(tau_S(p)) = tau_T(g_c(p))"
