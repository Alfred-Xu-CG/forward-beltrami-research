import json

from qcopt.experiments.beurling_periodic_audit import run_periodic_audit


def test_periodic_audit_separates_mean_zero_and_constant_modes(tmp_path):
    output = tmp_path / "periodic"

    result = run_periodic_audit(output, nx=32, ny=24, amplitude=0.2)

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert result["config"]["nx"] == 32
    assert metrics["mean_zero"]["converged"]
    assert metrics["mean_zero"]["equation_linf"] < 1e-8
    assert metrics["constant"]["converged"]
    assert metrics["constant"]["equation_linf"] > 0.2
