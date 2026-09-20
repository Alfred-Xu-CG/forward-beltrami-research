from pathlib import Path

from qcopt.experiments.beurling_fixed_point_residual_floor_audit import run


def test_fixed_point_residual_floor_audit(tmp_path: Path) -> None:
    result = run(tmp_path, n=32, amplitude=0.25, iteration_points=(0, 1, 2, 4))
    assert len(result["records"]) == 4
    assert result["records"][-1]["fixed_point_residual"] < result["records"][0]["fixed_point_residual"]
    assert (tmp_path / "beurling_fixed_point_residual_floor_audit.json").exists()
