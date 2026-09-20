from pathlib import Path

from qcopt.experiments.tutte_weight_learning_safe_audit import run


def test_safe_tutte_weight_update_preserves_margin(tmp_path: Path) -> None:
    result = run(tmp_path, n=16, iterations=2, initial_step=0.5, determinant_margin=1e-8)
    assert result["accepted_steps"] >= 1
    assert result["flipped_faces"] == 0
    assert result["min_determinant"] >= result["determinant_margin"]
    assert (tmp_path / "tutte_weight_learning_safe_audit.json").exists()
