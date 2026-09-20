from pathlib import Path

from qcopt.experiments.mmatrix_barycentric_enrichment_audit import run


def test_barycentric_enrichment_preserves_refined_orientation(tmp_path: Path) -> None:
    result = run(tmp_path, interior=60, side=12)
    assert result["finite"]
    assert result["flipped_refined_faces"] == 0
    assert result["boundary_exact_max_error"] == 0.0
    assert (tmp_path / "mmatrix_barycentric_enrichment_audit.json").exists()


def test_barycentric_anisotropic_center_ablation_is_finite(tmp_path: Path) -> None:
    result = run(tmp_path / "anisotropic", interior=40, side=10, anisotropic_centers=True)
    assert result["finite"]
    assert result["flipped_refined_faces"] == 0
    assert result["center_fit_residual_mean"] >= 0.0
