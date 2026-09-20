from pathlib import Path

from qcopt.experiments.mmatrix_global_unstructured_wide_audit import run


def test_wide_unstructured_decoder_audit_records_both_rings(tmp_path: Path) -> None:
    result = run(tmp_path, interior=80, side=12, rings=(1, 2))
    assert [record["rings"] for record in result["records"]] == [1, 2]
    assert all(record["finite"] for record in result["records"])
    assert (tmp_path / "mmatrix_global_unstructured_wide_audit.json").exists()
