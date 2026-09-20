from pathlib import Path


def test_random_highres_projection_receipt_has_explicit_residual_and_topology():
    from qcopt.experiments.compatibility_projection_random_highres_audit import run

    result = run(Path("D:/QC_optimization/tmp/test_compatibility_projection_random_highres"), n=24, max_nfev=2)
    record = result["record"]
    assert record["finite"]
    assert "relative_residual" in record
    assert record["flipped_faces"] >= 0
