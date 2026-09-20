from pathlib import Path


def test_unstructured_directed_audit_records_vjp_and_topology():
    from qcopt.experiments.tutte_directed_unstructured_audit import run

    result = run(Path("D:/QC_optimization/tmp/test_tutte_directed_unstructured_audit"), interior=120, side=12)
    assert result["vertices"] > 100
    assert all(record["finite_gradient"] for record in result["records"])
    assert all(record["flipped_faces"] >= 0 for record in result["records"])
