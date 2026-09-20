from pathlib import Path


def test_identity_boundary_control_exposes_nonzero_beltrami_residual():
    from qcopt.experiments.rectangle_boundary_free_control import run

    result = run(Path("D:/QC_optimization/tmp/test_rectangle_boundary_free_control"), sizes=(17, 25))
    records = result["records"]
    assert len(records) == 2
    assert all(record["finite"] for record in records)
    assert all(record["min_triangle_determinant"] > 0.0 for record in records)
    assert all(record["equation_residual"] > 0.1 for record in records)
