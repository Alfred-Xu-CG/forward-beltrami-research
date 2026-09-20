from qcopt.experiments.tutte_symmetric_nonsymmetric_audit import run


def test_positive_directed_tutte_audit_small_mesh(tmp_path):
    result = run(tmp_path, n=8, seed=3)
    for record in result["records"]:
        assert record["finite"]
        assert record["flipped_faces"] == 0
        assert record["minimum_signed_area"] > 0.0
