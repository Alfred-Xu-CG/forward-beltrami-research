import json

from qcopt.experiments.audit_results import audit_artifacts
from qcopt.experiments.run_all import run_all


def test_quick_runner_builds_manifest_and_independent_audit_detects_corruption(tmp_path):
    manifest = run_all(tmp_path, quick=True, seed=23)
    assert manifest["mode"] == "quick"
    assert manifest["files"]
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])

    audit = audit_artifacts(tmp_path)
    assert audit["status"] == "VERIFIED"
    assert not audit["mismatches"]

    metrics_path = tmp_path / "i_to_s" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["methods"][0]["flipped_faces"] += 1
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    corrupted = audit_artifacts(tmp_path)
    assert corrupted["status"] == "FAILED"
    assert any("flipped_faces" in item for item in corrupted["mismatches"])
