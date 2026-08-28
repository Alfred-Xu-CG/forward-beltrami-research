import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def test_sync_delivery_copies_hidden_state_and_verifies_manifest(tmp_path):
    powershell = shutil.which("pwsh")
    assert powershell is not None
    project_root = Path(__file__).parents[1]
    script = project_root / "scripts" / "sync_delivery.ps1"
    source = tmp_path / "source"
    target = tmp_path / "target"
    artifact = source / "artifacts" / "evidence.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("verified evidence\n", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "files": [{"path": "evidence.txt", "bytes": artifact.stat().st_size, "sha256": digest}]
    }
    (source / "artifacts" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-File",
            str(script),
            "-SourcePath",
            str(source),
            "-TargetPath",
            str(target),
        ],
        check=False,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["manifest_verified"] is True
    assert result["verified_files"] == 1
    assert (target / ".git" / "HEAD").is_file()
    assert (target / "artifacts" / "evidence.txt").read_text(encoding="utf-8") == "verified evidence\n"


def test_sync_delivery_refuses_nonempty_unrecognized_target(tmp_path):
    powershell = shutil.which("pwsh")
    assert powershell is not None
    project_root = Path(__file__).parents[1]
    script = project_root / "scripts" / "sync_delivery.ps1"
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (target / "unrelated.txt").write_text("do not overwrite", encoding="utf-8")

    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-File",
            str(script),
            "-SourcePath",
            str(source),
            "-TargetPath",
            str(target),
        ],
        check=False,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "not empty" in completed.stderr
    assert (target / "unrelated.txt").read_text(encoding="utf-8") == "do not overwrite"
