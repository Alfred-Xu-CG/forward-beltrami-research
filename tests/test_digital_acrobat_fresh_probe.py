"""Train/evaluate phase separation for unseen ACROBAT cases."""

from pathlib import Path
import inspect
import json

import numpy as np
import pytest
from PIL import Image
import torch

from tools.digital_acrobat_fresh_probe import train_only, validate_train_ids


def test_fresh_probe_predeclares_no_flow_hint_default():
    assert inspect.signature(train_only).parameters["flow_hint"].default is False


def test_train_ids_reject_duplicate_and_nonpositive():
    assert validate_train_ids([100, 156, 315]) == [100, 156, 315]
    with pytest.raises(ValueError):
        validate_train_ids([100, 100])
    with pytest.raises(ValueError):
        validate_train_ids([0, 100])
    with pytest.raises(ValueError):
        validate_train_ids([])


def test_eval_ids_must_be_disjoint_from_frozen_training_manifest(tmp_path: Path):
    from tools.digital_acrobat_fresh_probe import validate_eval_ids

    assert validate_eval_ids([73, 193], [100, 156]) == [73, 193]
    with pytest.raises(ValueError):
        validate_eval_ids([73, 100], [100, 156])
    with pytest.raises(ValueError):
        validate_eval_ids([73, 73], [100, 156])


def test_heldout_inputs_load_without_full_teacher(tmp_path: Path):
    from tools.digital_acrobat_teacher_probe import load_case_inputs

    for stain in ("HE", "ER"):
        Image.fromarray(np.full((512, 512), 127, dtype=np.uint8)).save(
            tmp_path / f"73_{stain}_physical512.png"
        )
    np.savez_compressed(
        tmp_path / "73_DHR_physical_initial_teacher_affine.npz",
        post_affine_matrix=np.eye(2, dtype=np.float32),
        post_affine_offset=np.zeros(2, dtype=np.float32),
    )
    from torch import device

    example = load_case_inputs(tmp_path, 73, device("cpu"))
    assert example["prewarped"].shape == (1, 1, 512, 512)
    assert "raw_teacher" not in example and "target" not in example
    assert not (tmp_path / "73_DHR_physical_full_teacher_affine.npz").exists()


def test_fresh_evaluation_seals_all_predictions_before_first_teacher(monkeypatch, tmp_path: Path):
    from tools import digital_acrobat_fresh_probe as probe
    from tools.digital_q1_dhr_distill import identity_vertices

    (tmp_path / "train_manifest.json").write_text(
        json.dumps({"train_case_ids": [100]}), encoding="utf-8"
    )
    identity = identity_vertices(257, device=torch.device("cpu"))
    image = torch.zeros(1, 1, 2, 2)
    model_calls = []

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, fixed, moving):
            model_calls.append((fixed, moving))
            return identity, None, None

    monkeypatch.setattr(probe, "load_checkpoint", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(probe, "load_case_inputs", lambda *args: {
        "fixed": image, "prewarped": image,
        "matrix": torch.eye(2)[None], "offset": torch.zeros(1, 2),
    })
    monkeypatch.setattr(probe, "certify_q1_binary_map",
                        lambda *args: {"valid": True})
    teacher_reads = []

    def read_teacher(root, case, device, matrix, offset):
        for heldout in (73, 193):
            for arm in ("actual", "blank"):
                assert (tmp_path / f"{heldout}_{arm}_safe_q1.npz").is_file()
        teacher_reads.append(case)
        return {"raw_teacher": identity}

    monkeypatch.setattr(probe, "load_case_teacher", read_teacher)
    probe.evaluate_frozen(root=tmp_path, output=tmp_path,
                          test_ids=[73, 193], device="cpu")
    assert teacher_reads == [73, 193]
    assert len(model_calls) == 4  # one actual plus one blank per held-out case
