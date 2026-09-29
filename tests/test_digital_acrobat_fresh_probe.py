"""Train/evaluate phase separation for unseen ACROBAT cases."""

from pathlib import Path
import inspect
import json

import numpy as np
import pytest
from PIL import Image
import torch

from tools.digital_acrobat_fresh_probe import (
    dihedral_augment, train_only, validate_train_ids,
)


def test_dihedral_augmentation_conjugates_vertex_map_and_permutates_images():
    axis = torch.linspace(0, 1, 5)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    shear = torch.stack((xx + .1 * yy, yy), dim=-1)[None]
    image = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    for flip in (False, True):
        for turns in range(4):
            fixed, moving, transformed_identity = dihedral_augment(
                image, 2 * image, identity, turns=turns, flip=flip,
            )
            expected = torch.rot90(
                torch.flip(image, (-1,)) if flip else image, turns, (-2, -1),
            )
            assert torch.equal(fixed, expected)
            assert torch.equal(moving, 2 * expected)
            assert torch.allclose(transformed_identity, identity, atol=1e-7)
    _, _, flipped_shear = dihedral_augment(image, image, shear, turns=0, flip=True)
    assert torch.allclose(flipped_shear[..., 0], xx[None] - .1 * yy[None], atol=1e-7)
    _, _, rotated_shear = dihedral_augment(image, image, shear, turns=1, flip=False)
    assert torch.allclose(rotated_shear[..., 1], yy[None] - .1 * xx[None], atol=1e-7)


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


def test_fresh_evaluation_refuses_to_overwrite_existing_heldout_report(tmp_path: Path):
    from tools.digital_acrobat_fresh_probe import evaluate_frozen

    (tmp_path / "train_manifest.json").write_text(
        json.dumps({"train_case_ids": [100]}), encoding="utf-8"
    )
    (tmp_path / "heldout_report.json").write_text("prior result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluate_frozen(root=tmp_path, output=tmp_path,
                        test_ids=[73], device="cpu")


def test_predict_only_seals_maps_without_reading_any_teacher(monkeypatch, tmp_path: Path):
    from tools import digital_acrobat_fresh_probe as probe
    from tools.digital_q1_dhr_distill import identity_vertices

    (tmp_path / "train_manifest.json").write_text(
        json.dumps({"train_case_ids": [100]}), encoding="utf-8"
    )
    identity = identity_vertices(257, device=torch.device("cpu"))
    image = torch.zeros(1, 1, 2, 2)

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, fixed, moving):
            return identity, None, None

    monkeypatch.setattr(probe, "load_checkpoint", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(probe, "load_case_inputs", lambda *args: {
        "fixed": image, "prewarped": image,
        "matrix": torch.eye(2)[None], "offset": torch.zeros(1, 2),
    })
    monkeypatch.setattr(probe, "load_case_teacher",
                        lambda *args: (_ for _ in ()).throw(AssertionError("teacher read")))
    monkeypatch.setattr(probe, "certify_q1_binary_map", lambda *args: {"valid": True})
    report = probe.evaluate_frozen(root=tmp_path, output=tmp_path,
                                   test_ids=[73], device="cpu", prediction_only=True)
    assert report["mode"] == "frozen_prediction_only_no_teacher"
    assert (tmp_path / "73_actual_safe_q1.npz").is_file()
    assert (tmp_path / "73_blank_safe_q1.npz").is_file()
    assert not (tmp_path / "heldout_report.json").exists()
    assert (tmp_path / "prediction_manifest.json").is_file()


def test_score_sealed_uses_only_saved_maps_and_teacher(monkeypatch, tmp_path: Path):
    from tools import digital_acrobat_fresh_probe as probe
    from tools.digital_q1_dhr_distill import identity_vertices

    (tmp_path / "train_manifest.json").write_text(
        json.dumps({"train_case_ids": [100]}), encoding="utf-8"
    )
    (tmp_path / "prediction_manifest.json").write_text(
        json.dumps({"train_case_ids": [100], "test_case_ids": [73]}), encoding="utf-8"
    )
    identity = identity_vertices(257, device=torch.device("cpu"))
    for arm in ("actual", "blank"):
        np.savez_compressed(
            tmp_path / f"73_{arm}_safe_q1.npz",
            vertices=identity.numpy(),
            boundary_reference=identity.numpy(),
            post_affine_matrix=np.eye(2, dtype=np.float32),
            post_affine_offset=np.zeros(2, dtype=np.float32),
        )
    monkeypatch.setattr(probe, "load_checkpoint",
                        lambda *args: (_ for _ in ()).throw(AssertionError("model read")))
    monkeypatch.setattr(probe, "load_case_inputs",
                        lambda *args: (_ for _ in ()).throw(AssertionError("image read")))
    monkeypatch.setattr(probe, "load_case_teacher",
                        lambda *args: {"raw_teacher": identity})
    report = probe.score_sealed(root=tmp_path, output=tmp_path, device="cpu")
    assert report["cases"][0]["actual_to_DHR_full_vertex_rmse"] == 0
    assert report["cases"][0]["blank_to_DHR_full_vertex_rmse"] == 0
    assert report["cases"][0]["initial_affine_to_DHR_full_vertex_rmse"] == 0
