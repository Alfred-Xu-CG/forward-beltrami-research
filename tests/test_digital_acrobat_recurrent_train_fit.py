"""The recurrent teacher fit must use the external case affine."""

import json

import torch

from tools import digital_acrobat_recurrent_train_fit as probe


def test_train_fit_uses_case_affine_not_network_internal_affine(tmp_path, monkeypatch) -> None:
    output = tmp_path / "model"
    output.mkdir()
    (output / "train_manifest.json").write_text(
        json.dumps({"train_case_ids": [1]}), encoding="utf-8",
    )
    grid = torch.tensor([[[[0., 0.], [1., 0.]],
                          [[0., 1.], [1., 1.]]]])
    external = torch.eye(2)[None] * .5
    internal = torch.eye(2)[None] * 2

    class FakeModel:
        def __call__(self, _fixed, _moving):
            return grid, internal, torch.zeros(1, 2)

    monkeypatch.setattr(probe, "_load_model", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(probe, "load_case", lambda *_args, **_kwargs: {
        "fixed": torch.zeros(1, 1, 2, 2),
        "prewarped": torch.zeros(1, 1, 2, 2),
        "matrix": external,
        "offset": torch.zeros(1, 2),
        "raw_teacher": grid * .5,
    })
    report = probe.evaluate(tmp_path, output, tmp_path / "fit.json", device="cpu")
    assert report["mean_case_rmse"] == 0
