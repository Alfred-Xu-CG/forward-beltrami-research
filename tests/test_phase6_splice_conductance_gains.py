"""The Route C factorial must change only the intended gain coordinates."""

import sys

import torch

from phase6_splice_conductance_gains import main


def test_old_and_frequency64_gain_splice(tmp_path, monkeypatch):
    base_path = tmp_path / "base18.pt"
    trained_path = tmp_path / "trained21.pt"
    base_old_path = tmp_path / "base_old_trained64.pt"
    trained_old_path = tmp_path / "trained_old_neutral64.pt"
    base = torch.arange(18, dtype=torch.float64) / 100
    trained = torch.arange(21, dtype=torch.float64) / 50
    torch.save({"raw_gains": base}, base_path)
    torch.save({"raw_gains": trained,
                "args": {"frequencies": "1,2,4,8,16,32,64"}}, trained_path)
    monkeypatch.setattr(sys, "argv", [
        "splice", "--base18", str(base_path), "--trained21", str(trained_path),
        "--base-old-trained64", str(base_old_path),
        "--trained-old-neutral64", str(trained_old_path)])
    main()
    base_old = torch.load(base_old_path, weights_only=False)["raw_gains"].reshape(3, 7)
    trained_old = torch.load(trained_old_path, weights_only=False)["raw_gains"].reshape(3, 7)
    torch.testing.assert_close(base_old[:, :6], base.reshape(3, 6))
    torch.testing.assert_close(base_old[:, 6], trained.reshape(3, 7)[:, 6])
    torch.testing.assert_close(trained_old[:, :6], trained.reshape(3, 7)[:, :6])
    torch.testing.assert_close(trained_old[:, 6], torch.zeros(3, dtype=torch.float64))
