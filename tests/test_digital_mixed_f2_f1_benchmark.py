from pathlib import Path

import numpy as np

from tools.digital_mixed_f2_f1_benchmark import run


def test_mixed_checkpoint_preserves_all_stages_and_vjp(tmp_path: Path) -> None:
    ordinary = run(17, 2, "cpu", tmp_path / "ordinary", 1, False)
    recomputed = run(17, 2, "cpu", tmp_path / "checkpoint", 1, True)
    assert not ordinary["checkpoint_rounds"]
    assert recomputed["checkpoint_rounds"]
    assert all(stage["certificate"]["valid"] for stage in ordinary["stages"])
    assert all(stage["certificate"]["valid"] for stage in recomputed["stages"])
    assert len(ordinary["stages"]) == len(recomputed["stages"]) == 4
    for name in ("f2_latent_vjp", "f1_latent_vjp"):
        with np.load(tmp_path / "ordinary_latents_and_vjp.npz") as first:
            with np.load(tmp_path / "checkpoint_latents_and_vjp.npz") as second:
                np.testing.assert_allclose(first[name], second[name], rtol=2e-6, atol=2e-8)
    for old, new in zip(ordinary["stages"], recomputed["stages"], strict=True):
        with np.load(tmp_path / old["saved_map"]) as first:
            with np.load(tmp_path / new["saved_map"]) as second:
                np.testing.assert_array_equal(first["vertices"], second["vertices"])


def test_width_four_mixed_passes_remain_valid(tmp_path: Path) -> None:
    report = run(17, 2, "cpu", tmp_path / "patch4", 1, True, .05, 4, 1.0)
    assert len(report["stages"]) == 4
    assert all(stage["certificate"]["valid"] for stage in report["stages"])
    assert report["nonzero_f2_latent_vjp"]
    assert report["nonzero_f1_latent_vjp"]
