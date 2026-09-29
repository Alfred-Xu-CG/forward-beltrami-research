from pathlib import Path

import numpy as np

from tools.digital_q1_f2_current_edge_benchmark import run


def test_f2_round_checkpoint_preserves_map_and_vjp(tmp_path: Path) -> None:
    ordinary = run(17, 2, 8, "cpu", 1, tmp_path / "ordinary")
    recomputed = run(17, 2, 8, "cpu", 1, tmp_path / "checkpoint", True)
    assert not ordinary["checkpoint_rounds"]
    assert recomputed["checkpoint_rounds"]
    for mode in ("fixed_h", "current_edge"):
        assert ordinary["arms"][mode]["certificate"]["valid"]
        assert recomputed["arms"][mode]["certificate"]["valid"]
        with np.load(tmp_path / f"ordinary_{mode}.npz") as first:
            with np.load(tmp_path / f"checkpoint_{mode}.npz") as second:
                np.testing.assert_array_equal(first["vertices"], second["vertices"])
                np.testing.assert_allclose(first["latent_vjp"], second["latent_vjp"],
                                           rtol=2e-6, atol=2e-8)
