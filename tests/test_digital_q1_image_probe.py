"""A complete image-to-latent Q1 training step must be runnable."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.digital_q1_image_probe import run_probe
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def test_small_image_to_q1_training_probe_has_finite_vjp_and_safe_output() -> None:
    result = run_probe(final_side=17, image_side=32, width=4, steps=2, device="cpu")
    assert result["control_vertices"] == 17 * 17
    assert result["image_pixels"] == 32 * 32
    assert result["steps"] == 2
    assert math.isfinite(result["loss_start"]) and math.isfinite(result["loss_end"])
    assert result["finite_gradient_steps"] == 2
    assert result["nonzero_gradient_steps"] == 2
    assert result["nonpositive_corners"] == 0
    assert result["boundary_ordered_rectangle"]


def test_probe_can_export_and_certify_its_actual_float32_map() -> None:
    repository = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(dir=repository / "docs/digital_topology_wsi") as directory:
        exported = Path(directory) / "trained_map.npz"
        result = run_probe(
            final_side=17, image_side=32, width=4, steps=1,
            device="cpu", save_map=exported,
        )
        assert exported.is_file()
        saved = certify_q1_binary_map(exported)
        assert saved["valid"] and saved["corners"] == 4 * 16 * 16
        assert result["saved_map_valid"]
