import numpy as np

from qcopt.forward.beurling_direct import direct_beurling_apply
from qcopt.forward.beurling_treecode import treecode_beurling_apply


def test_near_radius_correction_improves_paired_near_field():
    rng = np.random.default_rng(123)
    source = 0.1 + 0.8 * rng.random(512) + 1j * (0.1 + 0.8 * rng.random(512))
    targets = source + 1e-3 * np.exp(2j * np.pi * rng.random(source.size))
    values = rng.normal(size=source.size) + 1j * rng.normal(size=source.size)
    weights = np.ones(source.size) / source.size
    reference = direct_beurling_apply(source, values, weights, target_points=targets, block_size=128)
    baseline = treecode_beurling_apply(source, values, weights, target_points=targets, theta=0.5, order=4)
    corrected = treecode_beurling_apply(
        source, values, weights, target_points=targets, theta=0.5, order=4, near_radius=0.1
    )
    baseline_error = np.linalg.norm(baseline - reference)
    corrected_error = np.linalg.norm(corrected - reference)
    assert corrected_error < baseline_error
