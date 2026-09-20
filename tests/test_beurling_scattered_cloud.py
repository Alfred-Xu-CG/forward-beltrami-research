import numpy as np

from qcopt.experiments.beurling_scattered_highres_crosscheck import generate_cloud


def test_paired_near_cloud_has_requested_nonzero_source_target_offsets():
    cloud = generate_cloud(128, near=True, near_offset=1e-3)
    distances = np.abs(cloud["targets"] - cloud["points"])
    assert np.all(np.isfinite(distances))
    assert float(np.min(distances)) > 0.0
    assert np.allclose(distances, 1e-3, rtol=1e-12, atol=1e-12)
