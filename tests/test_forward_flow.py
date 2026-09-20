import numpy as np

from qcopt.forward.flow import safe_pl_flow
from qcopt.mesh import structured_rectangle


def test_safe_pl_flow_composes_multiple_certified_velocity_updates():
    mesh = structured_rectangle(16, 12)
    initial = mesh.vertices.copy()
    velocity = np.zeros_like(initial)
    velocity[:, 0] = 0.2 * np.sin(2.0 * np.pi * initial[:, 1])
    velocity[:, 1] = 0.1 * np.sin(2.0 * np.pi * initial[:, 0])

    result = safe_pl_flow(mesh, initial, [velocity] * 4, min_det_margin=0.2)

    assert len(result.maps) == 5
    assert len(result.steps) == 4
    assert all(report.certified for report in result.reports)
    assert min(result.steps) > 0.0
    assert min(report.minimum_signed_area_ratio for report in result.reports) > 0.2
