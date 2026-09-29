import numpy as np

from tools.digital_current_edge_condition import edge_condition


def _grid(side: int) -> np.ndarray:
    axis = np.linspace(0, 1, side)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx, yy), axis=-1)


def test_identity_centered_basis_is_identity():
    result = edge_condition(_grid(9))
    assert result["interior_vertices"] == 49
    assert result["nonpositive_centered_edge_determinants"] == 0
    assert result["condition_median"] == 1
    assert result["condition_gt_2_fraction"] == 0
    assert result["frobenius_distance_from_identity_median"] == 0
    assert result["horizontal_absolute_angle_deg_p95"] == 0
    assert result["vertical_absolute_angle_deg_p95"] == 0


def test_anisotropic_affine_basis_known_singular_values():
    vertices = _grid(9)
    vertices[..., 0] *= .1
    vertices[..., 1] *= 10
    result = edge_condition(vertices)
    np.testing.assert_allclose(result["sigma_min_min"], .1, rtol=1e-13)
    np.testing.assert_allclose(result["sigma_max_max"], 10, rtol=1e-13)
    np.testing.assert_allclose(result["condition_median"], 100, rtol=1e-13)
    np.testing.assert_allclose(result["horizontal_length_over_h_p05_p50_p95"],
                               [.1, .1, .1], rtol=1e-13)
    np.testing.assert_allclose(result["vertical_length_over_h_p05_p50_p95"],
                               [10, 10, 10], rtol=1e-13)
    assert result["condition_gt_10_fraction"] == 1
