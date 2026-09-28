"""Independent checks for saved backward-field geometry and landmark inversion."""

import numpy as np

from tools.digital_dhr_field_eval import canvas_xy, invert_q1_at_point, q1_corner_determinants


def test_canvas_uses_pixel_center_scaling_then_yx_padding():
    xy = canvas_xy(np.array([[10.0, 20.0]]), 0.1, [[1, 1], [0, 0]])
    np.testing.assert_allclose(xy, [[0.55, 2.55]])


def test_four_q1_corners_detect_fold_missed_by_top_corners():
    field = np.zeros((2, 2, 2), dtype=np.float64)
    field[0, 1, 1] = -2.0
    corners = q1_corner_determinants(field)
    np.testing.assert_allclose(corners[:, 0, 0], [1, 1, -1, -1])


def test_inverse_finds_exact_bilinear_root_with_small_residual():
    field = np.zeros((2, 2, 2), dtype=np.float64)
    field[0, 1, 1] = 0.2
    # At (s,t)=(0.25,0.75), F=(s + 0.2*s*t, t).
    solutions = invert_q1_at_point(field, np.array([0.2875, 0.75]))
    assert len(solutions) == 1
    np.testing.assert_allclose(solutions[0]["fixed_xy"], [0.25, 0.75], atol=1e-10)
    assert solutions[0]["residual_px"] < 1e-10
    np.testing.assert_allclose(solutions[0]["local_jacobian_det"], 1.15, atol=1e-10)


def test_inverse_reports_two_roots_for_noninjective_fold():
    field = np.zeros((2, 2, 3), dtype=np.float64)
    field[0, :, 2] = -2.0
    solutions = invert_q1_at_point(field, np.array([0.5, 0.4]))
    assert len(solutions) == 2
    np.testing.assert_allclose(sorted(s["fixed_xy"][0] for s in solutions), [0.5, 1.5], atol=1e-10)
    assert sorted(s["local_jacobian_det"] for s in solutions) == [-1.0, 1.0]


def test_inverse_reports_no_root_outside_mapped_domain():
    field = np.zeros((2, 2, 2), dtype=np.float64)
    assert invert_q1_at_point(field, np.array([2.0, 0.5])) == []
