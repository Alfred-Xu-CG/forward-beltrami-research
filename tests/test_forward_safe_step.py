import numpy as np

from qcopt.forward.safe_step import (
    certified_residual_step,
    certified_step_toward_target,
    maximum_safe_step,
    maximum_safe_step_with_active_face,
    maximum_safe_step_directional_derivative,
)
from qcopt.mesh import structured_rectangle
from qcopt.injectivity import audit_injectivity


def test_safe_step_stays_above_the_face_area_margin():
    mesh = structured_rectangle(8, 8)
    uv = mesh.vertices.copy()
    delta = np.zeros_like(uv)
    delta[-1] = np.asarray([-2.0, 0.0])

    bound = maximum_safe_step(mesh, uv, delta, min_det_margin=0.1)
    stepped, step = certified_residual_step(
        mesh, uv, delta, min_det_margin=0.1, safety=0.9
    )
    report = audit_injectivity(mesh, stepped)

    assert 0.0 < step < bound
    assert report.certified
    assert report.minimum_signed_area_ratio > 0.1


def test_safe_step_rejects_an_already_degenerate_map():
    mesh = structured_rectangle(2, 2)
    uv = mesh.vertices.copy()
    uv[3] = uv[0]
    delta = np.zeros_like(uv)

    try:
        maximum_safe_step(mesh, uv, delta)
    except ValueError as error:
        assert "margin" in str(error)
    else:
        raise AssertionError("degenerate map was accepted")


def test_safe_step_reports_active_face_for_piecewise_smooth_differentiation():
    mesh = structured_rectangle(8, 8)
    uv = mesh.vertices.copy()
    delta = np.zeros_like(uv)
    delta[:, 0] = 0.6 * uv[:, 0] * (1.0 - uv[:, 0])
    delta[:, 1] = -0.9 * uv[:, 1] * (1.0 - uv[:, 1])
    value, active = maximum_safe_step_with_active_face(
        mesh, uv, delta, min_det_margin=0.1
    )
    assert value > 0.0
    assert 0 <= active < mesh.n_faces
    assert np.isclose(value, maximum_safe_step(mesh, uv, delta, min_det_margin=0.1))


def test_active_root_directional_derivative_matches_finite_difference():
    mesh = structured_rectangle(8, 8)
    uv = mesh.vertices.copy()
    rng = np.random.default_rng(71)
    delta = 0.1 * rng.normal(size=uv.shape)
    d_uv = 0.01 * rng.normal(size=uv.shape)
    d_delta = 0.01 * rng.normal(size=uv.shape)
    analytic = maximum_safe_step_directional_derivative(
        mesh, uv, delta, d_uv, d_delta, min_det_margin=0.1
    )
    eps = 1e-6
    plus = maximum_safe_step(mesh, uv + eps * d_uv, delta + eps * d_delta, min_det_margin=0.1)
    minus = maximum_safe_step(mesh, uv - eps * d_uv, delta - eps * d_delta, min_det_margin=0.1)
    assert abs(analytic - (plus - minus) / (2.0 * eps)) < 1e-6


def test_target_step_caps_a_safe_update_at_one():
    mesh = structured_rectangle(4, 4)
    uv = mesh.vertices.copy()
    target = 1.1 * mesh.vertices
    updated, step = certified_step_toward_target(mesh, uv, target)
    assert step == 1.0
    np.testing.assert_allclose(updated, target)
