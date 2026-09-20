import numpy as np

from qcopt.forward.sphere import (
    mobius_scale_sphere,
    mobius_scale_velocity_sphere,
    sphere_face_orientation,
    uv_sphere_mesh,
)
from qcopt.forward.sphere_charts import (
    stereographic_forward,
    stereographic_inverse,
    stereographic_transition,
    stereographic_transition_velocity,
)


def test_uv_sphere_has_outward_oriented_faces_and_expected_size():
    vertices, faces = uv_sphere_mesh(32, 16)

    orientation = sphere_face_orientation(vertices, faces)

    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(faces) == 2 * 32 * (16 - 1)
    assert np.min(orientation) > 0.0


def test_stereographic_mobius_scale_is_sphere_bijective_and_pole_stable():
    vertices, faces = uv_sphere_mesh(64, 32)
    mapped = mobius_scale_sphere(vertices, scale=1.7)

    assert np.allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=1e-12)
    assert np.min(sphere_face_orientation(mapped, faces)) > 0.0
    north = np.argmax(vertices[:, 2])
    assert np.allclose(mapped[north], vertices[north], atol=1e-12)


def test_north_south_atlas_round_trip_and_mobius_transition():
    vertices, _ = uv_sphere_mesh(64, 32)
    regular = np.abs(vertices[:, 2]) < 0.92
    points = vertices[regular]
    north = stereographic_forward(points, pole="north")
    south = stereographic_forward(points, pole="south")
    assert np.max(np.abs(stereographic_inverse(north, pole="north") - points)) < 1e-12
    assert np.max(np.abs(stereographic_inverse(south, pole="south") - points)) < 1e-12
    assert np.max(np.abs(north * south - 1.0)) < 1e-12


def test_mobius_scale_velocity_matches_log_scale_finite_difference():
    vertices, _ = uv_sphere_mesh(64, 32)
    scale = 1.4
    velocity = mobius_scale_velocity_sphere(vertices, scale=scale)
    eps = 1e-6
    finite = (
        mobius_scale_sphere(vertices, scale=scale * np.exp(eps))
        - mobius_scale_sphere(vertices, scale=scale * np.exp(-eps))
    ) / (2.0 * eps)
    assert np.max(np.abs(velocity - finite)) < 1e-9


def test_chart_transition_pushes_mobius_velocity_by_chain_rule():
    vertices, _ = uv_sphere_mesh(128, 64)
    regular = np.abs(vertices[:, 2]) < 0.96
    points = vertices[regular]
    north = stereographic_forward(points, pole="north")
    south = stereographic_forward(points, pole="south")
    assert np.max(np.abs(stereographic_transition(north, from_pole="north", to_pole="south") - south)) < 1e-12

    # A positive real Mobius scale has north-chart velocity zdot=z.  After
    # transition, the same tangent must be wdot=-w.
    north_velocity = north.copy()
    south_velocity = stereographic_transition_velocity(
        north,
        north_velocity,
        from_pole="north",
        to_pole="south",
    )
    assert np.max(np.abs(south_velocity + south)) < 1e-12
    round_trip = stereographic_transition_velocity(
        south,
        south_velocity,
        from_pole="south",
        to_pole="north",
    )
    assert np.max(np.abs(round_trip - north_velocity)) < 1e-12
