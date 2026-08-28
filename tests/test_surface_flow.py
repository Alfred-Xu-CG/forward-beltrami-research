import numpy as np

from qcopt.surface_atlas import SurfacePoint, transport_across_edge
from qcopt.surface_flow import (
    FlowHistory,
    FlowStep,
    IntrinsicFaceField,
    advect_points,
    evaluate_flow_field,
    intrinsic_field_lipschitz_bound,
    points_at_face_centroids,
    points_at_vertices,
    required_substeps_for_cfl,
)
from qcopt.surface_mesh import SurfaceMesh


def _torus(nu: int = 12, nv: int = 8) -> SurfaceMesh:
    vertices = []
    major, minor = 2.0, 0.6
    for i in range(nu):
        u = 2.0 * np.pi * i / nu
        for j in range(nv):
            v = 2.0 * np.pi * j / nv
            vertices.append(
                [
                    (major + minor * np.cos(v)) * np.cos(u),
                    (major + minor * np.cos(v)) * np.sin(u),
                    minor * np.sin(v),
                ]
            )
    index = lambda i, j: (i % nu) * nv + (j % nv)
    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i + 1, j + 1), index(i, j + 1)
            faces.extend(((a, b, c), (a, c, d)))
    return SurfaceMesh(np.asarray(vertices), np.asarray(faces))


def _swirl(mesh: SurfaceMesh) -> np.ndarray:
    field = np.column_stack((-mesh.vertices[:, 1], mesh.vertices[:, 0], np.zeros(mesh.n_vertices)))
    normal = np.sum(field * mesh.vertex_normals, axis=1)
    return field - normal[:, None] * mesh.vertex_normals


def test_zero_field_keeps_vertex_surface_points_exactly_fixed():
    mesh = _torus()
    points = points_at_vertices(mesh)
    result = advect_points(mesh, points, np.zeros((mesh.n_vertices, 3)))
    assert result.points == points
    assert result.transition_count == 0
    assert result.substeps == 1


def test_advection_changes_face_ids_without_an_optimized_chart_label():
    mesh = _torus()
    face = 0
    start = SurfacePoint(face, np.array([0.05, 0.475, 0.475]))
    triangle = mesh.vertices[mesh.faces[face]]
    edge_midpoint = 0.5 * (triangle[1] + triangle[2])
    direction = 1.4 * (edge_midpoint - start.position(mesh))
    field = np.tile(direction, (mesh.n_vertices, 1))
    result = advect_points(
        mesh,
        (start,),
        field,
        duration=1.0,
        fixed_substeps=1,
    )
    assert result.points[0].face != face
    assert result.transition_count >= 1


def test_cfl_rule_adds_substeps_for_large_velocity():
    mesh = _torus()
    points = points_at_vertices(mesh)[:5]
    result = advect_points(
        mesh,
        points,
        3.0 * _swirl(mesh),
        duration=1.0,
        max_step_fraction=0.1,
    )
    assert result.substeps > 1
    assert result.maximum_step_length <= 0.1 * mesh.minimum_edge_length * (1.0 + 1e-12)


def test_recorded_flow_has_small_forward_inverse_round_trip_error():
    mesh = _torus()
    # PL vertices are cone singularities without a unique tangent chart, so the
    # numerical inverse certificate samples face interiors.
    points = points_at_face_centroids(mesh)
    field = 0.4 * _swirl(mesh)
    step = FlowStep(field, duration=0.8, substeps=48)
    history = FlowHistory((step,))
    forward = history.apply(mesh, points)
    backward = history.inverse(mesh, forward.points)
    original_xyz = np.vstack([point.position(mesh) for point in points])
    recovered_xyz = np.vstack([point.position(mesh) for point in backward.points])
    relative_error = np.max(np.linalg.norm(recovered_xyz - original_xyz, axis=1)) / mesh.minimum_edge_length
    assert relative_error < 1e-3
    assert forward.transition_count > 0


def test_intrinsic_p2_field_is_transition_compatible_at_shared_edge_midpoint():
    mesh = _torus()
    field = IntrinsicFaceField.from_vertex_field(mesh, _swirl(mesh))
    face = 0
    local_edge = 0
    neighbor = int(mesh.face_neighbors[face, local_edge])
    neighbor_local_edge = int(np.flatnonzero(mesh.face_neighbors[neighbor] == face)[0])
    source_barycentric = np.zeros(3)
    source_barycentric[(local_edge + 1) % 3] = 0.5
    source_barycentric[(local_edge + 2) % 3] = 0.5
    target_barycentric = np.zeros(3)
    target_barycentric[(neighbor_local_edge + 1) % 3] = 0.5
    target_barycentric[(neighbor_local_edge + 2) % 3] = 0.5
    source_value = evaluate_flow_field(
        mesh, field, (SurfacePoint(face, source_barycentric),)
    )[0]
    target_value = evaluate_flow_field(
        mesh, field, (SurfacePoint(neighbor, target_barycentric),)
    )[0]
    _, transported = transport_across_edge(mesh, face, local_edge, source_value)
    assert np.allclose(transported, target_value, atol=1e-12)


def test_flow_fields_own_immutable_arrays_without_freezing_callers():
    mesh = _torus()
    edge = np.zeros((mesh.n_faces, 3, 3))
    bubble = np.zeros((mesh.n_faces, 3))
    field = IntrinsicFaceField(edge, bubble)
    edge[0, 0, 0] = 9.0
    bubble[0, 0] = 7.0

    vertex = np.zeros((mesh.n_vertices, 3))
    step = FlowStep(vertex, 1.0, 1)
    vertex[0, 0] = 5.0

    assert field.edge_values[0, 0, 0] == 0.0
    assert field.bubble_values[0, 0] == 0.0
    assert step.vertex_field[0, 0] == 0.0
    assert not field.edge_values.flags.writeable
    assert not field.bubble_values.flags.writeable
    assert not step.vertex_field.flags.writeable


def test_intrinsic_field_flow_remains_reversible_after_many_crossings():
    mesh = _torus()
    points = points_at_face_centroids(mesh)
    field = IntrinsicFaceField.from_vertex_field(mesh, 0.4 * _swirl(mesh))
    history = FlowHistory((FlowStep(field, duration=0.8, substeps=64),))
    forward = history.apply(mesh, points)
    backward = history.inverse(mesh, forward.points)
    original = np.vstack([point.position(mesh) for point in points])
    recovered = np.vstack([point.position(mesh) for point in backward.points])
    error = np.max(np.linalg.norm(recovered - original, axis=1)) / mesh.median_edge_length
    assert forward.transition_count > 0
    assert error < 2e-3


def test_intrinsic_field_substeps_are_derived_from_lipschitz_cfl():
    mesh = _torus()
    field = IntrinsicFaceField.from_vertex_field(mesh, 0.4 * _swirl(mesh))
    bound = intrinsic_field_lipschitz_bound(mesh, field)
    substeps = required_substeps_for_cfl(
        mesh, field, duration=0.8, cfl_limit=0.15
    )

    assert bound > 0.0
    assert bound * 0.8 / substeps <= 0.15
    if substeps > 1:
        assert bound * 0.8 / (substeps - 1) > 0.15
