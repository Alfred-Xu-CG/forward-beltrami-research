import numpy as np

from qcopt.surface_flow import FlowHistory, FlowStep, IntrinsicFaceField, points_at_face_centroids
from qcopt.surface_history_io import load_flow_history, save_flow_history
from qcopt.surface_mesh import SurfaceMesh


def _tetrahedron():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return SurfaceMesh(vertices, faces)


def test_flow_history_npz_round_trip_reproduces_mapped_points(tmp_path):
    mesh = _tetrahedron()
    ambient = np.tile(np.array([0.03, -0.02, 0.01]), (mesh.n_vertices, 1))
    intrinsic = IntrinsicFaceField.from_vertex_field(mesh, ambient)
    history = FlowHistory(
        (
            FlowStep(intrinsic, 0.4, 8),
            FlowStep(0.5 * ambient, 0.2, 4),
        )
    )
    path = tmp_path / "history.npz"
    save_flow_history(path, history)
    loaded = load_flow_history(path)

    points = points_at_face_centroids(mesh)
    expected = history.apply(mesh, points).points
    actual = loaded.apply(mesh, points).points
    expected_xyz = np.vstack([point.position(mesh) for point in expected])
    actual_xyz = np.vstack([point.position(mesh) for point in actual])
    assert np.allclose(actual_xyz, expected_xyz, atol=1e-12)
    assert len(loaded.steps) == 2
    assert isinstance(loaded.steps[0].vertex_field, IntrinsicFaceField)
