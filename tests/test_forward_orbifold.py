import numpy as np

from qcopt.forward.orbifold import polar_disk_mesh, radial_cone_inverse, radial_cone_map


def test_radial_cone_map_is_explicitly_invertible_for_both_angle_directions():
    vertices, faces = polar_disk_mesh(32, 64)
    for alpha in (0.65, 1.4):
        mapped = radial_cone_map(vertices, alpha)
        recovered = radial_cone_inverse(mapped, alpha)
        assert np.max(np.abs(recovered - vertices)) < 2e-14
        p0, p1, p2 = mapped[faces].transpose(1, 0, 2)
        determinant = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        assert np.min(determinant) > 0.0


def test_radial_cone_map_keeps_boundary_circle_fixed():
    angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    boundary = np.column_stack((np.cos(angles), np.sin(angles)))
    np.testing.assert_allclose(radial_cone_map(boundary, 0.7), boundary, atol=1e-14)
