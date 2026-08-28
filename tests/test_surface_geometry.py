import numpy as np

from qcopt.surface_atlas import SurfacePoint
from qcopt.surface_geometry import (
    ScreenedFieldSmoother,
    curvature_descriptors,
    interpolate_vertex_field,
    scatter_point_forces,
    surface_scalar_gradient,
    vertex_areas,
)
from qcopt.surface_mesh import SurfaceMesh


def _tetrahedron(scale: float = 1.0) -> SurfaceMesh:
    vertices = scale * np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return SurfaceMesh(vertices, faces)


def test_discrete_gauss_bonnet_and_scale_normalized_descriptors():
    first = _tetrahedron()
    second = _tetrahedron(scale=3.7)
    first_descriptor = curvature_descriptors(first)
    second_descriptor = curvature_descriptors(second)

    assert np.isclose(
        np.sum(first_descriptor.gaussian_raw * vertex_areas(first)),
        4.0 * np.pi,
        atol=1e-12,
    )
    assert np.allclose(first_descriptor.gaussian, second_descriptor.gaussian)
    assert np.allclose(first_descriptor.mean, second_descriptor.mean)
    assert np.all(np.isfinite(first_descriptor.mean_raw))


def test_face_scalar_gradient_recovers_linear_planar_field():
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    mesh = SurfaceMesh(vertices, np.array([[0, 1, 2]]))
    values = vertices[:, 0] + 2.0 * vertices[:, 1]
    gradient = surface_scalar_gradient(mesh, values)
    assert np.allclose(gradient[0], [1.0, 2.0, 0.0], atol=1e-14)


def test_screened_smoother_reuses_factorization_and_reduces_roughness():
    mesh = _tetrahedron()
    values = np.array([1.0, -1.0, 1.0, -1.0])
    smoother = ScreenedFieldSmoother(mesh, alpha=2.0)
    first = smoother.smooth(values)
    second = smoother.smooth(values)

    assert smoother.factorization_count == 1
    assert np.ptp(first) < np.ptp(values)
    assert np.allclose(first, second)

    vectors = np.tile(np.array([0.3, -0.7, 1.1]), (mesh.n_vertices, 1))
    tangent = smoother.smooth_tangent(vectors)
    assert np.max(np.abs(np.sum(tangent * mesh.vertex_normals, axis=1))) < 1e-12


def test_vertex_field_interpolation_and_force_scattering_are_barycentric():
    mesh = _tetrahedron()
    point = SurfacePoint(0, np.array([0.2, 0.3, 0.5]))
    scalar = np.arange(mesh.n_vertices, dtype=float)
    expected = point.barycentric @ scalar[mesh.faces[point.face]]
    assert np.isclose(interpolate_vertex_field(mesh, scalar, [point])[0], expected)

    force = np.array([[1.0, 2.0, -0.5]])
    scattered, weights = scatter_point_forces(mesh, [point], force)
    face_vertices = mesh.faces[point.face]
    assert np.allclose(weights[face_vertices], point.barycentric)
    assert np.allclose(scattered[face_vertices], point.barycentric[:, None] * force)
    untouched = np.setdiff1d(np.arange(mesh.n_vertices), face_vertices)
    assert np.allclose(scattered[untouched], 0.0)
