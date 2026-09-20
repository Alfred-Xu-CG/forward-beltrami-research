import numpy as np

from qcopt.beltrami import face_beltrami, face_jacobians
from qcopt.forward.compatibility import (
    compatibility_matrix,
    compatibility_nullity,
    compatibility_residual,
)
from qcopt.mesh import structured_rectangle


def test_manufactured_pl_map_satisfies_shared_edge_compatibility():
    mesh = structured_rectangle(6, 5)
    uv = mesh.vertices.copy()
    uv[:, 0] += 0.08 * np.sin(2.0 * np.pi * uv[:, 1])
    mu = face_beltrami(mesh, uv)
    jac = face_jacobians(mesh, uv)
    fz = 0.5 * ((jac[:, 0, 0] + jac[:, 1, 1]) + 1j * (jac[:, 1, 0] - jac[:, 0, 1]))

    matrix = compatibility_matrix(mesh, mu)

    assert matrix.shape[1] == mesh.n_faces
    assert compatibility_residual(matrix, fz) < 1e-12


def test_random_facewise_coefficient_is_generically_not_compatible():
    mesh = structured_rectangle(4, 4)
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(mesh.n_faces, 2))
    mu = 0.7 * (raw[:, 0] + 1j * raw[:, 1]) / np.maximum(
        1.0, np.linalg.norm(raw, axis=1)
    )
    matrix = compatibility_matrix(mesh, mu).toarray()
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    assert singular_values[-1] > 1e-4


def test_compatibility_kernel_has_one_complex_scale_for_manufactured_map():
    mesh = structured_rectangle(5, 5)
    uv = mesh.vertices.copy()
    uv[:, 0] += 0.04 * np.sin(2.0 * np.pi * uv[:, 1])
    uv[:, 1] += 0.03 * np.sin(2.0 * np.pi * uv[:, 0])
    mu = face_beltrami(mesh, uv)
    matrix = compatibility_matrix(mesh, mu)
    assert compatibility_nullity(matrix, tolerance=1e-9) == 1
