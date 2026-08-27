import numpy as np
import pytest

from qcopt.beltrami import face_beltrami
from qcopt.constraints import fixed_vertex_constraints, rectangle_sliding_constraints
from qcopt.lbs import assemble_lbs_stiffness, solve_lbs
from qcopt.mesh import structured_rectangle


def _boundary_constraints(mesh, target):
    boundary = mesh.boundary_loops[0]
    return fixed_vertex_constraints(mesh.n_vertices, boundary, target[boundary])


def test_lbs_identity_with_rectangle_sliding_boundary():
    mesh = structured_rectangle(3, 3)
    result = solve_lbs(
        mesh, np.zeros(mesh.n_faces, dtype=np.complex128), rectangle_sliding_constraints(mesh)
    )

    assert np.allclose(result.uv, mesh.vertices, atol=1e-11)
    assert result.primal_residual < 1e-10
    assert result.constraint_residual < 1e-12


def test_lbs_recovers_compatible_affine_map_with_fixed_boundary():
    mesh = structured_rectangle(4, 4)
    matrix = np.array([[1.4, 0.3], [-0.15, 0.75]])
    target = mesh.vertices @ matrix.T + np.array([0.2, -0.4])
    mu = face_beltrami(mesh, target)
    result = solve_lbs(mesh, mu, _boundary_constraints(mesh, target))

    assert np.max(np.abs(result.uv - target)) < 1e-10
    assert result.primal_residual < 1e-10


def test_lbs_stiffness_is_symmetric_and_rejects_non_qc_mu():
    mesh = structured_rectangle(2, 2)
    mu = np.full(mesh.n_faces, 0.2 + 0.1j)
    stiffness = assemble_lbs_stiffness(mesh, mu)
    assert np.allclose(stiffness.toarray(), stiffness.T.toarray(), atol=1e-13)

    with pytest.raises(ValueError, match="strictly below one"):
        assemble_lbs_stiffness(mesh, np.ones(mesh.n_faces, dtype=np.complex128))
