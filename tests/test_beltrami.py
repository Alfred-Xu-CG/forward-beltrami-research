import numpy as np
import torch

from qcopt.beltrami import (
    face_beltrami,
    face_jacobians,
    qc_dilation,
    radial_squash,
)
from qcopt.mesh import structured_rectangle


def test_identity_and_diagonal_affine_beltrami_are_analytic():
    mesh = structured_rectangle(2, 2)
    assert np.allclose(face_beltrami(mesh, mesh.vertices), 0.0)

    a, b = 2.0, 0.5
    uv = np.column_stack((a * mesh.vertices[:, 0], b * mesh.vertices[:, 1]))
    mu = face_beltrami(mesh, uv)
    assert np.allclose(mu, (a - b) / (a + b))
    assert np.allclose(qc_dilation(mu), a / b)


def test_face_jacobians_and_signed_determinants_match_affine_map():
    mesh = structured_rectangle(1, 1)
    matrix = np.array([[1.5, 0.25], [-0.2, 0.8]])
    uv = mesh.vertices @ matrix.T + np.array([0.3, -0.1])
    jacobians = face_jacobians(mesh, uv)

    assert np.allclose(jacobians, matrix[None, :, :])
    assert np.allclose(np.linalg.det(jacobians), np.linalg.det(matrix))


def test_radial_squash_is_smooth_at_zero_and_strictly_bounded():
    raw = torch.tensor(
        [[0.0, 0.0], [0.3, -0.4], [10.0, 2.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    mu = radial_squash(raw, k_max=0.93)

    assert torch.equal(mu[0], torch.zeros(2, dtype=torch.double))
    assert torch.all(torch.linalg.vector_norm(mu, dim=-1) < 0.93)
    assert torch.autograd.gradcheck(
        lambda value: radial_squash(value, 0.93),
        (raw,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
