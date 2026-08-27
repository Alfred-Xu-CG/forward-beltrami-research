import numpy as np
import pytest
import torch

from qcopt.energies import (
    amips_style_energy,
    lim_style_energy,
    log_det_barrier,
    symmetric_dirichlet_energy,
    torch_face_jacobians,
)
from qcopt.mesh import structured_rectangle


def _affine_jacobians(matrix):
    mesh = structured_rectangle(1, 1)
    uv = torch.as_tensor(mesh.vertices @ matrix.T, dtype=torch.double).requires_grad_()
    return mesh, uv, torch_face_jacobians(mesh, uv)


def test_identity_energy_normalizations_are_documented():
    _, _, jac = _affine_jacobians(np.eye(2))
    assert torch.allclose(log_det_barrier(jac), torch.tensor(0.0, dtype=torch.double))
    assert torch.allclose(lim_style_energy(jac), torch.tensor(0.0, dtype=torch.double))
    assert torch.allclose(
        symmetric_dirichlet_energy(jac), torch.tensor(4.0, dtype=torch.double)
    )
    assert torch.allclose(amips_style_energy(jac), torch.tensor(1.0, dtype=torch.double))


def test_barriers_grow_toward_positive_area_collapse():
    _, _, identity = _affine_jacobians(np.eye(2))
    _, _, compressed = _affine_jacobians(np.diag([1.0, 0.02]))
    assert log_det_barrier(compressed) > log_det_barrier(identity)
    assert lim_style_energy(compressed) > lim_style_energy(identity)
    assert symmetric_dirichlet_energy(compressed) > symmetric_dirichlet_energy(identity)
    assert amips_style_energy(compressed) > amips_style_energy(identity)


@pytest.mark.parametrize(
    "energy",
    [log_det_barrier, lim_style_energy, symmetric_dirichlet_energy, amips_style_energy],
)
def test_energies_have_finite_autograd_on_positive_faces(energy):
    _, uv, jac = _affine_jacobians(np.array([[1.2, 0.1], [-0.05, 0.8]]))
    value = energy(jac)
    value.backward()
    assert torch.isfinite(value)
    assert uv.grad is not None
    assert torch.all(torch.isfinite(uv.grad))


@pytest.mark.parametrize(
    "energy",
    [log_det_barrier, lim_style_energy, symmetric_dirichlet_energy, amips_style_energy],
)
def test_energies_return_infinity_for_nonpositive_orientation(energy):
    _, _, jac = _affine_jacobians(np.diag([1.0, -1.0]))
    assert torch.isinf(energy(jac))
