import numpy as np
import torch

from qcopt.beltrami import face_jacobians
from qcopt.forward.bhf_flow_recompute import _smooth_safe_step_torch
from qcopt.forward.bhf_flow_recompute import recompute_bhf_flow_real_pair
from qcopt.mesh import structured_rectangle


def test_smooth_safe_step_respects_every_face_determinant_margin():
    mesh = structured_rectangle(4, 4)
    source = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float32)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long)
    current = source.clone()
    # A spatially varying displacement that would fold several faces at t=1.
    x, y = current[:, 0], current[:, 1]
    delta = torch.stack((1.7 * (x - 0.5) * (y + 0.2), -1.4 * (y - 0.5) * (x + 0.1)), dim=1)
    margin = 1e-4
    step = _smooth_safe_step_torch(
        current, delta, faces, requested=1.0,
        min_det_margin=margin, safety=0.95, smoothing=1e-7,
    )
    mapped = (current + step * delta).detach().cpu().numpy()
    determinants = np.linalg.det(face_jacobians(mesh, mapped))
    assert bool(torch.isfinite(step))
    assert float(step) > 0.0
    assert float(np.min(determinants)) >= margin - 2e-6


def test_smooth_safe_step_has_a_finite_gradient():
    mesh = structured_rectangle(3, 3)
    current = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float32)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long)
    current.requires_grad_(True)
    delta = torch.as_tensor(np.tile([0.08, -0.04], (mesh.n_vertices, 1)), dtype=torch.float32)
    step = _smooth_safe_step_torch(
        current, delta, faces, requested=0.7,
        min_det_margin=1e-5, safety=0.95, smoothing=1e-7,
    )
    (step * step).backward()
    assert torch.isfinite(step)
    assert current.grad is not None
    assert bool(torch.isfinite(current.grad).all())


def test_smooth_safe_flow_is_orientation_safe_and_differentiable():
    mesh = structured_rectangle(4, 4)
    source = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float32)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long)
    x, y = source[:, 0], source[:, 1]
    initial = torch.stack((0.92 * x + 0.05 * y, 0.03 * x + 0.94 * y), dim=1)
    initial.requires_grad_(True)
    centers = source[faces].mean(dim=1)
    amplitude = 0.12 * torch.exp(-((centers[:, 0] - 0.4) ** 2 + (centers[:, 1] - 0.6) ** 2) / 0.12)
    variation = torch.stack((amplitude, torch.zeros_like(amplitude)), dim=1)
    output = recompute_bhf_flow_real_pair(
        initial, source, faces, variation, step_size=2.0, steps=2,
        near_order=4, target_block_size=32, pair_block_size=256,
        adaptive_safe=True, smooth_safe=True, smoothing=1e-7,
        min_det_margin=1e-5, safety=0.95,
    )
    loss = torch.sum(output * output)
    loss.backward()
    determinants = np.linalg.det(face_jacobians(mesh, output.detach().cpu().numpy()))
    assert float(np.min(determinants)) >= 1e-5 - 2e-5
    assert bool(torch.isfinite(initial.grad).all())
