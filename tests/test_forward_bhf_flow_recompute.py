import numpy as np
import torch

from qcopt.forward.bhf_flow_recompute import _face_fz_pair
from qcopt.forward.bhf_flow_recompute import recompute_bhf_flow_real_pair
from qcopt.forward.bhf_torch import bhf_near_far_apply_torch_differentiable_real_pair
from qcopt.mesh import structured_rectangle
from qcopt.injectivity import audit_injectivity


def _direct_flow(initial, source, faces, variation, steps):
    current = initial
    for _ in range(steps):
        fz = _face_fz_pair(source, current, faces)
        velocity = bhf_near_far_apply_torch_differentiable_real_pair(
            source, current, faces, fz, variation,
            near_order=4, target_block_size=16, pair_block_size=32,
        )
        current = current + 0.03 * velocity
    return current


def test_recompute_bhf_flow_matches_direct_unroll_and_vjp():
    mesh = structured_rectangle(3, 3)
    source = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float64)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long)
    variation = torch.zeros((mesh.n_faces, 2), dtype=torch.float64)
    variation[:, 0] = 0.015
    initial = (source * torch.tensor([0.94, 0.97], dtype=torch.float64)).requires_grad_(True)

    custom = recompute_bhf_flow_real_pair(
        initial, source, faces, variation, step_size=0.03, steps=2,
        near_order=4, target_block_size=16, pair_block_size=32,
    )
    custom_loss = torch.sum(custom * custom)
    custom_gradient = torch.autograd.grad(custom_loss, initial)[0]

    direct_initial = initial.detach().clone().requires_grad_(True)
    direct = _direct_flow(direct_initial, source, faces, variation, steps=2)
    direct_loss = torch.sum(direct * direct)
    direct_gradient = torch.autograd.grad(direct_loss, direct_initial)[0]

    assert torch.max(torch.abs(custom - direct.detach())).item() < 1e-12
    assert torch.max(torch.abs(custom_gradient - direct_gradient)).item() < 2e-8


def test_recompute_bhf_flow_adaptive_safe_step_keeps_positive_faces():
    mesh = structured_rectangle(4, 4)
    source = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float64)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long)
    variation = torch.zeros((mesh.n_faces, 2), dtype=torch.float64)
    variation[:, 0] = 0.08
    initial = source.clone().requires_grad_(True)
    output = recompute_bhf_flow_real_pair(
        initial, source, faces, variation, step_size=0.5, steps=3,
        near_order=4, target_block_size=16, pair_block_size=32,
        adaptive_safe=True, min_det_margin=1e-6, safety=0.95,
    )
    loss = torch.sum(output * output)
    loss.backward()
    report = audit_injectivity(mesh, output.detach().numpy(), rectangle=False)
    assert report.certified
    assert report.minimum_signed_area_ratio > 1e-6
    assert torch.isfinite(initial.grad).all()


def test_recompute_bhf_flow_fixed_active_root_vjp_matches_finite_difference():
    mesh = structured_rectangle(4, 4)
    source = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float64)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long)
    variation = torch.zeros((mesh.n_faces, 2), dtype=torch.float64)
    variation[:, 0] = 0.20
    initial = source.clone().requires_grad_(True)

    def evaluate(values):
        output = recompute_bhf_flow_real_pair(
            values, source, faces, variation, step_size=2.0, steps=1,
            near_order=4, target_block_size=16, pair_block_size=32,
            adaptive_safe=True, min_det_margin=1e-6, safety=0.95,
        )
        return torch.sum(output * output)

    loss = evaluate(initial)
    gradient = torch.autograd.grad(loss, initial)[0]
    direction = torch.as_tensor(np.random.default_rng(7).normal(size=initial.shape), dtype=torch.float64)
    eps = 1e-6
    finite = float((evaluate(initial.detach() + eps * direction) - evaluate(initial.detach() - eps * direction)) / (2.0 * eps))
    predicted = float(torch.sum(gradient * direction))
    assert abs(finite - predicted) / max(1.0, abs(finite)) < 3e-4
