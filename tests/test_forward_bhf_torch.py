import numpy as np
import torch

from qcopt.beltrami import face_jacobians
from qcopt.forward.bhf_torch import bhf_near_far_apply_torch
from qcopt.forward.bhf_torch import bhf_near_far_apply_torch_differentiable
from qcopt.forward.bhf_torch import bhf_near_far_apply_torch_differentiable_real_pair
from qcopt.forward.bhf_variation import all_vertex_near_far_bhf_variation
from qcopt.mesh import structured_rectangle


def test_torch_bhf_near_far_matches_numpy_reference_on_small_mesh():
    mesh = structured_rectangle(6, 6)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    coefficient = 0.18 + 0.07j
    image = (1.0 - coefficient) * source + coefficient * np.conjugate(source)
    image_xy = np.column_stack((image.real, image.imag))
    jacobian = face_jacobians(mesh, image_xy)
    fz = 0.5 * ((jacobian[:, 0, 0] + jacobian[:, 1, 1]) + 1j * (jacobian[:, 1, 0] - jacobian[:, 0, 1]))
    triangles = source[mesh.faces]
    variation = 0.04 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.38 - 0.42j) ** 2 / 0.12)
    reference = all_vertex_near_far_bhf_variation(
        source, image, mesh.faces, fz, variation, near_order=4, far_block_size=32
    )
    candidate = bhf_near_far_apply_torch(
        source, image, mesh.faces, fz, variation, device="cpu", dtype=torch.complex128,
        near_order=4, target_block_size=32, pair_block_size=64
    )
    assert np.max(np.abs(reference - candidate)) < 1e-12


def test_torch_bhf_differentiable_image_vjp_matches_finite_difference():
    mesh = structured_rectangle(4, 4)
    source = torch.as_tensor(mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1], dtype=torch.complex128)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long)
    coefficient = 0.12 + 0.04j
    base = (1.0 - coefficient) * source + coefficient * torch.conj(source)
    image = base.clone().requires_grad_(True)
    triangles = source.detach()[faces]
    variation = 0.03 * torch.exp(-torch.abs(torch.mean(triangles, dim=1) - (0.4 + 0.45j)) ** 2 / 0.12)

    def evaluate(values):
        mapped = values[faces]
        ux = (mapped[:, 1].real - mapped[:, 0].real) / (1.0 / 3.0)
        uy = (mapped[:, 2].real - mapped[:, 0].real) / (1.0 / 3.0)
        vx = (mapped[:, 1].imag - mapped[:, 0].imag) / (1.0 / 3.0)
        vy = (mapped[:, 2].imag - mapped[:, 0].imag) / (1.0 / 3.0)
        fz = 0.5 * ((ux + vy) + 1j * (vx - uy))
        velocity = bhf_near_far_apply_torch_differentiable(
            source, values, faces, fz, variation, near_order=4,
            target_block_size=16, pair_block_size=32,
        )
        return torch.sum(torch.abs(velocity) ** 2)

    loss = evaluate(image)
    gradient = torch.autograd.grad(loss, image)[0]
    direction = torch.as_tensor(np.random.default_rng(18).normal(size=image.shape) + 1j * np.random.default_rng(19).normal(size=image.shape), dtype=torch.complex128)
    eps = 1e-6
    finite = float((evaluate(image.detach() + eps * direction) - evaluate(image.detach() - eps * direction)) / (2.0 * eps))
    predicted = float(torch.real(torch.sum(torch.conj(gradient) * direction)))
    assert abs(finite - predicted) / max(1.0, abs(finite)) < 2e-5


def test_torch_bhf_differentiable_real_pair_smoke():
    mesh = structured_rectangle(3, 3)
    source_pair = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float64)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long)
    image_pair = source_pair.clone().requires_grad_(True)
    source_tri = source_pair[faces]
    dx1 = source_tri[:, 1, 0] - source_tri[:, 0, 0]
    dy1 = source_tri[:, 1, 1] - source_tri[:, 0, 1]
    dx2 = source_tri[:, 2, 0] - source_tri[:, 0, 0]
    dy2 = source_tri[:, 2, 1] - source_tri[:, 0, 1]
    determinant = dx1 * dy2 - dx2 * dy1
    fz = torch.complex(torch.ones_like(determinant), torch.zeros_like(determinant))
    variation = torch.full_like(fz, 0.02)
    velocity = bhf_near_far_apply_torch_differentiable(
        source_pair, image_pair, faces, fz, variation, near_order=4,
        target_block_size=16, pair_block_size=32, real_pair=True,
    )
    loss = torch.sum(velocity.real * velocity.real + velocity.imag * velocity.imag)
    gradient = torch.autograd.grad(loss, image_pair)[0]
    assert torch.isfinite(velocity).all()
    assert torch.isfinite(gradient).all()


def test_torch_bhf_fully_real_pair_matches_complex_differentiable_path():
    mesh = structured_rectangle(3, 3)
    source_pair = torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.float64)
    faces = torch.as_tensor(np.array(mesh.faces, copy=True), dtype=torch.long)
    image_pair = (source_pair * torch.tensor([0.93, 0.97], dtype=torch.float64)).requires_grad_(True)
    source = torch.complex(source_pair[:, 0], source_pair[:, 1])
    image = torch.complex(image_pair[:, 0], image_pair[:, 1])
    source_tri = source_pair[faces]
    center = torch.complex(source_tri[:, :, 0].mean(dim=1), source_tri[:, :, 1].mean(dim=1))
    variation_complex = 0.02 * torch.exp(-torch.abs(center - (0.4 + 0.45j)) ** 2 / 0.12)
    variation_pair = torch.stack((variation_complex, torch.zeros_like(variation_complex)), dim=1)
    def compute_fz(values):
        source_tri_c = source[faces]
        image_tri_c = torch.complex(values[:, 0], values[:, 1])[faces]
        dx1 = source_tri_c[:, 1].real - source_tri_c[:, 0].real
        dy1 = source_tri_c[:, 1].imag - source_tri_c[:, 0].imag
        dx2 = source_tri_c[:, 2].real - source_tri_c[:, 0].real
        dy2 = source_tri_c[:, 2].imag - source_tri_c[:, 0].imag
        det = dx1 * dy2 - dx2 * dy1
        du1 = image_tri_c[:, 1].real - image_tri_c[:, 0].real
        du2 = image_tri_c[:, 2].real - image_tri_c[:, 0].real
        dv1 = image_tri_c[:, 1].imag - image_tri_c[:, 0].imag
        dv2 = image_tri_c[:, 2].imag - image_tri_c[:, 0].imag
        fz_complex = 0.5 * (((du1 * dy2 - du2 * dy1) / det + (-dv1 * dx2 + dv2 * dx1) / det)
                             + 1j * ((dv1 * dy2 - dv2 * dy1) / det - (-du1 * dx2 + du2 * dx1) / det))
        return fz_complex, torch.stack((fz_complex.real, fz_complex.imag), dim=1)

    fz_complex, fz_pair = compute_fz(image_pair)
    real_result = bhf_near_far_apply_torch_differentiable_real_pair(
        source_pair, image_pair, faces, fz_pair, variation_pair, near_order=4,
        target_block_size=16, pair_block_size=32,
    )
    complex_result = bhf_near_far_apply_torch_differentiable(
        source, image, faces, fz_complex, variation_complex, near_order=4,
        target_block_size=16, pair_block_size=32,
    )
    expected = torch.stack((complex_result.real, complex_result.imag), dim=1)
    assert torch.max(torch.abs(real_result - expected)).item() < 1e-11
    loss = torch.sum(real_result * real_result)
    gradient = torch.autograd.grad(loss, image_pair)[0]
    direction = torch.as_tensor(
        np.random.default_rng(33).normal(size=image_pair.shape), dtype=torch.float64
    )

    def evaluate(values):
        output = bhf_near_far_apply_torch_differentiable_real_pair(
            source_pair, values, faces, compute_fz(values)[1], variation_pair, near_order=4,
            target_block_size=16, pair_block_size=32,
        )
        return torch.sum(output * output)

    eps = 1e-6
    finite = float((evaluate(image_pair.detach() + eps * direction) - evaluate(image_pair.detach() - eps * direction)) / (2.0 * eps))
    predicted = float(torch.sum(gradient * direction))
    assert abs(finite - predicted) / max(1.0, abs(finite)) < 2e-5
