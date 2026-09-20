"""GPU-native blocked BHF near/far assembly.

This module mirrors the NumPy seven-point far quadrature and Duffy incident
face correction, but keeps the dense pair work on a selected torch device.
It is an acceleration/control implementation; the PV and atlas theorems are
unchanged from the reference discretization.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint


_BARY = np.asarray(
    [
        [1 / 3, 1 / 3, 1 / 3],
        [0.05971587178977, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.05971587178977, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.05971587178977],
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
    ],
    dtype=np.float64,
)
_REF_WEIGHTS = np.asarray(
    [0.1125, 0.066197076394253, 0.066197076394253, 0.066197076394253,
     0.0629695902724135, 0.0629695902724135, 0.0629695902724135],
    dtype=np.float64,
)


def _validate(source_vertices, image_vertices, faces, fz, variation):
    source = np.asarray(source_vertices, dtype=np.complex128)
    image = np.asarray(image_vertices, dtype=np.complex128)
    tri = np.array(faces, dtype=np.int64, copy=True)
    fz_array = np.asarray(fz, dtype=np.complex128)
    variation_array = np.asarray(variation, dtype=np.complex128)
    if source.ndim != 1 or image.shape != source.shape:
        raise ValueError("source_vertices and image_vertices must be matching 1D arrays")
    if tri.ndim != 2 or tri.shape[1] != 3 or np.any(tri < 0) or np.any(tri >= source.size):
        raise ValueError("faces must be an in-range (n_faces,3) array")
    if fz_array.shape != (tri.shape[0],) or variation_array.shape != fz_array.shape:
        raise ValueError("face data must have one value per face")
    return source, image, tri, fz_array, variation_array


def _complex_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Complex division written with real arithmetic for old torch CUDA builds."""
    dr, di = denominator.real, denominator.imag
    nr, ni = numerator.real, numerator.imag
    scale = dr * dr + di * di
    return torch.complex((nr * dr + ni * di) / scale, (ni * dr - nr * di) / scale)


def _kernel(source_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
    delta = source_image - target_image[:, None]
    singular = delta.real * delta.real + delta.imag * delta.imag <= 1e-28
    safe = torch.where(singular, torch.ones_like(delta), delta)
    regular = -_complex_div(target_image[:, None], source_image[None, :] - 1.0)
    regular = regular + _complex_div(target_image[:, None] - 1.0, source_image[None, :])
    result = _complex_div(torch.ones_like(safe), safe) + regular
    return torch.where(singular, regular, result)


def _far_assembly(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    weighted: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, int(target_image.numel()), block_size):
        stop = min(start + block_size, int(target_image.numel()))
        kernel = _kernel(source_image, target_image[start:stop])
        real = kernel.real.matmul(weighted.real) - kernel.imag.matmul(weighted.imag)
        imag = kernel.real.matmul(weighted.imag) + kernel.imag.matmul(weighted.real)
        outputs.append(torch.complex(-real / math.pi, -imag / math.pi))
    return torch.cat(outputs, dim=0)


def _pair_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (left[..., 0] * right[..., 0] - left[..., 1] * right[..., 1],
         left[..., 0] * right[..., 1] + left[..., 1] * right[..., 0]),
        dim=-1,
    )


def _pair_inv(value: torch.Tensor) -> torch.Tensor:
    scale = value[..., 0] * value[..., 0] + value[..., 1] * value[..., 1]
    return torch.stack((value[..., 0] / scale, -value[..., 1] / scale), dim=-1)


def _pair_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return _pair_mul(numerator, _pair_inv(denominator))


def _pair_kernel(source_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
    """Beurling kernel in two real channels, with no complex autograd ops."""
    delta = source_image[None, :, :] - target_image[:, None, :]
    singular = torch.sum(delta * delta, dim=-1) <= 1e-28
    one = torch.ones_like(delta)
    safe = torch.where(singular[..., None], one, delta)
    target = target_image[:, None, :]
    source = source_image[None, :, :]
    regular = -_pair_div(target, source - torch.as_tensor([1.0, 0.0], dtype=source.dtype, device=source.device))
    regular = regular + _pair_div(target - torch.as_tensor([1.0, 0.0], dtype=source.dtype, device=source.device), source)
    result = _pair_inv(safe) + regular
    return torch.where(singular[..., None], regular, result)


def _pair_far_assembly(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    weighted: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    def evaluate_block(source_arg: torch.Tensor, weighted_arg: torch.Tensor, block_target: torch.Tensor) -> torch.Tensor:
        kernel = _pair_kernel(source_arg, block_target)
        real = kernel[..., 0].matmul(weighted_arg[:, 0]) - kernel[..., 1].matmul(weighted_arg[:, 1])
        imag = kernel[..., 0].matmul(weighted_arg[:, 1]) + kernel[..., 1].matmul(weighted_arg[:, 0])
        return torch.stack((-real / math.pi, -imag / math.pi), dim=-1)

    outputs = []
    for start in range(0, int(target_image.shape[0]), block_size):
        stop = min(start + block_size, int(target_image.shape[0]))
        block_target = target_image[start:stop]
        # Recompute the O(source*block) kernel during backward instead of
        # retaining every dense block graph.  This is essential at 128²+
        # resolutions and mirrors checkpointed adjoint implementations.
        if torch.is_grad_enabled() and (
            source_image.requires_grad or block_target.requires_grad or weighted.requires_grad
        ):
            if int(torch.__version__.split(".")[0]) >= 2:
                outputs.append(
                    _torch_checkpoint(
                        evaluate_block, source_image, weighted, block_target,
                        use_reentrant=False,
                    )
                )
            else:
                outputs.append(_torch_checkpoint(evaluate_block, source_image, weighted, block_target))
        else:
            outputs.append(evaluate_block(source_image, weighted, block_target))
    return torch.cat(outputs, dim=0)


def _pair_duffy_and_regular_pairs(
    source_triangles: torch.Tensor,
    image_triangles: torch.Tensor,
    target_images: torch.Tensor,
    fz: torch.Tensor,
    variation: torch.Tensor,
    *,
    order: int,
    block_size: int,
) -> torch.Tensor:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    real_dtype = source_triangles.dtype
    s = torch.as_tensor(nodes, dtype=real_dtype, device=source_triangles.device)
    q = torch.as_tensor(weights, dtype=real_dtype, device=source_triangles.device)
    total = torch.zeros_like(target_images)
    one = torch.as_tensor([1.0, 0.0], dtype=real_dtype, device=source_triangles.device)
    for start in range(0, int(target_images.shape[0]), block_size):
        stop = min(start + block_size, int(target_images.shape[0]))
        st = source_triangles[start:stop]
        it = image_triangles[start:stop]
        eval_image = target_images[start:stop]
        p0, p1, p2 = st[:, 0], st[:, 1], st[:, 2]
        f0, f1, f2 = it[:, 0], it[:, 1], it[:, 2]
        p01, p02 = p1 - p0, p2 - p0
        f01, f02 = f1 - f0, f2 - f0
        jacobian = torch.abs(p01[:, 0] * p02[:, 1] - p01[:, 1] * p02[:, 0])
        duffy = torch.zeros_like(eval_image)
        regular_integral = torch.zeros_like(eval_image)
        for si, ws in zip(s, q):
            for ti, wt in zip(s, q):
                source_point = p0 + si * ((1.0 - ti) * p01 + ti * p02)
                image_point = f0 + si * ((1.0 - ti) * f01 + ti * f02)
                delta = image_point - eval_image
                regular = -_pair_div(eval_image, image_point - one) + _pair_div(eval_image - one, image_point)
                singular = torch.sum(delta * delta, dim=-1) <= 1e-28
                safe_delta = torch.where(singular[:, None], torch.ones_like(delta), delta)
                kernel = _pair_inv(safe_delta) + regular
                duffy = duffy + ws * wt * jacobian[:, None] * si * kernel
        bary = torch.as_tensor(_BARY, dtype=real_dtype, device=source_triangles.device)
        rw = torch.as_tensor(_REF_WEIGHTS, dtype=real_dtype, device=source_triangles.device)
        source_q = torch.sum(st[:, None, :, :] * bary[None, :, :, None], dim=2)
        image_q = torch.sum(it[:, None, :, :] * bary[None, :, :, None], dim=2)
        for quad_index in range(bary.shape[0]):
            image_point = image_q[:, quad_index]
            delta = image_point - eval_image
            regular = -_pair_div(eval_image, image_point - one) + _pair_div(eval_image - one, image_point)
            singular = torch.sum(delta * delta, dim=-1) <= 1e-28
            safe_delta = torch.where(singular[:, None], torch.ones_like(delta), delta)
            kernel = _pair_inv(safe_delta) + regular
            regular_integral = regular_integral + jacobian[:, None] * rw[quad_index] * kernel
        fz_squared = _pair_mul(fz[start:stop], fz[start:stop])
        factor = -_pair_mul(variation[start:stop], fz_squared) / math.pi
        total[start:stop] = _pair_mul(factor, duffy - regular_integral)
    return total


def bhf_near_far_apply_torch_differentiable_real_pair(
    source_vertices: torch.Tensor,
    image_vertices: torch.Tensor,
    faces: torch.Tensor | np.ndarray,
    fz_face: torch.Tensor,
    variation_face: torch.Tensor,
    *,
    near_order: int = 16,
    target_block_size: int = 128,
    pair_block_size: int = 2048,
) -> torch.Tensor:
    """Fully real-valued differentiable BHF assembly for legacy Torch CUDA.

    Every complex quantity is represented as a final dimension of length two;
    no ``torch.complex`` operation appears in the autograd path.  The return
    value has shape ``(n_vertices, 2)`` and stores real/imaginary velocity.
    """
    if source_vertices.ndim != 2 or source_vertices.shape[1] != 2 or image_vertices.shape != source_vertices.shape:
        raise ValueError("source_vertices and image_vertices must have shape (n,2)")
    if fz_face.ndim != 2 or fz_face.shape[1] != 2 or variation_face.shape != fz_face.shape:
        raise ValueError("face data must have shape (n_faces,2)")
    if source_vertices.dtype not in (torch.float32, torch.float64):
        raise ValueError("real-pair inputs must be float tensors")
    if near_order < 2 or target_block_size < 1 or pair_block_size < 1:
        raise ValueError("quadrature order and block sizes must be positive")
    device = image_vertices.device
    faces_t = torch.as_tensor(faces, dtype=torch.long, device=device)
    source = source_vertices.to(device=device)
    image = image_vertices
    triangles = source[faces_t]
    image_triangles = image[faces_t]
    bary = torch.as_tensor(_BARY, dtype=source.dtype, device=device)
    ref_weights = torch.as_tensor(_REF_WEIGHTS, dtype=source.dtype, device=device)
    source_q = torch.sum(triangles[:, None, :, :] * bary[None, :, :, None], dim=2).reshape(-1, 2)
    image_q = torch.sum(image_triangles[:, None, :, :] * bary[None, :, :, None], dim=2).reshape(-1, 2)
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    jacobian = torch.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
    weights_q = jacobian[:, None] * ref_weights[None, :]
    weighted_face = _pair_mul(variation_face, _pair_mul(fz_face, fz_face))
    weighted = (weighted_face[:, None, :] * weights_q[:, :, None]).reshape(-1, 2)
    result = _pair_far_assembly(image_q, image, weighted, block_size=target_block_size)

    face_ids = torch.arange(faces_t.shape[0], device=device).repeat_interleave(3)
    local = torch.arange(3, device=device).repeat(faces_t.shape[0])
    pair_faces = faces_t[face_ids]
    pair_index = torch.arange(pair_faces.shape[0], device=device)
    next_local = (local + 1) % 3
    next_next_local = (local + 2) % 3
    pair_vertex_indices = torch.stack(
        (pair_faces[pair_index, local], pair_faces[pair_index, next_local], pair_faces[pair_index, next_next_local]),
        dim=1,
    )
    pair_source = source[pair_vertex_indices]
    pair_image = image[pair_vertex_indices]
    pair_target = image[pair_faces[pair_index, local]]
    pair_fz = fz_face[face_ids]
    pair_variation = variation_face[face_ids]
    corrections = _pair_duffy_and_regular_pairs(
        pair_source, pair_image, pair_target, pair_fz, pair_variation,
        order=near_order, block_size=pair_block_size,
    )
    target_indices = pair_faces[pair_index, local]
    result_real = result[:, 0].clone().index_add(0, target_indices, corrections[:, 0])
    result_imag = result[:, 1].clone().index_add(0, target_indices, corrections[:, 1])
    return torch.stack((result_real, result_imag), dim=1)


def _duffy_and_regular_pairs(
    source_triangles: torch.Tensor,
    image_triangles: torch.Tensor,
    target_images: torch.Tensor,
    fz: torch.Tensor,
    variation: torch.Tensor,
    *,
    order: int,
    block_size: int,
) -> torch.Tensor:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    s = torch.as_tensor(nodes, dtype=source_triangles.real.dtype, device=source_triangles.device)
    q = torch.as_tensor(weights, dtype=source_triangles.real.dtype, device=source_triangles.device)
    total = torch.zeros_like(target_images)
    for start in range(0, int(target_images.numel()), block_size):
        stop = min(start + block_size, int(target_images.numel()))
        st = source_triangles[start:stop]
        it = image_triangles[start:stop]
        eval_image = target_images[start:stop]
        p0, p1, p2 = st[:, 0], st[:, 1], st[:, 2]
        f0, f1, f2 = it[:, 0], it[:, 1], it[:, 2]
        jacobian = torch.abs(torch.imag((p1 - p0) * torch.conj(p2 - p0)))
        duffy = torch.zeros_like(eval_image)
        regular_integral = torch.zeros_like(eval_image)
        for si, ws in zip(s, q):
            for ti, wt in zip(s, q):
                source_point = p0 + si * ((1.0 - ti) * (p1 - p0) + ti * (p2 - p0))
                image_point = f0 + si * ((1.0 - ti) * (f1 - f0) + ti * (f2 - f0))
                delta = image_point - eval_image
                regular = -_complex_div(eval_image, image_point - 1.0) + _complex_div(eval_image - 1.0, image_point)
                singular = delta.real * delta.real + delta.imag * delta.imag <= 1e-28
                delta = _complex_div(torch.ones_like(delta), torch.where(singular, torch.ones_like(delta), delta)) + regular
                duffy = duffy + ws * wt * jacobian * si * delta
        bary = torch.as_tensor(_BARY, dtype=source_triangles.real.dtype, device=source_triangles.device)
        rw = torch.as_tensor(_REF_WEIGHTS, dtype=source_triangles.real.dtype, device=source_triangles.device)
        bary_complex = bary.to(source_triangles.dtype)
        source_q = (st[:, None, :] * bary_complex[None, :, :]).sum(dim=2)
        image_q = (it[:, None, :] * bary_complex[None, :, :]).sum(dim=2)
        for quad_index in range(bary.shape[0]):
            image_point = image_q[:, quad_index]
            delta = image_point - eval_image
            regular = -_complex_div(eval_image, image_point - 1.0) + _complex_div(eval_image - 1.0, image_point)
            singular = delta.real * delta.real + delta.imag * delta.imag <= 1e-28
            kernel = _complex_div(torch.ones_like(delta), torch.where(singular, torch.ones_like(delta), delta)) + regular
            regular_integral = regular_integral + jacobian * rw[quad_index] * kernel
        factor = -variation[start:stop] * (fz[start:stop] * fz[start:stop]) / math.pi
        total[start:stop] = factor * (duffy - regular_integral)
    return total


def bhf_near_far_apply_torch(
    source_vertices: np.ndarray,
    image_vertices: np.ndarray,
    faces: np.ndarray,
    fz_face: np.ndarray,
    variation_face: np.ndarray,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex64,
    near_order: int = 16,
    target_block_size: int = 128,
    pair_block_size: int = 2048,
) -> np.ndarray:
    """Apply the BHF vertex near/far assembly on a torch device."""

    source, image, tri, fz, variation = _validate(
        source_vertices, image_vertices, faces, fz_face, variation_face
    )
    if near_order < 2 or target_block_size < 1 or pair_block_size < 1:
        raise ValueError("quadrature order and block sizes must be positive")
    target_device = torch.device(device)
    source_t = torch.as_tensor(source, dtype=dtype, device=target_device)
    image_t = torch.as_tensor(image, dtype=dtype, device=target_device)
    faces_t = torch.as_tensor(tri, dtype=torch.long, device=target_device)
    fz_t = torch.as_tensor(fz, dtype=dtype, device=target_device)
    variation_t = torch.as_tensor(variation, dtype=dtype, device=target_device)
    triangles = source_t[faces_t]
    image_triangles = image_t[faces_t]
    bary = torch.as_tensor(_BARY, dtype=source_t.real.dtype, device=target_device)
    ref_weights = torch.as_tensor(_REF_WEIGHTS, dtype=source_t.real.dtype, device=target_device)
    bary_complex = bary.to(dtype)
    source_q = (triangles[:, None, :] * bary_complex[None, :, :]).sum(dim=2).reshape(-1)
    image_q = (image_triangles[:, None, :] * bary_complex[None, :, :]).sum(dim=2).reshape(-1)
    jacobian = torch.abs(torch.imag((triangles[:, 1] - triangles[:, 0]) * torch.conj(triangles[:, 2] - triangles[:, 0])))
    weights_q = (jacobian[:, None] * ref_weights[None, :]).reshape(-1).to(dtype)
    weighted_face = (variation_t * (fz_t * fz_t))[:, None].expand(-1, bary.shape[0])
    weighted = weighted_face.contiguous().reshape(-1) * weights_q
    result = _far_assembly(image_q, image_t, weighted, block_size=target_block_size)

    face_ids = torch.arange(tri.shape[0], device=target_device).repeat_interleave(3)
    local = torch.arange(3, device=target_device).repeat(tri.shape[0])
    pair_faces = faces_t[face_ids]
    next_local = (local + 1) % 3
    next_next_local = (local + 2) % 3
    pair_source = torch.stack(
        (source_t[pair_faces[torch.arange(pair_faces.shape[0], device=target_device), local]],
         source_t[pair_faces[torch.arange(pair_faces.shape[0], device=target_device), next_local]],
         source_t[pair_faces[torch.arange(pair_faces.shape[0], device=target_device), next_next_local]]), dim=1
    )
    pair_image = torch.stack(
        (image_t[pair_faces[torch.arange(pair_faces.shape[0], device=target_device), local]],
         image_t[pair_faces[torch.arange(pair_faces.shape[0], device=target_device), next_local]],
         image_t[pair_faces[torch.arange(pair_faces.shape[0], device=target_device), next_next_local]]), dim=1
    )
    pair_target = image_t[pair_faces[torch.arange(pair_faces.shape[0], device=target_device), local]]
    pair_fz = fz_t[face_ids]
    pair_variation = variation_t[face_ids]
    corrections = _duffy_and_regular_pairs(
        pair_source, pair_image, pair_target, pair_fz, pair_variation,
        order=near_order, block_size=pair_block_size
    )
    result = result.clone()
    target_indices = pair_faces[torch.arange(pair_faces.shape[0], device=target_device), local]
    # Older CUDA torch builds do not implement complex index_add directly.
    result_real = result.real.clone()
    result_imag = result.imag.clone()
    result_real.index_add_(0, target_indices, corrections.real)
    result_imag.index_add_(0, target_indices, corrections.imag)
    result = torch.complex(result_real, result_imag)
    return result.detach().cpu().numpy().astype(np.complex128, copy=False)


def bhf_near_far_apply_torch_differentiable(
    source_vertices: torch.Tensor,
    image_vertices: torch.Tensor,
    faces: torch.Tensor | np.ndarray,
    fz_face: torch.Tensor,
    variation_face: torch.Tensor,
    *,
    near_order: int = 16,
    target_block_size: int = 128,
    pair_block_size: int = 2048,
    real_pair: bool = False,
) -> torch.Tensor:
    """Differentiable tensor-native BHF assembly.

    Unlike :func:`bhf_near_far_apply_torch`, this function never converts the
    image, derivative, or variation tensors through NumPy and therefore keeps
    the PyTorch autograd graph.  It is intended for VJP experiments; callers
    still need an implicit treatment for a multi-step nonlinear flow.
    """

    if not isinstance(source_vertices, torch.Tensor) or not isinstance(image_vertices, torch.Tensor):
        raise ValueError("source_vertices and image_vertices must be torch tensors")
    if real_pair:
        if source_vertices.ndim != 2 or source_vertices.shape[1] != 2 or image_vertices.shape != source_vertices.shape:
            raise ValueError("real-pair vertices must have shape (n,2)")
    elif source_vertices.ndim != 1 or image_vertices.shape != source_vertices.shape:
        raise ValueError("source_vertices and image_vertices must be matching 1D tensors")
    if fz_face.ndim != 1 or variation_face.shape != fz_face.shape:
        raise ValueError("fz_face and variation_face must be matching 1D tensors")
    if near_order < 2 or target_block_size < 1 or pair_block_size < 1:
        raise ValueError("quadrature order and block sizes must be positive")
    if faces is None:
        raise ValueError("faces are required")
    target_device = image_vertices.device
    dtype = torch.complex64 if image_vertices.dtype == torch.float32 else torch.complex128
    if (real_pair and image_vertices.dtype not in (torch.float32, torch.float64)) or (
        not real_pair and image_vertices.dtype not in (torch.complex64, torch.complex128)
    ):
        raise ValueError("image_vertices must be complex or a float real-pair tensor")
    faces_t = torch.as_tensor(faces, dtype=torch.long, device=target_device)
    if faces_t.ndim != 2 or faces_t.shape[1] != 3:
        raise ValueError("faces must have shape (n_faces,3)")
    if real_pair:
        source_pair = source_vertices.to(device=target_device)
        image_pair = image_vertices
        source = torch.complex(source_pair[:, 0], source_pair[:, 1]).to(dtype)
        image = torch.complex(image_pair[:, 0], image_pair[:, 1]).to(dtype)
    else:
        source = source_vertices.to(device=target_device, dtype=dtype)
        image = image_vertices
    fz = fz_face.to(device=target_device, dtype=dtype)
    variation = variation_face.to(device=target_device, dtype=dtype)
    if real_pair:
        source_tri_pair = source_pair[faces_t]
        image_tri_pair = image_pair[faces_t]
        triangles = torch.complex(source_tri_pair[..., 0], source_tri_pair[..., 1]).to(dtype)
        image_triangles = torch.complex(image_tri_pair[..., 0], image_tri_pair[..., 1]).to(dtype)
    else:
        triangles = source[faces_t]
        image_triangles = image[faces_t]
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    bary = torch.as_tensor(_BARY, dtype=real_dtype, device=target_device)
    ref_weights = torch.as_tensor(_REF_WEIGHTS, dtype=real_dtype, device=target_device)
    bary_complex = bary.to(dtype)
    source_q = (triangles[:, None, :] * bary_complex[None, :, :]).sum(dim=2).reshape(-1)
    image_q = (image_triangles[:, None, :] * bary_complex[None, :, :]).sum(dim=2).reshape(-1)
    jacobian = torch.abs(torch.imag((triangles[:, 1] - triangles[:, 0]) * torch.conj(triangles[:, 2] - triangles[:, 0])))
    weights_q = (jacobian[:, None] * ref_weights[None, :]).reshape(-1).to(dtype)
    weighted_face = (variation * (fz * fz))[:, None].expand(-1, bary.shape[0])
    weighted = weighted_face.contiguous().reshape(-1) * weights_q
    result = _far_assembly(image_q, image, weighted, block_size=target_block_size)

    face_ids = torch.arange(faces_t.shape[0], device=target_device).repeat_interleave(3)
    local = torch.arange(3, device=target_device).repeat(faces_t.shape[0])
    pair_faces = faces_t[face_ids]
    pair_indices = torch.arange(pair_faces.shape[0], device=target_device)
    next_local = (local + 1) % 3
    next_next_local = (local + 2) % 3
    pair_vertex_indices = torch.stack(
        (pair_faces[pair_indices, local], pair_faces[pair_indices, next_local], pair_faces[pair_indices, next_next_local]), dim=1
    )
    if real_pair:
        pair_source_pair = source_pair[pair_vertex_indices]
        pair_image_pair = image_pair[pair_vertex_indices]
        pair_source = torch.complex(pair_source_pair[..., 0], pair_source_pair[..., 1]).to(dtype)
        pair_image = torch.complex(pair_image_pair[..., 0], pair_image_pair[..., 1]).to(dtype)
        pair_target = torch.complex(
            image_pair[pair_faces[pair_indices, local], 0],
            image_pair[pair_faces[pair_indices, local], 1],
        ).to(dtype)
    else:
        pair_source = source[pair_vertex_indices]
        pair_image = image[pair_vertex_indices]
        pair_target = image[pair_faces[pair_indices, local]]
    pair_fz = torch.complex(fz.real[face_ids], fz.imag[face_ids])
    pair_variation = torch.complex(variation.real[face_ids], variation.imag[face_ids])
    corrections = _duffy_and_regular_pairs(
        pair_source, pair_image, pair_target, pair_fz, pair_variation,
        order=near_order, block_size=pair_block_size,
    )
    target_indices = pair_faces[pair_indices, local]
    result_real = result.real.clone()
    result_imag = result.imag.clone()
    result_real.index_add_(0, target_indices, corrections.real)
    result_imag.index_add_(0, target_indices, corrections.imag)
    return torch.complex(result_real, result_imag)
