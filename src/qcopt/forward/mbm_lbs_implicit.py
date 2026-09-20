"""Implicit-autograd prototype for the structured MBM-LBS decoder.

The forward pass uses two sparse P1 conductivity solves. The backward pass uses
two transpose solves and contracts the facewise derivative of A(mu). This is a
CPU reference for gradient correctness; it is not yet a GPU implementation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

from .mbm_lbs import _conductivity_tensor, _structured_geometry


def mbm_lbs_torch_implicit(mu_real: torch.Tensor, mu_imag: torch.Tensor) -> torch.Tensor:
    """Return a real map tensor with shape (ny, nx, 2)."""
    return _MBMLBSFunction.apply(mu_real, mu_imag)


class _MBMLBSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu_real: torch.Tensor, mu_imag: torch.Tensor) -> torch.Tensor:
        if mu_real.ndim != 2 or mu_imag.shape != mu_real.shape:
            raise ValueError("mu_real and mu_imag must be matching 2D tensors")
        if mu_real.shape[0] < 3 or mu_real.shape[1] < 3:
            raise ValueError("mu_real and mu_imag must have both axes at least three")
        if mu_real.device != mu_imag.device:
            raise ValueError("mu_real and mu_imag must be on the same device")
        if not torch.is_floating_point(mu_real) or not torch.is_floating_point(mu_imag):
            raise ValueError("mu_real and mu_imag must be floating point tensors")
        real = mu_real.detach().cpu().numpy().astype(np.float64, copy=False)
        imag = mu_imag.detach().cpu().numpy().astype(np.float64, copy=False)
        coefficients = real + 1j * imag
        if not np.all(np.isfinite(coefficients)) or np.max(np.abs(coefficients)) >= 1.0:
            raise ValueError("Beltrami coefficients must be finite and satisfy |mu| < 1")

        ny, nx = coefficients.shape
        triangles, gradients, areas, face_mu = _structured_geometry(coefficients)
        face_A = np.stack([_conductivity_tensor(value) for value in face_mu])
        n_vertices = nx * ny
        matrix = lil_matrix((n_vertices, n_vertices), dtype=np.float64)
        for face_index, face in enumerate(triangles):
            local = areas[face_index] * (
                gradients[face_index] @ face_A[face_index] @ gradients[face_index].T
            )
            for i_local, i_global in enumerate(face):
                for j_local, j_global in enumerate(face):
                    matrix[i_global, j_global] += local[i_local, j_local]
        matrix = matrix.tocsc()

        left = np.arange(0, n_vertices, nx)
        right = np.arange(nx - 1, n_vertices, nx)
        bottom = np.arange(nx)
        top = np.arange((ny - 1) * nx, ny * nx)
        fixed_u = np.concatenate((left, right))
        values_u = np.r_[np.zeros(ny), np.ones(ny)]
        fixed_w = np.concatenate((bottom, top))
        values_w = np.r_[np.zeros(nx), np.ones(nx)]
        free_u = np.setdiff1d(np.arange(n_vertices), fixed_u)
        free_w = np.setdiff1d(np.arange(n_vertices), fixed_w)
        matrix_u = matrix[free_u][:, free_u].tocsc()
        matrix_w = matrix[free_w][:, free_w].tocsc()
        solve_u = factorized(matrix_u)
        solve_w = factorized(matrix_w)
        solve_u_transpose = factorized(matrix_u.T.tocsc())
        solve_w_transpose = factorized(matrix_w.T.tocsc())

        u = np.zeros(n_vertices, dtype=np.float64)
        u[fixed_u] = values_u
        u[free_u] = solve_u(-matrix[free_u][:, fixed_u] @ values_u)
        w = np.zeros(n_vertices, dtype=np.float64)
        w[fixed_w] = values_w
        w[free_w] = solve_w(-matrix[free_w][:, fixed_w] @ values_w)
        energy = float(w @ (matrix @ w))
        if not np.isfinite(energy) or energy <= 0.0:
            raise ValueError("complementary energy must be positive")
        modulus = 1.0 / energy
        v = modulus * w
        output = np.stack((u.reshape(ny, nx), v.reshape(ny, nx)), axis=-1)

        ctx.ny = ny
        ctx.nx = nx
        ctx.triangles = triangles
        ctx.gradients = gradients
        ctx.areas = areas
        ctx.face_mu = face_mu
        ctx.face_A = face_A
        ctx.matrix = matrix
        ctx.solve_u_transpose = solve_u_transpose
        ctx.solve_w_transpose = solve_w_transpose
        ctx.free_u = free_u
        ctx.free_w = free_w
        ctx.u = u
        ctx.w = w
        ctx.modulus = modulus
        ctx.device = mu_real.device
        ctx.real_dtype = mu_real.dtype
        ctx.imag_dtype = mu_imag.dtype
        return torch.as_tensor(output, dtype=mu_real.dtype, device=mu_real.device)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        cotangent = gradient.detach().cpu().numpy().astype(np.float64, copy=False)
        cotangent_u = cotangent[..., 0].reshape(-1)
        cotangent_v = cotangent[..., 1].reshape(-1)

        adjoint_u = np.zeros_like(ctx.u)
        adjoint_u[ctx.free_u] = ctx.solve_u_transpose(cotangent_u[ctx.free_u])
        adjoint_w = np.zeros_like(ctx.w)
        adjoint_w[ctx.free_w] = ctx.solve_w_transpose(
            ctx.modulus * cotangent_v[ctx.free_w]
        )
        modulus_cotangent = float(np.dot(cotangent_v, ctx.w))
        modulus_factor = -modulus_cotangent * ctx.modulus * ctx.modulus

        gradient_real = np.zeros((ctx.ny, ctx.nx), dtype=np.float64)
        gradient_imag = np.zeros((ctx.ny, ctx.nx), dtype=np.float64)
        for face_index, face in enumerate(ctx.triangles):
            G = ctx.gradients[face_index]
            area = ctx.areas[face_index]
            u_face = ctx.u[list(face)]
            w_face = ctx.w[list(face)]
            au = G.T @ adjoint_u[list(face)]
            bu = G.T @ u_face
            aw = G.T @ adjoint_w[list(face)]
            bw = G.T @ w_face
            local_A_gradient = (
                -area * _symmetric_outer(au, bu)
                -area * _symmetric_outer(aw, bw)
                +modulus_factor * area * np.outer(bw, bw)
            )
            dA_da, dA_db = _conductivity_derivatives(ctx.face_mu[face_index])
            grad_a = float(np.sum(local_A_gradient * dA_da))
            grad_b = float(np.sum(local_A_gradient * dA_db))
            for vertex in face:
                row, col = divmod(int(vertex), ctx.nx)
                gradient_real[row, col] += grad_a / 3.0
                gradient_imag[row, col] += grad_b / 3.0

        return (
            torch.as_tensor(gradient_real, dtype=ctx.real_dtype, device=ctx.device),
            torch.as_tensor(gradient_imag, dtype=ctx.imag_dtype, device=ctx.device),
        )


def _symmetric_outer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.5 * (np.outer(left, right) + np.outer(right, left))


def _conductivity_derivatives(coefficient: complex):
    a, b = float(coefficient.real), float(coefficient.imag)
    denominator = 1.0 - a * a - b * b
    d_da_denominator = -2.0 * a
    d_db_denominator = -2.0 * b
    numerator = np.array(
        [
            [(1.0 - a) ** 2 + b * b, -2.0 * b],
            [-2.0 * b, (1.0 + a) ** 2 + b * b],
        ],
        dtype=np.float64,
    )
    d_da_numerator = np.array(
        [[-2.0 * (1.0 - a), 0.0], [0.0, 2.0 * (1.0 + a)]],
        dtype=np.float64,
    )
    d_db_numerator = np.array([[2.0 * b, -2.0], [-2.0, 2.0 * b]], dtype=np.float64)
    d_da = (
        d_da_numerator * denominator - numerator * d_da_denominator
    ) / (denominator * denominator)
    d_db = (
        d_db_numerator * denominator - numerator * d_db_denominator
    ) / (denominator * denominator)
    return d_da, d_db
