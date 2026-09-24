"""Local affine-window image fit with only 6x6 or 8x8 normal equations."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense import physical_image_gradient
from phase7_matched_low_patch import (
    matched_low_patch, multistart_refine_low_params,
)


def energy_window_origin(
    fixed: torch.Tensor, moving: torch.Tensor,
    *, width: float = 1 / 16,
) -> torch.Tensor:
    """Locate a compact window by local residual energy, without translation fit."""
    if fixed.shape != moving.shape or fixed.shape[-1] != 512:
        raise ValueError("requires matching 512-side image pairs")
    length = math.ceil(511 * width) + 1
    t = torch.arange(
        length, device=fixed.device,
        dtype=fixed.dtype) / (511 * width)
    window = torch.where(
        t <= 1, torch.sin(math.pi * t).square(), 0).square()
    field = (fixed - moving).square()
    score = F.conv2d(
        F.conv2d(field, window[None, None, None, :]),
        window[None, None, :, None])
    batch = len(fixed)
    peak = score.flatten(1).argmax(dim=1)
    side = score.shape[-1]
    row, column = peak // side, peak % side
    index = torch.arange(batch, device=fixed.device)

    def correction(vertical: bool) -> torch.Tensor:
        if vertical:
            minus = score[index, 0, (row - 1).clamp_min(0), column]
            middle = score[index, 0, row, column]
            plus = score[index, 0, (row + 1).clamp_max(side - 1), column]
        else:
            minus = score[index, 0, row, (column - 1).clamp_min(0)]
            middle = score[index, 0, row, column]
            plus = score[index, 0, row, (column + 1).clamp_max(side - 1)]
        curvature = minus - 2 * middle + plus
        return torch.where(
            curvature.abs() > 1e-12,
            .5 * (minus - plus) / curvature,
            torch.zeros_like(curvature)).clamp(-.5, .5)

    return torch.stack((
        (column.float() + correction(False)) / 511,
        (row.float() + correction(True)) / 511,
    ), dim=1)


def affine_window_map(
    origin: torch.Tensor, vector: torch.Tensor,
    matrix: torch.Tensor, side: int, dtype: torch.dtype,
    *, width: float = 1 / 16,
) -> torch.Tensor:
    """Return x + W_o(x) [v + M(x-o-(w/2,w/2))]."""
    axis = torch.arange(
        side, device=origin.device, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    delta = identity - origin.to(dtype)[:, None, None] - width / 2
    tx = (xx[None] - origin[:, 0, None, None].to(dtype)) / width
    ty = (yy[None] - origin[:, 1, None, None].to(dtype)) / width
    wx = torch.where((tx >= 0) & (tx <= 1),
                     torch.sin(math.pi * tx).square(), 0)
    wy = torch.where((ty >= 0) & (ty <= 1),
                     torch.sin(math.pi * ty).square(), 0)
    local = vector.to(dtype)[:, None, None] + torch.einsum(
        "bij,bhwj->bhwi", matrix.to(dtype), delta)
    return identity + (wx * wy)[..., None] * local


def fit_affine_window(
    fixed: torch.Tensor, moving: torch.Tensor,
    initial_origin: torch.Tensor,
    initial_vector: torch.Tensor,
    initial_matrix: torch.Tensor | None = None,
    *, steps: int = 8, moving_origin: bool = False,
    width: float = 1 / 16, damping: float = .001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit translation and local 2x2 matrix; optionally refine the origin."""
    batch = len(fixed)
    side = fixed.shape[-1]
    axis = torch.arange(
        side, device=fixed.device, dtype=fixed.dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    moving_gradient = physical_image_gradient(moving)

    def evaluate(
        p: torch.Tensor, v: torch.Tensor,
        matrix: torch.Tensor, jacobian: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        tx = (xx[None] - p[:, 0, None, None]) / width
        ty = (yy[None] - p[:, 1, None, None]) / width
        inside_x = (tx >= 0) & (tx <= 1)
        inside_y = (ty >= 0) & (ty <= 1)
        wx = torch.where(
            inside_x, torch.sin(math.pi * tx).square(), 0)
        wy = torch.where(
            inside_y, torch.sin(math.pi * ty).square(), 0)
        window = wx * wy
        delta = identity - p[:, None, None] - width / 2
        local = v[:, None, None] + torch.einsum(
            "bij,bhwj->bhwi", matrix, delta)
        sample_grid = 2 * (
            identity + window[..., None] * local) - 1
        predicted = F.grid_sample(
            moving, sample_grid, mode="bilinear",
            padding_mode="border", align_corners=True)
        residual = fixed - predicted
        loss = residual.square().mean(dim=(1, 2, 3))
        if not jacobian:
            return loss, None, residual
        gradient = F.grid_sample(
            moving_gradient, sample_grid,
            mode="bilinear", padding_mode="border",
            align_corners=True)
        gx, gy = gradient[:, 0], gradient[:, 1]
        dx, dy = delta[..., 0], delta[..., 1]
        columns = [
            window * gx, window * gy,
            window * dx * gx, window * dy * gx,
            window * dx * gy, window * dy * gy,
        ]
        if moving_origin:
            dwx = torch.where(
                inside_x, -math.pi / width *
                torch.sin(2 * math.pi * tx), 0)
            dwy = torch.where(
                inside_y, -math.pi / width *
                torch.sin(2 * math.pi * ty), 0)
            directional = (
                gx * local[..., 0] + gy * local[..., 1])
            columns.extend((
                dwx * wy * directional - window * (
                    matrix[:, 0, 0, None, None] * gx +
                    matrix[:, 1, 0, None, None] * gy),
                wx * dwy * directional - window * (
                    matrix[:, 0, 1, None, None] * gx +
                    matrix[:, 1, 1, None, None] * gy),
            ))
        return loss, torch.stack(columns, dim=1), residual

    p = initial_origin
    v = initial_vector
    matrix = (torch.zeros(
        batch, 2, 2, device=fixed.device,
        dtype=fixed.dtype)
        if initial_matrix is None else initial_matrix)
    for _ in range(steps):
        old_loss, jacobian, residual = evaluate(p, v, matrix, True)
        assert jacobian is not None
        normal = torch.einsum(
            "bihw,bjhw->bij", jacobian, jacobian)
        rhs = torch.einsum(
            "bihw,bhw->bi", jacobian, residual[:, 0])
        diagonal = normal.diagonal(dim1=1, dim2=2)
        normal = normal + torch.diag_embed(
            damping * diagonal + 1e-6)
        update = torch.linalg.solve(normal, rhs)
        vector_step = update[:, :2].clamp(-.002, .002)
        matrix_step = update[:, 2:6].reshape(batch, 2, 2).clamp(-.04, .04)
        origin_step = (update[:, 6:8].clamp(-.01, .01)
                       if moving_origin else None)
        best_p, best_v, best_matrix, best_loss = (
            p, v, matrix, old_loss)
        for alpha in (1., .5, .25):
            candidate_v = v + alpha * vector_step
            norm = torch.linalg.vector_norm(candidate_v, dim=-1)
            candidate_v = candidate_v * (
                .004 / norm.clamp_min(1e-12)).clamp_max(1)[:, None]
            candidate_matrix = matrix + alpha * matrix_step
            sigma = torch.linalg.matrix_norm(
                candidate_matrix, ord=2)
            candidate_matrix = candidate_matrix * (
                .12 / sigma.clamp_min(1e-12)).clamp_max(1)[:, None, None]
            candidate_p = ((p + alpha * origin_step).clamp(.04, .9)
                           if origin_step is not None else p)
            loss, _, _ = evaluate(
                candidate_p, candidate_v,
                candidate_matrix, False)
            accepted = loss < best_loss
            best_p = torch.where(accepted[:, None], candidate_p, best_p)
            best_v = torch.where(accepted[:, None], candidate_v, best_v)
            best_matrix = torch.where(
                accepted[:, None, None], candidate_matrix, best_matrix)
            best_loss = torch.minimum(best_loss, loss)
        p, v, matrix = best_p, best_v, best_matrix
    final_loss, _, _ = evaluate(p, v, matrix, False)
    return p, v, matrix, final_loss


def infer_affine_two_start(
    fixed: torch.Tensor, moving: torch.Tensor,
    *, steps: int = 8, damping: float = .001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard two-start local image fit, returning a 257² endpoint proposal."""
    translation_origin, translation_vector, _ = matched_low_patch(
        fixed, moving)
    translation_origin, translation_vector, _ = (
        multistart_refine_low_params(
            fixed, moving,
            translation_origin, translation_vector,
            steps=4, directions=4))
    energy_origin = energy_window_origin(fixed, moving)
    proposals = []
    for initial_origin in (translation_origin, energy_origin):
        p6, v6, m6, _ = fit_affine_window(
            fixed, moving, initial_origin,
            translation_vector, steps=steps,
            moving_origin=False, damping=damping)
        p8, v8, m8, loss = fit_affine_window(
            fixed, moving, p6, v6, m6,
            steps=steps, moving_origin=True,
            damping=damping)
        proposals.append((p8, v8, m8, loss))
    choose_second = proposals[1][3] < proposals[0][3]
    best = [torch.where(
        choose_second.reshape((-1,) + (1,) * (proposals[0][i].ndim - 1)),
        proposals[1][i], proposals[0][i]) for i in range(3)]
    return (best[0], best[1], affine_window_map(
        best[0], best[1], best[2], 257,
        torch.float64))
