"""Constant 8x8 joint fit for two local low-displacement packets."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense import physical_image_gradient


def joint_refine_two_patches(
    fixed: torch.Tensor, moving: torch.Tensor,
    origin: torch.Tensor, vector: torch.Tensor,
    *, steps: int = 4, width: float = 1 / 16,
    damping: float = .05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small damped Gauss-Newton; parameters are (v_x,v_y,o_x,o_y) twice."""
    if origin.shape != (len(fixed), 2, 2):
        raise ValueError("origin must have shape (batch,2,2)")
    if vector.shape != origin.shape:
        raise ValueError("vector must have the same shape as origin")
    batch = len(fixed)
    side = fixed.shape[-1]
    axis = torch.arange(
        side, device=fixed.device, dtype=fixed.dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    moving_gradient = physical_image_gradient(moving)

    def evaluate(
        p: torch.Tensor, v: torch.Tensor,
        jacobian: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        windows = []
        derivatives = []
        displacement = torch.zeros(
            batch, side, side, 2,
            device=fixed.device, dtype=fixed.dtype)
        for packet in range(2):
            tx = (xx[None] - p[:, packet, 0, None, None]) / width
            ty = (yy[None] - p[:, packet, 1, None, None]) / width
            inside_x = (tx >= 0) & (tx <= 1)
            inside_y = (ty >= 0) & (ty <= 1)
            wx = torch.where(
                inside_x, torch.sin(math.pi * tx).square(), 0)
            wy = torch.where(
                inside_y, torch.sin(math.pi * ty).square(), 0)
            window = wx * wy
            displacement += window[..., None] * v[:, packet, None, None]
            windows.append(window)
            if jacobian:
                dwx = torch.where(
                    inside_x, -math.pi / width *
                    torch.sin(2 * math.pi * tx), 0)
                dwy = torch.where(
                    inside_y, -math.pi / width *
                    torch.sin(2 * math.pi * ty), 0)
                derivatives.append((dwx * wy, wx * dwy))
        sample_grid = 2 * (identity + displacement) - 1
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
        columns = []
        for packet in range(2):
            directional = (
                gx * v[:, packet, 0, None, None] +
                gy * v[:, packet, 1, None, None])
            columns.extend((
                windows[packet] * gx,
                windows[packet] * gy,
                derivatives[packet][0] * directional,
                derivatives[packet][1] * directional,
            ))
        return loss, torch.stack(columns, dim=1), residual

    p, v = origin, vector
    for _ in range(steps):
        previous, jacobian, residual = evaluate(p, v, True)
        assert jacobian is not None
        normal = torch.einsum(
            "bihw,bjhw->bij", jacobian, jacobian)
        rhs = torch.einsum(
            "bihw,bhw->bi", jacobian, residual[:, 0])
        diagonal = normal.diagonal(dim1=1, dim2=2)
        normal = normal + torch.diag_embed(
            damping * diagonal + 1e-6)
        update = torch.linalg.solve(normal, rhs).reshape(batch, 2, 4)
        vector_step = update[:, :, :2].clamp(-.002, .002)
        origin_step = update[:, :, 2:].clamp(-.01, .01)
        best_p, best_v, best_loss = p, v, previous
        for alpha in (1., .5, .25):
            candidate_v = v + alpha * vector_step
            norm = torch.linalg.vector_norm(candidate_v, dim=-1)
            candidate_v = candidate_v * (
                .007 / norm.clamp_min(1e-12)).clamp_max(1)[..., None]
            candidate_p = (
                p + alpha * origin_step).clamp(.04, .9)
            loss, _, _ = evaluate(candidate_p, candidate_v, False)
            accepted = loss < best_loss
            best_p = torch.where(accepted[:, None, None], candidate_p, best_p)
            best_v = torch.where(accepted[:, None, None], candidate_v, best_v)
            best_loss = torch.minimum(best_loss, loss)
        p, v = best_p, best_v
    final_loss, _, _ = evaluate(p, v, False)
    return p, v, final_loss


def multistart_joint_refine_two_patches(
    fixed: torch.Tensor, moving: torch.Tensor,
    origin: torch.Tensor, vector: torch.Tensor,
    *, steps: int = 4, width: float = 1 / 16,
    damping: float = .05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy start plus local symmetric splits; choose actual image error."""
    starts = [(origin, vector)]
    anchor = origin[:, 0]
    half_vector = vector[:, 0] / 2
    for direction in ((.018, 0.), (0., .018), (.013, .013),
                      (.013, -.013)):
        delta = torch.tensor(
            direction, device=fixed.device,
            dtype=fixed.dtype)[None]
        split_origin = torch.stack((
            (anchor - delta).clamp(.04, .9),
            (anchor + delta).clamp(.04, .9)), dim=1)
        split_vector = torch.stack((half_vector, half_vector), dim=1)
        starts.append((split_origin, split_vector))
    best_loss = torch.full(
        (len(fixed),), math.inf,
        device=fixed.device, dtype=fixed.dtype)
    best_origin, best_vector = origin, vector
    for proposed_origin, proposed_vector in starts:
        result_origin, result_vector, loss = joint_refine_two_patches(
            fixed, moving, proposed_origin, proposed_vector,
            steps=steps, width=width, damping=damping)
        accepted = loss < best_loss
        best_origin = torch.where(
            accepted[:, None, None], result_origin, best_origin)
        best_vector = torch.where(
            accepted[:, None, None], result_vector, best_vector)
        best_loss = torch.minimum(best_loss, loss)
    return best_origin, best_vector, best_loss
