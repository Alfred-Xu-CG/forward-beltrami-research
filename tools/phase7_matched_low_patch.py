"""Matched local low-frequency patch readout in a mixed-scale image pair."""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    ResidualStaggeredPatchP1Layer,
    certify_p1_or_identity,
    exact_dyadic_p1_refine,
    physical_image_gradient,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable
from phase7_multiscale_fiber_reachability import minimum_jacobian


def _separable_valid(
    field: torch.Tensor, weights: torch.Tensor,
) -> torch.Tensor:
    kernel_x = weights[None, None, None, :]
    kernel_y = weights[None, None, :, None]
    return F.conv2d(
        F.conv2d(field, kernel_x), kernel_y)


def matched_low_patch(
    fixed: torch.Tensor, moving: torch.Tensor,
    *, width: float = 1 / 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return origin and 2-vector for a known-width sin² product window.

    Image origin candidates lie on pixel positions. A local quadratic fit
    gives subpixel origin; the vector is then regressed at that origin.
    """
    if fixed.shape != moving.shape or fixed.shape[-1] != 512:
        raise ValueError("requires matching 512-side image pairs")
    if not (0 < width < .25):
        raise ValueError("width must lie in (0, .25)")
    batch = len(fixed)
    gradient = physical_image_gradient(moving)
    gx, gy = gradient[:, :1], gradient[:, 1:2]
    residual = fixed - moving
    t = torch.arange(
        math.ceil(511 * width) + 1,
        device=fixed.device, dtype=fixed.dtype) / (511 * width)
    window = torch.where(
        t <= 1, torch.sin(math.pi * t).square(), 0)
    window_sq = window.square()
    a = _separable_valid(gx.square(), window_sq) + 1.
    b = _separable_valid(gx * gy, window_sq)
    d = _separable_valid(gy.square(), window_sq) + 1.
    bx = _separable_valid(gx * residual, window)
    by = _separable_valid(gy * residual, window)
    determinant = a * d - b.square()
    score = .5 * (
        d * bx.square() - 2 * b * bx * by +
        a * by.square()) / determinant
    # The synthetic origin lies inside [.08,.86]^2, comfortably
    # inside the valid convolutional candidate range.
    flat = score.flatten(1)
    peak = flat.argmax(dim=1)
    side = score.shape[-1]
    row = peak // side
    column = peak % side

    def subpixel(axis: int) -> torch.Tensor:
        if axis == 0:
            minus = score[
                torch.arange(batch, device=fixed.device), 0,
                (row - 1).clamp_min(0), column]
            middle = score[
                torch.arange(batch, device=fixed.device), 0,
                row, column]
            plus = score[
                torch.arange(batch, device=fixed.device), 0,
                (row + 1).clamp_max(side - 1), column]
        else:
            minus = score[
                torch.arange(batch, device=fixed.device), 0,
                row, (column - 1).clamp_min(0)]
            middle = score[
                torch.arange(batch, device=fixed.device), 0,
                row, column]
            plus = score[
                torch.arange(batch, device=fixed.device), 0,
                row, (column + 1).clamp_max(side - 1)]
        curvature = minus - 2 * middle + plus
        return torch.where(
            curvature.abs() > 1e-12,
            .5 * (minus - plus) / curvature,
            torch.zeros_like(curvature)).clamp(-.5, .5)

    origin = torch.stack((
        (column.float() + subpixel(1)) / 511,
        (row.float() + subpixel(0)) / 511,
    ), dim=1)
    axis = torch.arange(
        512, device=fixed.device, dtype=fixed.dtype) / 511
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    tx = (xx[None] - origin[:, 0, None, None]) / width
    ty = (yy[None] - origin[:, 1, None, None]) / width
    wx = torch.where(
        (tx >= 0) & (tx <= 1),
        torch.sin(math.pi * tx).square(), 0)
    wy = torch.where(
        (ty >= 0) & (ty <= 1),
        torch.sin(math.pi * ty).square(), 0)
    w = (wx * wy)[:, None]
    ax = (w.square() * gx.square()).sum(dim=(2, 3)) + 1.
    axy = (w.square() * gx * gy).sum(dim=(2, 3))
    ay = (w.square() * gy.square()).sum(dim=(2, 3)) + 1.
    vx = (w * gx * residual).sum(dim=(2, 3))
    vy = (w * gy * residual).sum(dim=(2, 3))
    den = ax * ay - axy.square()
    vector = torch.stack((
        (ay * vx - axy * vy) / den,
        (ax * vy - axy * vx) / den), dim=-1)[:, 0]
    return origin, vector, score.amax(dim=(1, 2, 3))


def low_map_from_params(
    origin: torch.Tensor, vector: torch.Tensor,
    side: int, dtype: torch.dtype,
    *, width: float | torch.Tensor = 1 / 16,
) -> torch.Tensor:
    axis = torch.arange(
        side, device=origin.device, dtype=dtype) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    widths = torch.as_tensor(
        width, dtype=dtype, device=origin.device).reshape(-1, 1, 1)
    tx = (xx[None] - origin[:, 0, None, None].to(dtype)) / widths
    ty = (yy[None] - origin[:, 1, None, None].to(dtype)) / widths
    wx = torch.where(
        (tx >= 0) & (tx <= 1),
        torch.sin(math.pi * tx).square(), 0)
    wy = torch.where(
        (ty >= 0) & (ty <= 1),
        torch.sin(math.pi * ty).square(), 0)
    identity = torch.stack((xx, yy), dim=-1)[None]
    return identity + (wx * wy)[..., None] * vector.to(
        dtype)[:, None, None, :]


def refine_low_params(
    fixed: torch.Tensor, moving: torch.Tensor,
    origin: torch.Tensor, vector: torch.Tensor,
    steps: int,
    *, width: float = 1 / 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small 4x4 damped Gauss-Newton with a per-sample descent test."""
    batch = len(fixed)
    axis = torch.arange(
        512, device=fixed.device, dtype=fixed.dtype) / 511
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    moving_gradient = physical_image_gradient(moving)

    def evaluate(
        p: torch.Tensor, v: torch.Tensor,
        with_jacobian: bool,
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
        grid = identity + window[..., None] * v[:, None, None]
        sample_grid = 2 * grid - 1
        prediction = F.grid_sample(
            moving, sample_grid, mode="bilinear",
            padding_mode="border", align_corners=True)
        residual = fixed - prediction
        loss = residual.square().mean(dim=(1, 2, 3))
        if not with_jacobian:
            return loss, None, residual
        gradient = F.grid_sample(
            moving_gradient, sample_grid,
            mode="bilinear", padding_mode="border",
            align_corners=True)
        gx, gy = gradient[:, 0], gradient[:, 1]
        d_wx = torch.where(
            inside_x, -math.pi / width * torch.sin(2 * math.pi * tx), 0)
        d_wy = torch.where(
            inside_y, -math.pi / width * torch.sin(2 * math.pi * ty), 0)
        directional = gx * v[:, 0, None, None] + gy * v[:, 1, None, None]
        jacobian = torch.stack((
            window * gx,
            window * gy,
            d_wx * wy * directional,
            wx * d_wy * directional), dim=1)
        return loss, jacobian, residual

    p, v = origin, vector
    for _ in range(steps):
        old_loss, jacobian, residual = evaluate(
            p, v, with_jacobian=True)
        assert jacobian is not None
        normal = torch.einsum(
            "bihw,bjhw->bij", jacobian, jacobian)
        rhs = torch.einsum(
            "bihw,bhw->bi", jacobian, residual[:, 0])
        diagonal = normal.diagonal(dim1=1, dim2=2)
        normal = normal + torch.diag_embed(
            .05 * diagonal + 1e-6)
        update = torch.linalg.solve(normal, rhs)
        update = torch.cat((
            update[:, :2].clamp(-.003, .003),
            update[:, 2:].clamp(-.01, .01)), dim=1)
        best_p, best_v, best_loss = p, v, old_loss
        for alpha in (1., .5, .25):
            candidate_v = v + alpha * update[:, :2]
            norm = torch.linalg.vector_norm(candidate_v, dim=1)
            candidate_v = candidate_v * (
                (.007 / norm.clamp_min(1e-12)).clamp_max(1))[:, None]
            candidate_p = (p + alpha * update[:, 2:]).clamp(.04, .9)
            candidate_loss, _, _ = evaluate(
                candidate_p, candidate_v, with_jacobian=False)
            accepted = candidate_loss < best_loss
            best_p = torch.where(accepted[:, None], candidate_p, best_p)
            best_v = torch.where(accepted[:, None], candidate_v, best_v)
            best_loss = torch.minimum(best_loss, candidate_loss)
        p, v = best_p, best_v
    return p, v


def template_image_mse(
    fixed: torch.Tensor, moving: torch.Tensor,
    origin: torch.Tensor, vector: torch.Tensor,
    *, width: float = 1 / 16,
) -> torch.Tensor:
    mapped = low_map_from_params(
        origin, vector, 512, fixed.dtype, width=width)
    warped = F.grid_sample(
        moving, 2 * mapped - 1,
        mode="bilinear", padding_mode="border",
        align_corners=True)
    return (warped - fixed).square().mean(dim=(1, 2, 3))


def multistart_refine_low_params(
    fixed: torch.Tensor, moving: torch.Tensor,
    origin: torch.Tensor, vector: torch.Tensor,
    steps: int, directions: int = 8,
    *, width: float = 1 / 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Try a bounded initial vector plus diverse directions; choose image fit."""
    norm = torch.linalg.vector_norm(vector, dim=1).clamp_min(1e-12)
    bounded = vector * (.007 / norm).clamp_max(1)[:, None]
    starts = [bounded]
    for index in range(directions):
        theta = 2 * math.pi * index / directions
        starts.append(torch.tensor(
            [.0045 * math.cos(theta), .0045 * math.sin(theta)],
            device=fixed.device, dtype=fixed.dtype)[None].expand(
                len(fixed), -1))
    best_loss = torch.full(
        (len(fixed),), math.inf,
        device=fixed.device, dtype=fixed.dtype)
    best_origin, best_vector = origin, bounded
    for start_vector in starts:
        candidate_origin, candidate_vector = refine_low_params(
            fixed, moving, origin, start_vector, steps,
            width=width)
        loss = template_image_mse(
            fixed, moving, candidate_origin, candidate_vector,
            width=width)
        accepted = loss < best_loss
        best_origin = torch.where(
            accepted[:, None], candidate_origin, best_origin)
        best_vector = torch.where(
            accepted[:, None], candidate_vector, best_vector)
        best_loss = torch.minimum(best_loss, loss)
    return best_origin, best_vector, best_loss


def main() -> None:
    from phase7_test_mixed_scale_image_pipeline import mixed_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=59473)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gn-steps", type=int, default=0)
    parser.add_argument("--gn-multistart", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(1024, 1024), height=512, width=512)
    table.prepare(device=device, dtype=torch.float64)
    data = mixed_dataset(
        args.count, args.seed, device, args.batch, table,
        (80.5, 96.5, 112.5, 127.5),
        return_params=True)
    layer = ResidualStaggeredPatchP1Layer(
        257, patch_cells=16, cycles=2,
        minimum_jacobian=.05).to(device)
    axis = torch.arange(
        257, device=device, dtype=torch.float64) / 256
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    fine_axis = torch.arange(
        1025, device=device, dtype=torch.float64) / 1024
    fine_yy, fine_xx = torch.meshgrid(
        fine_axis, fine_axis, indexing="ij")
    fine_identity = torch.stack((fine_xx, fine_yy), dim=-1)[None]
    sums = {
        name: dict(map=0., low=0., low_n=0,
                   image=0., min_j=math.inf, passed=0)
        for name in ("identity", "oracle_low", "analytic_template",
                     "safe_patch_template", "gn_safe_patch_template",
                     "multistart_safe_patch_template")
    }
    times = []
    multistart_times = []
    per_sample = []
    with torch.no_grad():
        for start in range(0, args.count, args.batch):
            stop = min(start + args.batch, args.count)
            (fixed, moving, target, low_coarse, low_fine,
             low_support, _, _, true_origin, true_vector) = (
                 item[start:stop] for item in data)
            batch = len(fixed)
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            origin, vector, _ = matched_low_patch(
                fixed, moving)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - tick)
            estimated_coarse = low_map_from_params(
                origin, vector, 257, torch.float64)
            proposal = (
                estimated_coarse - identity)[
                    :, 1:-1, 1:-1] / (8 / 256)
            latent = torch.atanh(
                proposal.clamp(-.95, .95))
            safe_coarse = layer(
                identity.expand(batch, -1, -1, -1),
                latent)
            safe_coarse, valid_coarse = certify_p1_or_identity(
                safe_coarse, identity)
            analytic_fine = low_map_from_params(
                origin, vector, 1025, torch.float64)
            _, valid_analytic = certify_p1_or_identity(
                analytic_fine, fine_identity)
            safe_fine = exact_dyadic_p1_refine(
                exact_dyadic_p1_refine(safe_coarse))
            refined_origin, refined_vector = refine_low_params(
                fixed, moving, origin, vector, args.gn_steps)
            refined_coarse = low_map_from_params(
                refined_origin, refined_vector, 257, torch.float64)
            refined_proposal = (
                refined_coarse - identity)[
                    :, 1:-1, 1:-1] / (8 / 256)
            refined_latent = torch.atanh(
                refined_proposal.clamp(-.95, .95))
            refined_safe = layer(
                identity.expand(batch, -1, -1, -1),
                refined_latent)
            refined_safe, valid_refined = certify_p1_or_identity(
                refined_safe, identity)
            refined_fine = exact_dyadic_p1_refine(
                exact_dyadic_p1_refine(refined_safe))
            if args.gn_multistart:
                torch.cuda.synchronize(device)
                multi_tick = time.perf_counter()
                multi_origin, multi_vector, _ = multistart_refine_low_params(
                    fixed, moving, origin, vector, args.gn_steps,
                    directions=args.gn_multistart)
                torch.cuda.synchronize(device)
                multistart_times.append(time.perf_counter() - multi_tick)
            else:
                multi_origin, multi_vector = refined_origin, refined_vector
            multi_coarse = low_map_from_params(
                multi_origin, multi_vector, 257, torch.float64)
            multi_latent = torch.atanh((
                (multi_coarse - identity)[:, 1:-1, 1:-1] /
                (8 / 256)).clamp(-.95, .95))
            multi_safe = layer(
                identity.expand(batch, -1, -1, -1),
                multi_latent)
            multi_safe, valid_multi = certify_p1_or_identity(
                multi_safe, identity)
            multi_fine = exact_dyadic_p1_refine(
                exact_dyadic_p1_refine(multi_safe))
            sample_rmse = (
                (refined_fine - target).square().sum(dim=-1)
                .mean(dim=(1, 2)).sqrt())
            matched_rmse = (
                (safe_fine - target).square().sum(dim=-1)
                .mean(dim=(1, 2)).sqrt())
            for local_index in range(batch):
                per_sample.append(dict(
                    index=start + local_index,
                    matched_map_rmse=float(matched_rmse[local_index]),
                    gn_map_rmse=float(sample_rmse[local_index]),
                    true_origin=true_origin[local_index].cpu().tolist(),
                    matched_origin=origin[local_index].cpu().tolist(),
                    gn_origin=refined_origin[local_index].cpu().tolist(),
                    true_vector=true_vector[local_index].cpu().tolist(),
                    matched_vector=vector[local_index].cpu().tolist(),
                    gn_vector=refined_vector[local_index].cpu().tolist(),
                    multi_origin=multi_origin[local_index].cpu().tolist(),
                    multi_vector=multi_vector[local_index].cpu().tolist(),
                    multi_map_rmse=float((
                        (multi_fine[local_index] - target[local_index])
                        .square().sum(dim=-1).mean().sqrt())),
                ))
            oracle_fine = exact_dyadic_p1_refine(
                exact_dyadic_p1_refine(low_coarse))
            for name, output, valid in (
                ("identity", fine_identity.expand(batch, -1, -1, -1),
                 torch.ones(batch, dtype=torch.bool, device=device)),
                ("oracle_low", oracle_fine,
                 torch.ones(batch, dtype=torch.bool, device=device)),
                ("analytic_template", analytic_fine,
                 valid_analytic),
                ("safe_patch_template", safe_fine, valid_coarse),
                ("gn_safe_patch_template", refined_fine, valid_refined),
                ("multistart_safe_patch_template", multi_fine, valid_multi),
            ):
                row = sums[name]
                square = (output - target).square().sum(dim=-1)
                row["map"] += float(square.mean(
                    dim=(1, 2)).sum())
                mask = low_support > 0
                row["low"] += float(square[mask].sum())
                row["low_n"] += int(mask.sum())
                query = table.interpolate(
                    output.reshape(batch, -1, 2))
                warped = F.grid_sample(
                    moving, 2 * query.float() - 1,
                    mode="bilinear", padding_mode="border",
                    align_corners=True)
                row["image"] += float(
                    (warped - fixed).square().mean(
                        dim=(1, 2, 3)).sum())
                row["min_j"] = min(
                    row["min_j"], minimum_jacobian(output))
                row["passed"] += int(valid.sum())
    rows = []
    for name, row in sums.items():
        rows.append(dict(
            method=name,
            map_vector_rmse=math.sqrt(
                row["map"] / args.count),
            low_support_vector_rmse=math.sqrt(
                row["low"] / row["low_n"]),
            image_mse=row["image"] / args.count,
            minimum_jacobian=row["min_j"],
            passed=row["passed"],
        ))
    print(json.dumps(dict(
        experiment="phase7_mixed_low_matched_template",
        control_vertices=1025 ** 2,
        count=args.count,
        batch=args.batch,
        seed=args.seed,
        gn_steps=args.gn_steps,
        gn_multistart=args.gn_multistart,
        mean_matched_search_seconds_per_batch=(
            sum(times) / len(times)),
        mean_multistart_seconds_per_batch=(
            sum(multistart_times) / len(multistart_times)
            if multistart_times else None),
        worst_gn_samples=sorted(
            per_sample, key=lambda item: item["gn_map_rmse"],
            reverse=True)[:8],
        rows=rows,
        torch_version=torch.__version__,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
