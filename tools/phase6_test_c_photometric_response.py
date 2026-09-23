"""Route C diagnostic: image-only 18-mode photometric coefficient solve."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from phase6_eval_photographic_content import _dataset as photo_dataset
from phase6_eval_crossroute_photographic import _edges_and_midpoints
from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_spectral_edge_encoder import SpectralEdgeImageEncoder
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer, evaluate_structured_p1_with_jacobian
from qcopt.neural_bijection.dense.sine_pcg_tutte import _laplacian, _pcg
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def _edge_bases(side: int, points: np.ndarray, device: torch.device):
    encoder = SpectralEdgeImageEncoder(side, points).to(device)
    kh = len(encoder.frequencies)
    nh = side * (side-1)
    nd = (side-1)**2
    horizontal = torch.zeros(3*kh, side, side-1, device=device, dtype=torch.float64)
    vertical = torch.zeros(3*kh, side-1, side, device=device, dtype=torch.float64)
    diagonal = torch.zeros(3*kh, side-1, side-1, device=device, dtype=torch.float64)
    horizontal[:kh] = encoder.horizontal_basis.double().reshape(kh, side, side-1)
    vertical[kh:2*kh] = encoder.vertical_basis.double().reshape(kh, side-1, side)
    diagonal[2*kh:] = encoder.diagonal_basis.double().reshape(kh, side-1, side-1)
    assert horizontal.numel() + vertical.numel() + diagonal.numel() == 3*kh*(2*nh+nd)
    return horizontal, vertical, diagonal


@torch.no_grad()
def _analytic_responses(solver, bases, base_logit: float):
    count = bases[0].shape[0]
    sigmoid = torch.sigmoid(torch.tensor(base_logit, device=bases[0].device, dtype=torch.float64))
    constant = solver.minimum_conductance + (
        solver.maximum_conductance-solver.minimum_conductance) * sigmoid
    conductances = tuple(torch.full_like(value, constant) for value in bases)
    derivative = tuple((solver.maximum_conductance-solver.minimum_conductance)
                       * sigmoid*(1-sigmoid) * value for value in bases)
    source = solver._source[None].expand(count, -1, -1, -1)
    rhs = -_laplacian(source, *derivative)[:, 1:-1, 1:-1]
    response, iterations, residual = _pcg(
        rhs, conductances, solver._eigenvalues,
        tolerance=1e-11, max_iterations=160,
    )
    full = F.pad(response.permute(0,3,1,2), (1,1,1,1)).permute(0,2,3,1)
    # Independent finite-difference check for one low and one fine mode.
    discrepancies = {}
    for index in (0, 5, 17):
        epsilon = 1e-3
        def solve(sign):
            logits = tuple(torch.full_like(part[index:index+1], base_logit)
                           + sign*epsilon*part[index:index+1] for part in bases)
            return solver(*logits)
        finite = (solve(1)-solve(-1))/(2*epsilon)
        discrepancies[str(index)] = float((finite-full[index:index+1]).abs().amax())
    return full, {"iterations": iterations, "true_relative_residual": residual,
                  "finite_difference_max_abs": discrepancies}


def _image_gradients(moving: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    side = moving.shape[-1]
    padded_x = F.pad(moving, (1,1,0,0), mode="replicate")
    padded_y = F.pad(moving, (0,0,1,1), mode="replicate")
    x = 0.5*(side-1)*(padded_x[...,2:]-padded_x[...,:-2])
    y = 0.5*(side-1)*(padded_y[...,2:,:]-padded_y[...,:-2,:])
    return x, y


def _estimate(fixed, moving, response_query, *, ridge: float,
              ridge_mode: str, coefficient_bound: float,
              sample_at: torch.Tensor | None = None):
    grad_x, grad_y = _image_gradients(moving)
    if sample_at is not None:
        sample_grid = 2*sample_at-1
        grad_x = F.grid_sample(grad_x,sample_grid,mode="bilinear",
                               padding_mode="border",align_corners=True)
        grad_y = F.grid_sample(grad_y,sample_grid,mode="bilinear",
                               padding_mode="border",align_corners=True)
        warped = F.grid_sample(moving,sample_grid,mode="bilinear",
                               padding_mode="border",align_corners=True)
    else:
        warped = moving
    # B,K,H,W; analytic response is fixed, so only the small coefficient
    # system carries per-image state through the inverse calculation.
    design = (response_query[None,:, :, :, 0]*grad_x[:,None,0]
              + response_query[None,:, :, :, 1]*grad_y[:,None,0]).flatten(2)
    residual = (fixed-warped).flatten(2).double()
    design = design.double()
    normal = design @ design.transpose(1,2) / design.shape[-1]
    rhs = (design * residual).mean(dim=-1)
    scale = normal.diagonal(dim1=1,dim2=2).mean(dim=1)
    identity = torch.eye(normal.shape[1], device=normal.device, dtype=normal.dtype)
    if ridge_mode == "mean":
        regularizer = (ridge*scale.clamp_min(1e-12))[:,None,None]*identity
    elif ridge_mode == "diagonal":
        regularizer = ridge*torch.diag_embed(
            normal.diagonal(dim1=1,dim2=2).clamp_min(1e-12))
    else:
        raise ValueError("ridge_mode must be mean or diagonal")
    regularized = normal + regularizer
    coefficients = torch.linalg.solve(regularized, rhs[...,None])[...,0]
    singular = torch.linalg.eigvalsh(normal)
    condition = singular[:,-1]/singular[:,0].clamp_min(1e-20)
    bounded = coefficient_bound*torch.tanh(coefficients/coefficient_bound)
    return bounded, {"raw_coefficients": coefficients,
                     "normal_condition": condition,
                     "normal_scale": scale}


def _evaluate(dataset, photo_ids, names, *, solver, bases, response_query,
              mesh, image_side, fit_side, ridge, ridge_mode, gain,
              coefficient_bound, batch, mode_gains=None,
              refine_passes=0, refine_step=0.5):
    device = dataset[0].device
    side = solver.side
    table = StructuredDenseQueryTable.from_mesh(mesh, height=image_side, width=image_side)
    table.prepare(device=device, dtype=torch.float32)
    fit_table = StructuredDenseQueryTable.from_mesh(
        mesh,height=fit_side,width=fit_side)
    fit_table.prepare(device=device,dtype=torch.float32)
    vertices = torch.tensor(mesh.vertices.copy(), device=device, dtype=torch.float32)
    faces = torch.tensor(mesh.faces.copy(), device=device, dtype=torch.int64)
    centroids = vertices[faces].mean(dim=1)
    source_grid = vertices.reshape(side,side,2)
    high_basis = (torch.sin(64*math.pi*source_grid[...,0])
                  * torch.sin(64*math.pi*source_grid[...,1]))
    rows, timings = [], []
    with torch.no_grad():
        for begin in range(0, len(dataset[0]), batch):
            end = min(begin+batch, len(dataset[0]))
            fixed, moving, target, coeff = (part[begin:end] for part in dataset)
            if device.type == "cuda": torch.cuda.synchronize(device)
            start = time.perf_counter()
            fixed_fit = F.interpolate(fixed, size=(fit_side,fit_side), mode="bilinear", align_corners=True)
            moving_fit = F.interpolate(moving, size=(fit_side,fit_side), mode="bilinear", align_corners=True)
            estimate, diagnostics = _estimate(
                fixed_fit, moving_fit, response_query,
                ridge=ridge, ridge_mode=ridge_mode,
                coefficient_bound=coefficient_bound,
            )
            estimate = gain*estimate
            if mode_gains is not None:
                estimate = estimate*mode_gains[None]
            logits = tuple(-0.8 + torch.einsum("bk,k...->b...", estimate, part)
                           for part in bases)
            mapped = solver(*logits).float()
            for _ in range(refine_passes):
                fit_query = fit_table.interpolate(mapped.reshape(end-begin,-1,2))
                correction,_ = _estimate(
                    fixed_fit,moving_fit,response_query,ridge=ridge,
                    ridge_mode=ridge_mode,coefficient_bound=coefficient_bound,
                    sample_at=fit_query)
                estimate = estimate + refine_step*correction
                logits = tuple(-0.8 + torch.einsum("bk,k...->b...", estimate, part)
                               for part in bases)
                mapped = solver(*logits).float()
            solver_residual = solver.last_forward_stats["true_relative_residual"]
            query = table.interpolate(mapped.reshape(end-begin, -1, 2))
            warped = F.grid_sample(moving, 2*query-1, mode="bilinear",
                                   padding_mode="border", align_corners=True)
            if device.type == "cuda": torch.cuda.synchronize(device)
            timings.append(time.perf_counter()-start)
            face_query = centroids[None].expand(end-begin,-1,-1)
            _, jacobian = evaluate_structured_p1_with_jacobian(mapped, face_query)
            _, target_jacobian = _target_on_faces(face_query, coeff, 32)
            mu_pred = _mu(jacobian)
            mu_mse = (mu_pred-_mu(target_jacobian)).abs().square().mean(dim=1)
            image_mse = (warped-fixed).square().mean(dim=(1,2,3))
            map_mse = (query-target).square().mean(dim=(1,2,3))
            det = torch.linalg.det(jacobian)
            projected = ((mapped-source_grid)*high_basis[None,:,:,None]).sum(dim=(1,2))/high_basis.square().sum()
            for index in range(end-begin):
                rows.append({
                    "photo": names[photo_ids[begin+index]] if names else None,
                    "image_mse": float(image_mse[index]),
                    "query_map_mse": float(map_mse[index]),
                    "face_beltrami_mse": float(mu_mse[index]),
                    "maximum_predicted_beltrami_modulus": float(mu_pred[index].abs().amax()),
                    "minimum_face_determinant": float(det[index].amin()),
                    "maximum_abs_coefficient": float(estimate[index].abs().amax()),
                    "estimated_fine_coefficients": estimate[index,[5,11,17]].tolist(),
                    "projected_fine_amplitudes": projected[index].tolist(),
                    "true_fine_amplitude": float(coeff[index,2]),
                    "normal_condition": float(diagnostics["normal_condition"][index]),
                    "solver_true_relative_residual_in_batch": solver_residual,
                })
    return {
        "count": len(rows),
        "image_mse": statistics.mean(row["image_mse"] for row in rows),
        "query_map_rmse": math.sqrt(statistics.mean(row["query_map_mse"] for row in rows)),
        "face_beltrami_rmse": math.sqrt(statistics.mean(row["face_beltrami_mse"] for row in rows)),
        "maximum_predicted_beltrami_modulus": max(row["maximum_predicted_beltrami_modulus"] for row in rows),
        "minimum_face_determinant": min(row["minimum_face_determinant"] for row in rows),
        "maximum_abs_coefficient": max(row["maximum_abs_coefficient"] for row in rows),
        "maximum_normal_condition": max(row["normal_condition"] for row in rows),
        "maximum_solver_true_relative_residual": max(row["solver_true_relative_residual_in_batch"] for row in rows),
        "fine_amplitude_slope": (
            sum(row["true_fine_amplitude"] * sum(row["projected_fine_amplitudes"])/2 for row in rows)
            / max(sum(row["true_fine_amplitude"]**2 for row in rows),1e-20)),
        "mean_batch_forward_seconds": statistics.mean(timings),
        "per_sample": rows,
    }


def _benchmark_vjp(dataset, *, solver, bases, response_query, mesh,
                   image_side, fit_side, ridge, ridge_mode, gain,
                   coefficient_bound, batch, repeats=5):
    device = dataset[0].device
    table = StructuredDenseQueryTable.from_mesh(mesh, height=image_side, width=image_side)
    table.prepare(device=device, dtype=torch.float32)
    def synchronize():
        if device.type == "cuda": torch.cuda.synchronize(device)
    forward_times, backward_times, norms = [], [], []
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for repeat in range(repeats+1):
        fixed = dataset[0][:batch].detach().clone().requires_grad_()
        moving = dataset[1][:batch].detach().clone().requires_grad_()
        gain_parameter = torch.tensor(gain, device=device, dtype=torch.float64,
                                      requires_grad=True)
        synchronize(); started = time.perf_counter()
        fixed_fit = F.interpolate(fixed, size=(fit_side,fit_side),
                                  mode="bilinear", align_corners=True)
        moving_fit = F.interpolate(moving, size=(fit_side,fit_side),
                                   mode="bilinear", align_corners=True)
        coefficients,_ = _estimate(fixed_fit,moving_fit,response_query,
                                   ridge=ridge,ridge_mode=ridge_mode,
                                   coefficient_bound=coefficient_bound)
        logits = tuple(-0.8 + torch.einsum("bk,k...->b...",
                                           gain_parameter*coefficients, part)
                       for part in bases)
        mapped = solver(*logits).float()
        query = table.interpolate(mapped.reshape(batch,-1,2))
        warped = F.grid_sample(moving,2*query-1,mode="bilinear",
                               padding_mode="border",align_corners=True)
        loss = (warped-fixed).square().mean()
        synchronize(); middle = time.perf_counter()
        grad_fixed,grad_moving,grad_gain = torch.autograd.grad(
            loss,(fixed,moving,gain_parameter))
        synchronize(); finished = time.perf_counter()
        if not (torch.isfinite(grad_fixed).all() and torch.isfinite(grad_moving).all()
                and torch.isfinite(grad_gain)):
            raise RuntimeError("nonfinite photometric-response first-order VJP")
        if repeat:
            forward_times.append(middle-started)
            backward_times.append(finished-middle)
            norms.append([float(grad_fixed.norm()),float(grad_moving.norm()),
                          float(grad_gain)])
    return {
        "median_full_forward_seconds":statistics.median(forward_times),
        "median_full_vjp_seconds":statistics.median(backward_times),
        "gradient_norms_last":norms[-1],
        "maximum_true_relative_adjoint_residual":solver.last_backward_stats["true_relative_residual"],
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device)
        if device.type == "cuda" else None,
        "repeats_after_warmup":repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--fit-side", type=int, default=None,
                        help="resolution of the coefficient normal equation; final image remains image-side")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--photo-variants", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--ridge-mode", choices=("mean", "diagonal"), default="mean")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--coefficient-bound", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark-vjp", action="store_true")
    parser.add_argument("--calibration-checkpoint", default=None)
    parser.add_argument("--refine-passes", type=int, default=0)
    parser.add_argument("--refine-step", type=float, default=0.5)
    args = parser.parse_args()
    if args.fit_side is None:
        args.fit_side = args.image_side
    if args.fit_side < 2 or args.fit_side > args.image_side:
        raise ValueError("fit-side must lie in [2,image-side]")
    if args.refine_passes < 0 or not 0 < args.refine_step <= 1:
        raise ValueError("invalid refinement configuration")
    device = torch.device(args.device)
    mesh = structured_rectangle(args.side-1,args.side-1)
    midpoints = _edges_and_midpoints(args.side, mesh.vertices)
    bases = _edge_bases(args.side, midpoints, device)
    solver = SinePreconditionedTutteLayer(
        args.side, minimum_conductance=1, maximum_conductance=16,
        tolerance=1e-10, max_iterations=120,
    ).to(device)
    started = time.perf_counter()
    responses, validation = _analytic_responses(solver, bases, -0.8)
    table = StructuredDenseQueryTable.from_mesh(mesh, height=args.fit_side,
                                                 width=args.fit_side)
    table.prepare(device=device,dtype=torch.float32)
    response_query = table.interpolate(responses.float().reshape(18,-1,2))
    precompute_seconds = time.perf_counter()-started
    if args.photo_variants:
        names, photo_ids, dataset = photo_dataset(
            args.photo_variants,args.image_side,args.test_seed,device)
    else:
        names,photo_ids=[],[None]*args.test_count
        dataset = tuple(part.to(device) for part in make_dataset(
            args.test_count,args.image_side,args.test_seed,
            target_family="high32",return_coefficients=True))
    mode_gains = None
    if args.calibration_checkpoint:
        state = torch.load(args.calibration_checkpoint,map_location=device,weights_only=False)
        if state["args"]["side"] != args.side or state["args"]["fit_side"] != args.fit_side:
            raise ValueError("calibration checkpoint side mismatch")
        mode_gains = torch.exp(3*torch.tanh(state["raw_gains"].to(device)))
    result = _evaluate(dataset, photo_ids, names, solver=solver,
                       bases=bases,response_query=response_query,mesh=mesh,
                       image_side=args.image_side,fit_side=args.fit_side,ridge=args.ridge,
                       ridge_mode=args.ridge_mode,gain=args.gain,
                       coefficient_bound=args.coefficient_bound,batch=args.batch,
                       mode_gains=mode_gains,refine_passes=args.refine_passes,
                       refine_step=args.refine_step)
    vjp = (_benchmark_vjp(
        dataset,solver=solver,bases=bases,response_query=response_query,
        mesh=mesh,image_side=args.image_side,fit_side=args.fit_side,
        ridge=args.ridge,ridge_mode=args.ridge_mode,gain=args.gain,
        coefficient_bound=args.coefficient_bound,batch=args.batch)
        if args.benchmark_vjp else None)
    print(json.dumps({
        "method":"photometric_normal_equation_on_18_conductance_response_modes",
        "data":"photographic_content" if args.photo_variants else "synthetic_high32",
        "seed":args.test_seed,
        "photo_variants":args.photo_variants,
        "control_side":args.side,
        "control_vertices":args.side**2,
        "control_faces":2*(args.side-1)**2,
        "image_side":args.image_side,
        "fit_side":args.fit_side,
        "image_queries":args.image_side**2,
        "batch":args.batch,
        "device":str(device),
        "ridge":args.ridge,
        "ridge_mode":args.ridge_mode,
        "gain":args.gain,
        "coefficient_bound":args.coefficient_bound,
        "calibration_checkpoint":args.calibration_checkpoint,
        "mode_gains":mode_gains.tolist() if mode_gains is not None else None,
        "refine_passes":args.refine_passes,
        "refine_step":args.refine_step,
        "precompute_seconds":precompute_seconds,
        "response_validation":validation,
        "result":result,
        "full_vjp_benchmark":vjp,
    },sort_keys=True,separators=(",",":")))


if __name__ == "__main__":
    main()
