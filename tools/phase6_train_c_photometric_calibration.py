"""Train 18 mode gains on image loss through a positive-conductance P1 solver."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_eval_crossroute_photographic import _edges_and_midpoints
from phase6_test_c_photometric_response import (
    _analytic_responses, _edge_bases, _estimate, _evaluate,
)
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--fit-side", type=int, default=128)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=99317)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=0.005)
    parser.add_argument("--ridge-mode", choices=("mean","diagonal"), default="mean")
    parser.add_argument("--coefficient-bound", type=float, default=4.0)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(20260923)
    mesh = structured_rectangle(args.side-1,args.side-1)
    midpoints = _edges_and_midpoints(args.side, mesh.vertices)
    bases = _edge_bases(args.side, midpoints, device)
    solver = SinePreconditionedTutteLayer(
        args.side, minimum_conductance=1, maximum_conductance=16,
        tolerance=1e-10, max_iterations=120,
    ).to(device)
    responses,response_validation = _analytic_responses(solver,bases,-0.8)
    fit_table = StructuredDenseQueryTable.from_mesh(
        mesh,height=args.fit_side,width=args.fit_side)
    fit_table.prepare(device=device,dtype=torch.float32)
    response_query = fit_table.interpolate(responses.float().reshape(18,-1,2))
    image_table = StructuredDenseQueryTable.from_mesh(
        mesh,height=args.image_side,width=args.image_side)
    image_table.prepare(device=device,dtype=torch.float32)
    train = tuple(value.to(device) for value in make_dataset(
        args.train_count,args.image_side,55101,target_family="high32",
        return_coefficients=True))
    test = tuple(value.to(device) for value in make_dataset(
        args.test_count,args.image_side,args.test_seed,target_family="high32",
        return_coefficients=True))
    raw_gains = torch.nn.Parameter(torch.zeros(18,device=device,dtype=torch.float64))
    optimizer = torch.optim.Adam([raw_gains],lr=args.learning_rate)
    def mode_gains():
        return torch.exp(3*torch.tanh(raw_gains))
    def synchronize():
        if device.type == "cuda": torch.cuda.synchronize(device)
    def evaluate(dataset):
        with torch.no_grad():
            return _evaluate(
                dataset,[None]*len(dataset[0]),[],solver=solver,bases=bases,
                response_query=response_query,mesh=mesh,image_side=args.image_side,
                fit_side=args.fit_side,ridge=args.ridge,
                ridge_mode=args.ridge_mode,gain=1,
                coefficient_bound=args.coefficient_bound,batch=args.batch,
                mode_gains=mode_gains())
    initial_train,initial_test=evaluate(train),evaluate(test)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    generator=torch.Generator(device="cpu").manual_seed(38819)
    forward_times,backward_times,records=[],[],[]
    maximum_forward_residual=0.0
    maximum_adjoint_residual=0.0
    started_all=time.perf_counter()
    for step in range(args.steps):
        indices=torch.randint(args.train_count,(args.batch,),generator=generator).to(device)
        fixed,moving=train[0][indices],train[1][indices]
        optimizer.zero_grad(set_to_none=True)
        synchronize();started=time.perf_counter()
        fixed_fit=F.interpolate(fixed,size=(args.fit_side,args.fit_side),
                                mode="bilinear",align_corners=True)
        moving_fit=F.interpolate(moving,size=(args.fit_side,args.fit_side),
                                 mode="bilinear",align_corners=True)
        estimated,_=_estimate(fixed_fit,moving_fit,response_query,
                              ridge=args.ridge,ridge_mode=args.ridge_mode,
                              coefficient_bound=args.coefficient_bound)
        latent=estimated*mode_gains()[None]
        logits=tuple(-0.8+torch.einsum("bk,k...->b...",latent,part)
                     for part in bases)
        mapped=solver(*logits).float()
        maximum_forward_residual=max(maximum_forward_residual,
                                     solver.last_forward_stats["true_relative_residual"])
        query=image_table.interpolate(mapped.reshape(args.batch,-1,2))
        warped=F.grid_sample(moving,2*query-1,mode="bilinear",
                             padding_mode="border",align_corners=True)
        loss=(warped-fixed).square().mean()
        synchronize();middle=time.perf_counter()
        loss.backward()
        synchronize();finished=time.perf_counter()
        maximum_adjoint_residual=max(maximum_adjoint_residual,
                                     solver.last_backward_stats["true_relative_residual"])
        if not torch.isfinite(raw_gains.grad).all():
            raise RuntimeError("nonfinite mode-gain gradient")
        optimizer.step()
        forward_times.append(middle-started)
        backward_times.append(finished-middle)
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            records.append({"step":step+1,"sampled_image_mse_before_update":float(loss),
                            "gain_min":float(mode_gains().amin()),
                            "gain_max":float(mode_gains().amax()),
                            "elapsed_seconds":time.perf_counter()-started_all})
    training_seconds=time.perf_counter()-started_all
    final_train,final_test=evaluate(train),evaluate(test)
    if args.save_state:
        torch.save({"raw_gains":raw_gains.detach().cpu(),
                    "args":vars(args)},args.save_state)
    print(json.dumps({
        "method":"image_trained_18_mode_photometric_conductance_calibration",
        "representation":"original_grid_P1_positive_conductance",
        "control_side":args.side,
        "control_vertices":args.side**2,
        "control_faces":2*(args.side-1)**2,
        "image_side":args.image_side,
        "image_queries":args.image_side**2,
        "fit_side":args.fit_side,
        "batch":args.batch,
        "train_count":args.train_count,
        "test_count":args.test_count,
        "test_seed":args.test_seed,
        "steps":args.steps,
        "learning_rate":args.learning_rate,
        "ridge":args.ridge,
        "ridge_mode":args.ridge_mode,
        "coefficient_bound":args.coefficient_bound,
        "device":str(device),
        "response_validation":response_validation,
        "initial_train":initial_train,
        "initial_test":initial_test,
        "final_train":final_train,
        "final_test":final_test,
        "learned_mode_gains":mode_gains().detach().tolist(),
        "maximum_training_true_relative_forward_residual":maximum_forward_residual,
        "maximum_training_true_relative_adjoint_residual":maximum_adjoint_residual,
        "median_full_forward_seconds_after_first":statistics.median(forward_times[1:] or forward_times),
        "median_full_vjp_seconds_after_first":statistics.median(backward_times[1:] or backward_times),
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device)
        if device.type=="cuda" else None,
        "training_seconds":training_seconds,
        "records":records,
    },sort_keys=True,separators=(",",":")))


if __name__=="__main__":
    main()
