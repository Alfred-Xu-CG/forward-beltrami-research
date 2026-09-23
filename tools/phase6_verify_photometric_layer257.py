"""Check reusable Route C layer against its experiment implementation at 257²."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_eval_crossroute_photographic import _edges_and_midpoints
from phase6_test_c_photometric_response import _analytic_responses,_edge_bases,_estimate
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import PhotometricSpectralTutteLayer,SinePreconditionedTutteLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    device=torch.device(args.device)
    side,image_side,fit_side=257,512,128
    state=torch.load(args.checkpoint,map_location=device,weights_only=False)
    fixed,moving,true_map,_=tuple(value[:2].to(device) for value in make_dataset(
        8,image_side,99317,target_family="high32",return_coefficients=True))
    mesh=structured_rectangle(side-1,side-1)
    layer=PhotometricSpectralTutteLayer(side,fit_side=fit_side).to(device)
    with torch.no_grad():
        layer.raw_mode_gains.copy_(state["raw_gains"].to(device))
    def synchronize():
        if device.type=="cuda": torch.cuda.synchronize(device)
    synchronize();start=time.perf_counter()
    prepared=layer.prepare(device=device)
    synchronize();precompute_seconds=time.perf_counter()-start
    fit_table=StructuredDenseQueryTable.from_mesh(mesh,height=fit_side,width=fit_side)
    fit_table.prepare(device=device,dtype=torch.float32)
    image_table=StructuredDenseQueryTable.from_mesh(mesh,height=image_side,width=image_side)
    image_table.prepare(device=device,dtype=torch.float32)
    with torch.no_grad():
        delivered,latent=layer.forward_with_latent(fixed,moving)
        points=image_table.interpolate(delivered.reshape(2,-1,2))
        warped=F.grid_sample(moving,2*points-1,mode="bilinear",
                             padding_mode="border",align_corners=True)
        image_mse=float((warped-fixed).square().mean())
        map_rmse=float((points-true_map).square().mean().sqrt())

        basis=_edge_bases(side,_edges_and_midpoints(side,mesh.vertices),device)
        reference_solver=SinePreconditionedTutteLayer(
            side,minimum_conductance=1,maximum_conductance=16,
            tolerance=1e-10,max_iterations=120).to(device)
        responses,_=_analytic_responses(reference_solver,basis,-0.8)
        response_query=fit_table.interpolate(responses.float().reshape(18,-1,2))
        fixed_fit=F.interpolate(fixed,size=(fit_side,fit_side),mode="bilinear",align_corners=True)
        moving_fit=F.interpolate(moving,size=(fit_side,fit_side),mode="bilinear",align_corners=True)
        estimated,_=_estimate(fixed_fit,moving_fit,response_query,
                              ridge=0.005,ridge_mode="mean",coefficient_bound=4.0)
        reference_latent=estimated*layer.mode_gains[None]
        logits=tuple(-0.8+torch.einsum("bk,k...->b...",reference_latent,part)
                     for part in basis)
        reference=reference_solver(*logits).float()
        maximum_map_difference=float((delivered-reference).abs().amax())
        maximum_latent_difference=float((latent-reference_latent).abs().amax())
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    forward_times,backward_times=[],[]
    for repeat in range(6):
        layer.zero_grad(set_to_none=True)
        fixed_input=fixed.detach().clone().requires_grad_()
        moving_input=moving.detach().clone().requires_grad_()
        synchronize();start=time.perf_counter()
        mapped=layer(fixed_input,moving_input)
        query=image_table.interpolate(mapped.reshape(2,-1,2))
        image=F.grid_sample(moving_input,2*query-1,mode="bilinear",
                            padding_mode="border",align_corners=True)
        loss=(image-fixed_input).square().mean()
        synchronize();middle=time.perf_counter()
        gradients=torch.autograd.grad(loss,(fixed_input,moving_input,layer.raw_mode_gains))
        synchronize();end=time.perf_counter()
        if not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("nonfinite reusable-layer VJP")
        if repeat:
            forward_times.append(middle-start)
            backward_times.append(end-middle)
    print(json.dumps({
        "method":"photometric_spectral_tutte_reusable_module_verification",
        "control_side":side,"control_vertices":side**2,
        "image_side":image_side,"image_queries":image_side**2,
        "fit_side":fit_side,"batch":2,"device":str(device),
        "checkpoint":args.checkpoint,
        "response_precompute_seconds":precompute_seconds,
        "response_validation":prepared,
        "maximum_module_reference_map_difference":maximum_map_difference,
        "maximum_module_reference_latent_difference":maximum_latent_difference,
        "image_mse_two_examples":image_mse,
        "map_rmse_two_examples":map_rmse,
        "minimum_face_area_ratio":layer.solver.last_forward_stats["minimum_signed_area_ratio"],
        "maximum_true_relative_forward_residual":layer.solver.last_forward_stats["true_relative_residual"],
        "maximum_true_relative_adjoint_residual":layer.solver.last_backward_stats["true_relative_residual"],
        "median_full_forward_seconds":statistics.median(forward_times),
        "median_full_vjp_seconds":statistics.median(backward_times),
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device)
        if device.type=="cuda" else None,
        "gradient_norms":[float(g.norm()) for g in gradients],
    },sort_keys=True,separators=(",",":")))


if __name__=="__main__":
    main()
