"""Real control-grid scaling of the reusable spectral Route C neural layer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_evaluate_heldout_beltrami import _mu,_target_on_faces
from phase6_eval_photographic_content import _dataset as photographic_dataset
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    PhotometricSpectralTutteLayer,evaluate_structured_p1_with_jacobian,
)
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--side",type=int,required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--count",type=int,default=8)
    parser.add_argument("--seed",type=int,default=99317)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--fit-side",type=int,default=128)
    parser.add_argument("--frequencies",default="1,2,4,8,16,32",
                        help="comma-separated edge-basis frequencies; may append 64")
    parser.add_argument("--response-chunk-size",type=int,default=1)
    parser.add_argument("--photo-variants",type=int,default=0,
                        help="use six photographic images with this many high32 variants each")
    parser.add_argument("--photo-seed",type=int,default=973031,
                        help="synthetic deformation seed for photographic content")
    parser.add_argument("--target-family",choices=("high32","high64"),
                        default="high32")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    side,image_side= args.side,512
    frequencies=tuple(int(value) for value in args.frequencies.split(","))
    if len(frequencies)!=len(set(frequencies)):
        raise ValueError("frequencies must be unique")
    device=torch.device(args.device)
    state=torch.load(args.checkpoint,map_location=device,weights_only=False)
    layer=PhotometricSpectralTutteLayer(
        side,fit_side=args.fit_side,frequencies=frequencies,
        response_chunk_size=args.response_chunk_size).to(device)
    with torch.no_grad():
        original=tuple(int(value) for value in state.get("args",{}).get(
            "frequencies","1,2,4,8,16,32").split(","))
        original_gains=state["raw_gains"].to(device)
        if original_gains.numel()!=3*len(original):
            raise ValueError("checkpoint gains/frequencies mismatch")
        for axis in range(3):
            for new_index,frequency in enumerate(frequencies):
                if frequency in original:
                    layer.raw_mode_gains[axis*len(frequencies)+new_index]=\
                        original_gains[axis*len(original)+original.index(frequency)]
    if args.photo_variants < 0:
        raise ValueError("photo variants must be nonnegative")
    if args.photo_variants:
        if args.target_family!="high32":
            raise ValueError("photographic variants use the high32 target only")
        photo_names, _, dataset=photographic_dataset(
            args.photo_variants,image_side,args.photo_seed,device)
        count=6*args.photo_variants
        dataset_kind="photographic_content_synthetic_high32"
    else:
        photo_names=[]
        dataset=tuple(value.to(device) for value in make_dataset(
            args.count,image_side,args.seed,target_family=args.target_family,
            return_coefficients=True))
        count=args.count
        dataset_kind=f"synthetic_{args.target_family}_heldout"
    mesh=structured_rectangle(side-1,side-1)
    table=StructuredDenseQueryTable.from_mesh(
        mesh,height=image_side,width=image_side)
    table.prepare(device=device,dtype=torch.float32)
    def synchronize():
        if device.type=="cuda": torch.cuda.synchronize(device)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    synchronize();started=time.perf_counter()
    preparation=layer.prepare(device=device)
    synchronize();prepare_seconds=time.perf_counter()-started
    prepare_peak=(torch.cuda.max_memory_allocated(device)
                  if device.type=="cuda" else None)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    forward_times,backward_times=[],[]
    for repeat in range(args.repeats+1):
        fixed=dataset[0][:1].detach().clone().requires_grad_()
        moving=dataset[1][:1].detach().clone().requires_grad_()
        synchronize();started=time.perf_counter()
        mapped=layer(fixed,moving)
        query=table.interpolate(mapped.reshape(1,-1,2))
        warped=F.grid_sample(moving,2*query-1,mode="bilinear",
                             padding_mode="border",align_corners=True)
        loss=(warped-fixed).square().mean()
        synchronize();middle=time.perf_counter()
        gradients=torch.autograd.grad(loss,(fixed,moving,layer.raw_mode_gains))
        synchronize();finished=time.perf_counter()
        if not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError("nonfinite dense-control VJP")
        if repeat:
            forward_times.append(middle-started)
            backward_times.append(finished-middle)
    train_peak=(torch.cuda.max_memory_allocated(device)
                if device.type=="cuda" else None)
    vertices=torch.tensor(mesh.vertices.copy(),device=device,dtype=torch.float32)
    faces=torch.tensor(mesh.faces.copy(),device=device,dtype=torch.int64)
    centroids=vertices[faces].mean(dim=1)
    source=vertices.reshape(side,side,2)
    cycles=64 if args.target_family=="high64" else 32
    fine=torch.sin(2*cycles*math.pi*source[...,0])*torch.sin(
        2*cycles*math.pi*source[...,1])
    sample_rows=[]
    with torch.no_grad():
        for index in range(count):
            fixed,moving,target,coeff=(part[index:index+1] for part in dataset)
            mapped=layer(fixed,moving)
            query=table.interpolate(mapped.reshape(1,-1,2))
            warped=F.grid_sample(moving,2*query-1,mode="bilinear",
                                 padding_mode="border",align_corners=True)
            face_query=centroids[None]
            _,jacobian=evaluate_structured_p1_with_jacobian(mapped,face_query)
            _,target_jacobian=_target_on_faces(face_query,coeff,cycles)
            mu=_mu(jacobian)
            projection=((mapped-source)*fine[None,:,:,None]).sum(dim=(1,2))/fine.square().sum()
            sample_rows.append({
                "image_mse":float((warped-fixed).square().mean()),
                "query_map_mse":float((query-target).square().mean()),
                "face_beltrami_mse":float((mu-_mu(target_jacobian)).abs().square().mean()),
                "maximum_predicted_beltrami_modulus":float(mu.abs().amax()),
                "minimum_face_determinant":float(torch.linalg.det(jacobian).amin()),
                "projected_fine_amplitudes":projection[0].tolist(),
                "true_fine_amplitude":float(coeff[0,2]),
                "true_relative_forward_residual":layer.solver.last_forward_stats["true_relative_residual"],
            })
    print(json.dumps({
        "method":"photometric_spectral_tutte_control_scaling",
        "side":side,"control_vertices":side**2,
        "control_faces":2*(side-1)**2,
        "image_side":image_side,"image_queries":image_side**2,
        "fit_side":args.fit_side,"frequencies":frequencies,
        "batch":1,"device":str(device),
        "response_chunk_size":args.response_chunk_size,
        "checkpoint":args.checkpoint,
        "count":count,
        "seed":args.photo_seed if args.photo_variants else args.seed,
        "dataset_kind":dataset_kind,
        "target_family":args.target_family,
        "photo_names":photo_names,
        "prepare_seconds":prepare_seconds,
        "prepare_peak_cuda_allocated_bytes":prepare_peak,
        "response_validation":preparation,
        "median_full_forward_seconds":statistics.median(forward_times),
        "median_full_vjp_seconds":statistics.median(backward_times),
        "full_step_peak_cuda_allocated_bytes":train_peak,
        "true_relative_adjoint_residual":layer.solver.last_backward_stats["true_relative_residual"],
        "gradient_norms":[float(value.norm()) for value in gradients],
        "image_mse":statistics.mean(row["image_mse"] for row in sample_rows),
        "query_map_rmse":math.sqrt(statistics.mean(row["query_map_mse"] for row in sample_rows)),
        "face_beltrami_rmse":math.sqrt(statistics.mean(row["face_beltrami_mse"] for row in sample_rows)),
        "maximum_predicted_beltrami_modulus":max(row["maximum_predicted_beltrami_modulus"] for row in sample_rows),
        "minimum_face_determinant":min(row["minimum_face_determinant"] for row in sample_rows),
        "maximum_true_relative_forward_residual":max(row["true_relative_forward_residual"] for row in sample_rows),
        "fine_amplitude_slope":sum(row["true_fine_amplitude"]
            *sum(row["projected_fine_amplitudes"])/2 for row in sample_rows)
            /max(sum(row["true_fine_amplitude"]**2 for row in sample_rows),1e-20),
        "samples":sample_rows,
    },sort_keys=True,separators=(",",":")))


if __name__=="__main__":
    main()
