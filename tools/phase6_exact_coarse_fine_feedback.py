"""Exact nested P1 refinement of trained A8, then safe fine-grid image feedback."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from phase6_evaluate_heldout_beltrami import _mu, _target_on_faces
from phase6_eval_photographic_content import _dataset as photographic_dataset
from phase6_train_image_to_latent import _minimum_area_ratio
from phase6_train_multisample_image import ConvexQuadLocalImageEncoder, make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import (
    NestedP1PhotometricFeedbackLayer, SpectralSafeFeedbackLayer,
    evaluate_structured_p1_with_jacobian,
    structured_p1_face_beltrami_modulus,
)
from qcopt.neural_bijection.dense.photometric_hint import local_photometric_logits
from qcopt.neural_bijection.dense.qc_radial_relaxation import SafeColoredQCRadialRelaxation
from qcopt.neural_bijection.dense.sine_spectral import spectralize_bounded_logits
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--fine-side",type=int,default=1025)
    parser.add_argument("--passes",type=int,default=0)
    parser.add_argument("--gain",type=float,default=1.0)
    parser.add_argument("--coarse-initial-cap",type=float,default=None)
    parser.add_argument("--fine-qc-cap",type=float,default=0.8)
    parser.add_argument("--use-module",action="store_true")
    parser.add_argument("--test-count",type=int,default=8)
    parser.add_argument("--test-seed",type=int,default=99317)
    parser.add_argument("--photo-variants",type=int,default=0)
    parser.add_argument("--steps",type=int,default=0)
    parser.add_argument("--save-state",default=None)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    if args.fine_side < 257 or (args.fine_side-1)%256 or args.passes < 0:
        raise ValueError("fine side must be nested over 257; passes nonnegative")
    device=torch.device(args.device)
    torch.manual_seed(20260923)
    state=torch.load(args.checkpoint,map_location=device,weights_only=False)
    settings=state["args"]
    if settings["method"]!="A8" or settings["side"]!=257:
        raise ValueError("expected trained 257-side A8 checkpoint")
    coarse_side,fine_side,image_side=257,args.fine_side,512
    encoder=ConvexQuadLocalImageEncoder(
        coarse_side,width=settings["a2_width"],
        head_mode=settings["a2_head_mode"],
        body_mode=settings["a2_body_mode"]).to(device)
    encoder.load_state_dict(state["encoder"])
    coarse_decoder=SpectralSafeFeedbackLayer(
        coarse_side,initial_window=settings["hint_window"],
        initial_ridge=settings["hint_ridge"],
        initial_gain=settings["hint_gain"],
        initial_modes=settings["hint_sine_modes"],
        extra_passes=settings["extra_passes"],
        extra_gain=settings["extra_gain"],
        extra_modes=16,extra_qc_cap=args.fine_qc_cap,
        initial_qc_cap=args.coarse_initial_cap,
        floor_fraction=settings["floor_fraction"]).to(device)
    fine_layer=(NestedP1PhotometricFeedbackLayer(
        coarse_side,fine_side,fine_passes=args.passes,gain=args.gain,
        qc_cap=args.fine_qc_cap,window=3,ridge=1,
        spectral_modes=16,floor_fraction=0.8).to(device)
        if args.use_module else None)
    if fine_layer is not None:
        fine_layer.prepare(device=device)
    refiner=(fine_layer.refiner if fine_layer is not None else
             SafeColoredQCRadialRelaxation(
                 fine_side,qc_cap=args.fine_qc_cap,
                 floor_fraction=0.8).to(device))
    coarse_mesh=structured_rectangle(coarse_side-1,coarse_side-1)
    fine_mesh=structured_rectangle(fine_side-1,fine_side-1)
    refine_table=StructuredDenseQueryTable.from_mesh(
        coarse_mesh,height=fine_side,width=fine_side)
    refine_table.prepare(device=device,dtype=torch.float32)
    coarse_query=StructuredDenseQueryTable.from_mesh(
        coarse_mesh,height=image_side,width=image_side)
    coarse_query.prepare(device=device,dtype=torch.float32)
    fine_query=StructuredDenseQueryTable.from_mesh(
        fine_mesh,height=image_side,width=image_side)
    fine_query.prepare(device=device,dtype=torch.float32)
    fine_vertices=torch.tensor(
        fine_mesh.vertices.copy(),device=device,dtype=torch.float32)
    centroids=fine_vertices[torch.tensor(
        fine_mesh.faces.copy(),device=device)].mean(dim=1)
    train=tuple(value.to(device) for value in make_dataset(
        32,image_side,55101,target_family="high32",
        return_coefficients=True))
    photo_names,photo_indices=None,None
    if args.photo_variants:
        photo_names,photo_indices,heldout=photographic_dataset(
            args.photo_variants,image_side,args.test_seed,device)
    else:
        heldout=tuple(value.to(device) for value in make_dataset(
            args.test_count,image_side,args.test_seed,target_family="high32",
            return_coefficients=True))
    evaluation_count=len(heldout[0])

    def synchronize():
        if device.type=="cuda": torch.cuda.synchronize(device)

    def forward(fixed,moving):
        coarse=coarse_decoder(fixed,moving,encoder(torch.cat((fixed,moving),dim=1)))
        mapped=refine_table.interpolate(
            coarse.reshape(coarse.shape[0],-1,2)).reshape(
                coarse.shape[0],fine_side,fine_side,2)
        base=mapped
        if fine_layer is not None:
            mapped=fine_layer(fixed,moving,coarse)
        else:
            floor=refiner.compute_area_floor(mapped) if args.passes else None
            def refine(current,proposal,area_floor):
                return refiner(current,proposal,area_floor=area_floor)
            for _ in range(args.passes):
                hint=local_photometric_logits(
                    fixed,moving,mapped,window=3,ridge=1,
                    raw_span=refiner.raw_span)
                hint=spectralize_bounded_logits(
                    hint,side=fine_side,raw_span=refiner.raw_span,count=16)
                proposal=args.gain*hint
                mapped=(checkpoint(refine,mapped,proposal,floor,use_reentrant=False)
                        if torch.is_grad_enabled() else
                        refine(mapped,proposal,floor))
        dense=fine_query.interpolate(mapped.reshape(mapped.shape[0],-1,2))
        warped=F.grid_sample(moving,2*dense-1,mode="bilinear",
                             padding_mode="border",align_corners=True)
        return (warped-fixed).square().mean(),coarse,base,mapped,dense

    @torch.no_grad()
    def evaluate(dataset,count):
        rows=[]
        for index in range(count):
            fixed,moving,target,coeff=(part[index:index+1] for part in dataset)
            loss,coarse,base,mapped,dense=forward(fixed,moving)
            coarse_dense=coarse_query.interpolate(coarse.reshape(1,-1,2))
            base_dense=fine_query.interpolate(base.reshape(1,-1,2))
            _,jacobian=evaluate_structured_p1_with_jacobian(
                mapped,centroids[None])
            _,target_jacobian=_target_on_faces(centroids[None],coeff,32)
            mu=_mu(jacobian)
            rows.append({
                "image_mse":float(loss),
                "query_map_mse":float((dense-target).square().mean()),
                "face_beltrami_mse":float((mu-_mu(target_jacobian)).abs().square().mean()),
                "minimum_signed_area_ratio":_minimum_area_ratio(mapped),
                "maximum_beltrami_modulus":float(mu.abs().amax()),
                "coarse_maximum_beltrami_modulus":float(
                    structured_p1_face_beltrami_modulus(coarse).amax()),
                "coarse_refined_query_max_abs_difference":float(
                    (coarse_dense-base_dense).abs().amax()),
                "fine_update_max_abs":float((mapped-base).abs().amax()),
            })
        return {
            "count":count,
            "image_mse":statistics.mean(row["image_mse"] for row in rows),
            "query_map_rmse":math.sqrt(statistics.mean(
                row["query_map_mse"] for row in rows)),
            "face_beltrami_rmse":math.sqrt(statistics.mean(
                row["face_beltrami_mse"] for row in rows)),
            "minimum_signed_area_ratio":min(
                row["minimum_signed_area_ratio"] for row in rows),
            "maximum_beltrami_modulus":max(
                row["maximum_beltrami_modulus"] for row in rows),
            "coarse_cap_eligible_count":sum(
                row["coarse_maximum_beltrami_modulus"]<args.fine_qc_cap
                for row in rows),
            "coarse_refined_query_max_abs_difference":max(
                row["coarse_refined_query_max_abs_difference"] for row in rows),
            "maximum_fine_update_abs":max(
                row["fine_update_max_abs"] for row in rows),
            "samples":rows,
        }

    initial=evaluate(heldout,evaluation_count)
    optimizer=torch.optim.Adam(encoder.parameters(),lr=0.001)
    generator=torch.Generator(device="cpu").manual_seed(38819)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    forward_times,backward_times,records=[],[],[]
    minimum_train_area=math.inf
    began_all=time.perf_counter()
    repeats=max(3,args.steps) if args.steps==0 else args.steps
    for step in range(repeats):
        draw=torch.randint(32,(1,),generator=generator).to(device)
        fixed,moving=train[0][draw],train[1][draw]
        optimizer.zero_grad(set_to_none=True)
        synchronize();began=time.perf_counter()
        loss,_,_,mapped,_=forward(fixed,moving)
        minimum_train_area=min(minimum_train_area,_minimum_area_ratio(mapped))
        synchronize();middle=time.perf_counter()
        loss.backward()
        synchronize();ended=time.perf_counter()
        if not all(parameter.grad is None or torch.isfinite(
                parameter.grad).all() for parameter in encoder.parameters()):
            raise RuntimeError("nonfinite coarse-to-fine encoder VJP")
        if args.steps: optimizer.step()
        if step:
            forward_times.append(middle-began)
            backward_times.append(ended-middle)
        if args.steps and (step==0 or (step+1)%25==0 or step+1==args.steps):
            records.append({"step":step+1,
                            "sampled_image_mse_before_update":float(loss),
                            "elapsed_seconds":time.perf_counter()-began_all})
    training_seconds=time.perf_counter()-began_all
    peak=(torch.cuda.max_memory_allocated(device)
          if device.type=="cuda" else None)
    final=evaluate(heldout,evaluation_count) if args.steps else initial
    if args.save_state:
        torch.save({"encoder":encoder.state_dict(),"args":settings,
                    "fine_side":fine_side,"fine_passes":args.passes},
                   args.save_state)
    print(json.dumps({
        "method":"A8_exact_nested_P1_coarse_to_fine_feedback",
        "coarse_side":coarse_side,"fine_side":fine_side,
        "control_vertices":fine_side**2,
        "control_faces":2*(fine_side-1)**2,
        "image_side":image_side,"image_queries":image_side**2,
        "passes":args.passes,"fine_gain":args.gain,
        "coarse_initial_cap":args.coarse_initial_cap,
        "fine_qc_cap":args.fine_qc_cap,
        "use_module":args.use_module,
        "steps":args.steps,"batch":1,"device":str(device),
        "evaluation_kind":"photo_content" if args.photo_variants else "high32_synthetic",
        "test_seed":args.test_seed,
        "photo_names":photo_names,"photo_source_indices":photo_indices,
        "initial_heldout":initial,"final_heldout":final,
        "minimum_training_signed_area_ratio":minimum_train_area,
        "median_full_forward_seconds_after_first":statistics.median(forward_times),
        "median_full_vjp_seconds_after_first":statistics.median(backward_times),
        "training_seconds":training_seconds,
        "peak_cuda_allocated_bytes":peak,"records":records,
    },sort_keys=True,separators=(",",":")))


if __name__=="__main__":
    main()
