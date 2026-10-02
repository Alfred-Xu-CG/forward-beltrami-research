"""Explicit LABEL-ORACLE sparse capacity witness, NEVER registration/teacher.

One fixed7-to8 development pair and one predeclared300-gradient budget.
Only a valid map fitting EVERY107point within1canvaspx establishes this witness;
other outcomes are inconclusive about capacity, not impossibility evidence.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from tools.coordinated_miit_score import load_manifest,validate_layout,read_points,metrics
from tools.coordinated_lung_all20_score import _affine,_safe_map
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit
from tools.coordinated_real_case import Evidence,load_registration_evidence,load_image_matches,corner_symmetric_dirichlet
from qcopt.neural_bijection.dense.coordinated_fixed_sampling import FrozenP1Evaluator
from qcopt.neural_bijection.dense.coordinated_stage_cache import FrozenAnchorCoordinatedUpdate
from qcopt.neural_bijection.dense.coordinated_update import interpolate_proposal
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants,StaggeredPatchQ1Layer
from tools.coordinated_control_capacity import f2_coverage,decode as decode_control
from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def _resolve(directory,path):
    path=Path(path);return path if path.is_absolute() else directory/path


class OracleObjective:
    """Differentiable vertex-only physical-canvas least squares at fixed IDs."""
    def __init__(self,vertices,matrix,offset,query,target,side=512):
        if any(t.requires_grad for t in (matrix,offset,query,target)):
            raise ValueError("oracle affine/queries/targets must be fixed constants")
        if (matrix.shape!=(2,2) or offset.shape!=(2,) or
                any(t.dtype!=torch.float64 or t.device!=vertices.device for t in (matrix,offset,query,target))
                or not bool(torch.isfinite(matrix).all() and torch.isfinite(offset).all() and torch.linalg.det(matrix)>0)):
            raise ValueError("same-device float64 positive affine and fixed coordinate arrays required")
        if query.ndim!=2 or query.shape!=target.shape or query.shape[-1]!=2 or len(query)==0:
            raise ValueError("same nonempty Nx2 query/target arrays required")
        if not bool(torch.isfinite(target).all()):raise ValueError("finite ALL targets required")
        self.sampler=FrozenP1Evaluator(*vertices.shape[1:3],query[None],"ac")
        self.matrix,self.offset,self.target=matrix.clone(),offset.clone(),target.clone()
        self.side=side

    def errors(self,vertices):
        return self.side*(self.sampler(vertices)[0]@self.matrix.T+self.offset-self.target)

    def __call__(self,vertices):return self.errors(vertices).square().sum(-1).mean()


def _geometry(vertices,reference,eta):
    qref=q1_corner_determinants(reference.double())
    ratios=q1_corner_determinants(vertices.double())/qref
    finite=bool(torch.isfinite(vertices).all() and torch.isfinite(ratios).all() and (qref>0).all())
    boundary=all(torch.equal(vertices[index],reference[index]) for index in
        ((slice(None),0,slice(None)),(slice(None),-1,slice(None)),
         (slice(None),slice(None),0),(slice(None),slice(None),-1)))
    return dict(valid=finite and boundary and bool((ratios>eta).all()),
        minimum_normalized_corner=float(ratios.amin()),exact_boundary=boundary,finite=finite)


def fit_oracle(vertices,reference,objective,*,levels=(17,33,65,129,257),inner_steps=30,learning_rate=.004,eta=.001,method="analytic",patch_cells=8):
    """No derivative through iterations; smaller explicit schedules are TEST only."""
    if (vertices.requires_grad or reference.requires_grad or vertices.dtype!=torch.float64 or reference.dtype!=torch.float64
            or vertices.device!=reference.device or vertices.shape!=reference.shape
            or vertices.ndim!=4 or vertices.shape[0]!=1 or vertices.shape[-1]!=2 or min(vertices.shape[1:3])<3):
        raise ValueError("constant matching float64 vertex/reference tables required")
    if (not levels or any(isinstance(n,bool) or not isinstance(n,int) or n<3 for n in levels)
            or isinstance(inner_steps,bool) or not isinstance(inner_steps,int) or inner_steps<1
            or not np.isfinite(learning_rate) or learning_rate<=0):raise ValueError("positive declared levels/steps/rate required")
    if method not in ("analytic","f2"):raise ValueError("analytic or corrected f2 oracle only")
    if not callable(getattr(objective,"errors",None)):raise ValueError("oracle needs separate physical error query for stage metrics")
    if method=="f2":
        if (vertices.shape[1]!=vertices.shape[2] or isinstance(patch_cells,bool) or not isinstance(patch_cells,int)
                or patch_cells<2 or patch_cells%2 or patch_cells>vertices.shape[1]-1):raise ValueError("F2 requires square grid/even admissible patch size")
        axis=torch.linspace(0,1,vertices.shape[1],device=vertices.device,dtype=torch.float64)
        y,x=torch.meshgrid(axis,axis,indexing="ij")
        if not torch.equal(reference,torch.stack((x,y),-1)[None]):raise ValueError("F2 floor calibration requires actual unit-grid material reference")
    if not _geometry(vertices,reference,eta)["valid"]:raise ValueError("strictly legal initial map required")
    current=vertices.clone();stages=[];trace=[];failures=[]
    counts=dict(gradient_steps=0,decoder_trials=0,oracle_objective_evaluations=0,actual_geometry_checks=1,failed_trials=0,
        geometry_passes=0,accepted_metric_queries=0,threshold_certification_calls=0)
    def evaluate(y):counts["oracle_objective_evaluations"]+=1;return objective(y)
    initial=float(evaluate(current));tick=time.perf_counter()
    first_certified_threshold=None;first_threshold_vertices=None
    for cycle in range(2 if method=="f2" else 1):
        schedule=[(level,direction) for level in levels for direction in (((1.,0.),(0.,1.)) if method=="analytic" else (None,))]
        for level,direction in schedule:
            stage_tick=time.perf_counter()
            anchor=current.detach();anchor_loss=float(evaluate(anchor))
            best_loss,best_map=anchor_loss,anchor.clone();best_diagnostics={}
            channels=1 if method=="analytic" else 2
            coefficients=torch.nn.Parameter(torch.zeros(1,channels,level-2,level-2,dtype=current.dtype,device=current.device))
            lr=learning_rate*(levels[0]-1)/(level-1)
            optimizer=torch.optim.Adam([coefficients],lr=lr)
            if method=="analytic":
                layer=FrozenAnchorCoordinatedUpdate(anchor,reference=reference,direction=direction,mode="analytic",minimum_jacobian=eta,theta=.95)
                passes=1;coverage=None
            else:
                layer=StaggeredPatchQ1Layer(vertices.shape[1],patch_cells,proposal_mode="fixed_h",raw_span=.5,
                    safety_fraction=.75,minimum_jacobian=eta,accepted_gain=1.,floor_safety_fraction=.95).to(current.device)
                coverage=f2_coverage(layer,vertices.shape[1],current.dtype);passes=len(layer.passes)
            stage_gradients=0;stage_trials=0
            for step in range(inner_steps+1):
                optimizer.zero_grad(set_to_none=True)
                if method=="analytic":
                    coarse=F.pad(coefficients,(1,1,1,1))[:,0]
                    proposal=interpolate_proposal(coarse,tuple(current.shape[1:3]))
                    result=layer(proposal,validate=False);candidate=result.vertices
                    diagnostics=dict(scale=float(result.scale.detach().min()),gauge=float(result.gauge.detach().max()))
                else:
                    candidate=decode_control(layer,"f2",anchor,coefficients,coverage);diagnostics={}
                counts["decoder_trials"]+=1;stage_trials+=1
                counts["geometry_passes"]+=passes
                geometry=_geometry(candidate.detach(),reference,eta);counts["actual_geometry_checks"]+=1
                loss=evaluate(candidate);value=float(loss.detach())
                diagnostics.update(geometry)
                if not geometry["valid"] or not np.isfinite(value):
                    counts["failed_trials"]+=1;failures.append(dict(cycle=cycle,level=level,direction=direction,step=step,
                        reason="nonfinite oracle loss or actual rounded geometry",oracle_loss=value,**diagnostics));break
                if value<best_loss:best_loss,best_map,best_diagnostics=value,candidate.detach().clone(),diagnostics
                trace.append(dict(cycle=cycle,level=level,direction=direction,step=step,oracle_loss=value,**diagnostics))
                if step<inner_steps:
                    loss.backward()
                    if coefficients.grad is None or not bool(torch.isfinite(coefficients.grad).all()):
                        counts["failed_trials"]+=1;failures.append(dict(cycle=cycle,level=level,direction=direction,step=step,reason="nonfinite oracle gradient"));break
                    counts["gradient_steps"]+=1;stage_gradients+=1;optimizer.step()
            actual=_geometry(best_map,reference,eta);counts["actual_geometry_checks"]+=1
            if not actual["valid"]:raise RuntimeError("oracle best-map strict acceptance failure")
            current=best_map
            with torch.no_grad():errors=objective.errors(current).square().sum(-1).sqrt()
            counts["accepted_metric_queries"]+=1
            error_summary=dict(mean=float(errors.mean()),p90=float(torch.quantile(errors,.9)),maximum=float(errors.max()))
            if first_certified_threshold is None and error_summary["maximum"]<=1.:
                candidate_metric_elapsed=time.perf_counter()-tick;certificate_tick=time.perf_counter()
                # Ordinary in-memory NPZ input to the EXISTING exact-sign checker,
                # not a new certificate or a per-stage disk artifact.
                threshold_snapshot=current.detach().cpu().clone()
                buffer=io.BytesIO()
                np.savez(buffer,vertices=threshold_snapshot.numpy(),boundary_reference=reference.cpu().numpy())
                buffer.seek(0)
                try:certificate=certify_q1_binary_map(buffer)
                finally:buffer.close()  # Only the vertex clone remains resident.
                counts["threshold_certification_calls"]+=1
                certification_seconds=time.perf_counter()-certificate_tick
                if certificate["valid"]:
                    first_threshold_vertices=threshold_snapshot
                    first_certified_threshold=dict(stage=len(stages),cumulative_gradients=counts["gradient_steps"],
                        cumulative_trials=counts["decoder_trials"],elapsed_fit_seconds=time.perf_counter()-tick,
                        candidate_metric_elapsed_fit_seconds=candidate_metric_elapsed,
                        certification_seconds=certification_seconds,
                        certification_scope="CPU snapshot copy/in-memory NPZ serialization/filtered-sign check INCLUDED in elapsed_fit_seconds",
                        errors=error_summary,actual_geometry=actual,saved_binary_certificate=certificate)
            stages.append(dict(cycle=cycle,level=level,direction=direction,physical_lr=lr,anchor_oracle_loss=anchor_loss,
                accepted_oracle_loss=best_loss,gradient_steps=stage_gradients,decoder_trials=stage_trials,
                cumulative_gradients=counts["gradient_steps"],cumulative_trials=counts["decoder_trials"],
                stage_seconds=time.perf_counter()-stage_tick,elapsed_fit_seconds=time.perf_counter()-tick,
                accepted_error_canvas_px=error_summary,scalar_parameter_count=coefficients.numel(),
                raw_parameter_channels=channels,geometry_passes_per_trial=passes,
                **actual,**{k:v for k,v in best_diagnostics.items() if k in ("scale","gauge")}))
    if current.device.type=="cuda":torch.cuda.synchronize(current.device)
    elapsed=time.perf_counter()-tick
    final=float(evaluate(current));counts["actual_geometry_checks"]+=1
    return dict(vertices=current,initial_oracle_loss=initial,final_oracle_loss=final,stages=stages,trace=trace,
        failures=failures,counts=counts,fit_seconds=elapsed,
        method=method,first_certified_max_error_le_one=first_certified_threshold,
        first_threshold_vertices=first_threshold_vertices,
        threshold_vertex_snapshot_bytes=0 if first_threshold_vertices is None else first_threshold_vertices.numel()*first_threshold_vertices.element_size(),
        threshold_snapshot_memory_scope="ONE CPU vertex clone retained, no all-stage snapshot history",
        total_stage_scalar_parameter_count=sum(stage["scalar_parameter_count"] for stage in stages),
        metric_timing_scope="ten accepted physical-error queries plus first eligible in-memory exact-sign certificate INCLUDED in fit/stage timings; historical analytic timing is not the paired baseline",
        budget_complete=counts["gradient_steps"]==len(levels)*2*inner_steps and counts["failed_trials"]==0,
        final_geometry=_geometry(current,reference,eta))


def _json_safe(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {key:_json_safe(v) for key,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_safe(v) for v in value]
    return value


def _original_evidence(source,directory,config,a,b,device):
    if (config.get("mind_frame","original")!="original" or config.get("mind_order","transport")!="transport"
            or config.get("loss")!="mind" or config.get("image_weight",1.)!=1. or config.get("image_side")!=512):
        raise ValueError("unchanged ORIGINAL512 E1 required, not retired affine-frame evidence")
    f,m,mask,metadata=load_registration_evidence(_resolve(directory,source["fixed"]),_resolve(directory,source["moving"]),512,
        preprocessing=config.get("preprocessing","raw_inverted"),device=device,dtype=torch.float32)
    matches=None;weight=config.get("match_weight",0.)
    if weight:
        matches,_=load_image_matches(_resolve(directory,source["raw_matches"]["path"]),a,b,
            fixed_path=_resolve(directory,source["fixed"]),moving_path=_resolve(directory,source["moving"]),
            image_side=512,device=device,dtype=torch.float64,robust_scale=config["match_robust_scale"])
        if config.get("match_p1_sampling","existing")=="frozen":matches.prepare_fixed_p1_sampling(257,257,"ac")
    e=Evidence(f,m,torch.from_numpy(a).to(device=device,dtype=torch.float64),torch.from_numpy(b).to(device=device,dtype=torch.float64),
        config["loss"],config["strain_weight"],config["oob_weight"],config["shape_weight"],fixed_mask=mask,
        interpolation="p1_ac",strain_model=config["strain_model"],matches=matches,match_weight=weight,
        joint_prior_backend=config.get("joint_prior_backend","eager"))
    if config.get("p1_sampling","existing")=="frozen":e.prepare_fixed_p1_sampling(257,257,dtype=torch.float64,device=device)
    return e,metadata


def run(args):
    overall_tick=time.perf_counter()
    if getattr(args,"pair_name","miit_7_to_8")!="miit_7_to_8":raise ValueError("one predeclared7-to8 witness only")
    method=getattr(args,"method","analytic")
    if method not in ("analytic","f2"):raise ValueError("analytic or f2 oracle method required")
    output=Path(args.output)
    threshold_output=output.with_name(output.stem+"_first_threshold.npz")
    if output.suffix!=".npz" or output.exists() or output.with_suffix(".json").exists() or threshold_output.exists():raise ValueError("new separate label_oracle .npz/report/optional threshold snapshot required")
    manifest,directory=load_manifest(args.predictions)  # BEFORE labels.
    source=manifest["rows"][1];record=source["methods"]["analytic"]
    if record["status"]!="ok":raise ValueError("original successful analytic map required")
    a,b=_affine(_resolve(directory,source["affine"]))
    if a.dtype!=np.float32 or b.dtype!=np.float32:raise ValueError("original float32 affine archive required, no new rounding/refit")
    _safe_map(record,directory,a,b)
    original_report=json.loads(_resolve(directory,record["report"]).read_text(encoding="utf-8"));config=original_report["configuration"]
    if (config.get("grid_side")!=257 or config.get("precision")!="float64" or config.get("image_precision")!="float32" or config.get("interpolation")!="p1_ac"
            or config.get("minimum_jacobian")!=.001):raise ValueError("archived257float64P1ac eta.001 map required")
    layout=validate_layout(json.loads(_resolve(directory,source["layout"]).read_text(encoding="utf-8")))
    points={}
    for key in ("fixed","moving"):
        section=source[key+"_section"];root=Path(args.source_data)/str(section)
        with Image.open(root/"images"/"image.tif") as image:
            if list(image.size)!=layout[key]["original_wh"] or image.getexif().get(274,1)!=1:raise ValueError("native layout/orientation mismatch")
        points[key]=read_points(root/"landmarks"/f"{section:02d}.csv",layout[key]["original_wh"],allow_missing_inf=True)
    nominal=sorted(points["fixed"])
    if set(nominal)!=set(points["moving"]):raise ValueError("same124 nominal IDs required, no intersection dropping")
    absent={key:[i for i in nominal if np.isposinf(points[key][i]).all()] for key in points}
    ids=[i for i in nominal if i not in set(absent["fixed"])|set(absent["moving"])]
    if len(ids)!=107:raise ValueError("exact original107 annotation-only eligible IDs required")
    fixed_px=np.stack([points["fixed"][i] for i in ids]);moving_px=np.stack([points["moving"][i] for i in ids])
    q=original_pixel_to_canvas_unit(fixed_px,layout["fixed"],512);target=original_pixel_to_canvas_unit(moving_px,layout["moving"],512)
    device=torch.device(args.device);torch.set_num_threads(getattr(args,"threads",2))
    with np.load(_resolve(directory,record["output"]),allow_pickle=False) as saved:
        vertices=torch.from_numpy(saved["vertices"].copy()).to(device);reference=torch.from_numpy(saved["boundary_reference"].copy()).to(device)
    setup_tick=time.perf_counter();evidence,preprocessing=_original_evidence(source,directory,config,a,b,device)
    objective=OracleObjective(vertices,torch.from_numpy(a).to(device=device,dtype=torch.float64),torch.from_numpy(b).to(device=device,dtype=torch.float64),
        torch.tensor(q,device=device,dtype=torch.float64),torch.tensor(target,device=device,dtype=torch.float64))
    def evaluate_e1(y):
        with torch.no_grad():total,parts=evidence(y)
        return dict(total=float(total),**{k:float(v) for k,v in parts.items()})
    def distortion(y):
        d=(y-reference).square().sum(-1).sqrt()
        return dict(vertex_displacement_rms_canvas_px=float(d.square().mean().sqrt()*512),
            vertex_displacement_max_canvas_px=float(d.max()*512),arap=float(p1_arap_energy(y,"ac")),
            corner_symmetric_dirichlet=float(corner_symmetric_dirichlet(y)))
    start_e1=evaluate_e1(vertices);start_distortion=distortion(vertices)
    if device.type=="cuda":torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    fit_start_allocated_bytes=torch.cuda.memory_allocated(device) if device.type=="cuda" else None
    setup_seconds=time.perf_counter()-setup_tick
    result=fit_oracle(vertices,reference,objective,method=method)
    peak=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
    threshold_vertices=result.pop("first_threshold_vertices")
    final=result.pop("vertices");end_e1=evaluate_e1(final);end_distortion=distortion(final)
    with torch.no_grad():
        start_mapped=(objective.sampler(vertices)[0]@objective.matrix.T+objective.offset).cpu().numpy()
        end_mapped=(objective.sampler(final)[0]@objective.matrix.T+objective.offset).cpu().numpy()
    start_errors=metrics(start_mapped,moving_px,layout["moving"],ids);end_errors=metrics(end_mapped,moving_px,layout["moving"],ids)
    # These two exported error tables are separate sampler calls, not J calls
    # and not the accepted-stage metric instrumentation inside fit_oracle.
    result["counts"]["export_metric_queries"]=2
    result["counts"]["nonobjective_metric_queries"]=result["counts"]["accepted_metric_queries"]+2
    output.parent.mkdir(parents=True,exist_ok=True)
    serialization_tick=time.perf_counter()
    np.savez(output,vertices=final.cpu().numpy(),boundary_reference=reference.cpu().numpy(),post_affine_matrix=a,
        post_affine_offset=b,interpolation=np.asarray("p1_ac"),label_oracle=np.asarray(True),landmarks_used=np.asarray(True))
    serialization_seconds=time.perf_counter()-serialization_tick;certification_tick=time.perf_counter()
    certificate=certify_q1_binary_map(output)
    certification_seconds=time.perf_counter()-certification_tick
    threshold_serialization_seconds=0.
    if threshold_vertices is not None:
        threshold_tick=time.perf_counter()
        np.savez(threshold_output,vertices=threshold_vertices.numpy(),boundary_reference=reference.cpu().numpy(),
            post_affine_matrix=a,post_affine_offset=b,interpolation=np.asarray("p1_ac"),label_oracle=np.asarray(True),landmarks_used=np.asarray(True))
        threshold_serialization_seconds=time.perf_counter()-threshold_tick
    success=bool(result["final_geometry"]["valid"] and certificate["valid"] and end_errors["canvas_pixels"]["maximum"]<=1.)
    report=dict(**result,label_oracle=True,landmarks_used=True,not_registration_initializer_or_training_teacher=True,
        protocol="ONE label-informed development capacity witness, NOT inference accuracy; failure is inconclusive about capacity",
        pair_name=source["name"],map_direction="fixed_canvas_to_moving_canvas",nominal_ids=nominal,available_ids=ids,unavailable=absent,
        configuration=dict(levels=[17,33,65,129,257],inner_steps=30,learning_rate=.004,lr_calibration="edge .004*16/(level-1)",
            method=method,cycles=1 if method=="analytic" else 2,theta=.95 if method=="analytic" else None,
            f2_controls=None if method=="analytic" else dict(patch_cells=8,proposal_mode="fixed_h",raw_span=.5,safety_fraction=.75,
                accepted_gain=1.,floor_safety_fraction=.95),minimum_jacobian=.001,objective="mean squared original-moving canvas errors",interpolation="p1_ac"),
        start_errors=start_errors,end_errors=end_errors,start_original_e1=start_e1,end_original_e1=end_e1,
        original_e1_evaluations=2,start_distortion=start_distortion,end_distortion=end_distortion,setup_seconds=setup_seconds,
        peak_allocated_bytes=peak,peak_scope="fit interval including resident full512evidence; not whole cold application",
        fit_start_allocated_bytes=fit_start_allocated_bytes,end_to_end_seconds=time.perf_counter()-overall_tick,
        fit_timing_scope="after initial oracle evaluation through final accepted stage, includes stage setup/trial/backward/checks; excludes final oracle/E1/distortion/export evaluations",
        serialization_seconds=serialization_seconds,certification_seconds=certification_seconds,
        first_threshold_output=str(threshold_output) if threshold_vertices is not None else None,
        threshold_serialization_seconds=threshold_serialization_seconds,
        nonfinite_reporting="rejected nonfinite diagnostics serialized as null, never a successful trial",
        saved_binary_certificate=certificate,capacity_witness_success=success,
        interpretation="valid simultaneous sparse witness at all107centers only" if success else "inconclusive capacity outcome after ONE fixed budget; no impossibility claim",
        original_prediction=str(_resolve(directory,record["output"])),original_configuration=config,preprocessing=preprocessing,
        units="512canvas and original moving native pixels; no physical spacing claim; CSV zero-based pixel-center convention")
    report=_json_safe(report)
    output.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("predictions","source_data","output"):parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    parser.add_argument("--pair-name",choices=("miit_7_to_8",),default="miit_7_to_8")
    parser.add_argument("--method",choices=("analytic","f2"),default="analytic")
    parser.add_argument("--device",default="cpu");parser.add_argument("--threads",type=int,default=2)
    report=run(parser.parse_args());print(json.dumps({k:report[k] for k in ("capacity_witness_success","counts","end_errors")},allow_nan=False))


if __name__=="__main__":main()
