"""ONE fixed label-free similarity-versus-affine restriction diagnostic.

Frozen image-machine confidences are not anatomical truth. No landmarks,
oracle maps, rematching, registration, or production archive mutation.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path

import torch  # Established runtime imports torch before NumPy LAPACK.
import numpy as np

from tools.coordinated_miit_score import load_manifest
from tools.coordinated_lung_all20_score import _affine


MODELS=("similarity","affine")


def spatial_parity(q):
    q=np.asarray(q,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=2 or not np.isfinite(q).all() or ((q<0)|(q>1)).any():
        raise ValueError("finite normalized source points required")
    bins=np.minimum(3,np.floor(4*q).astype(np.int64))
    return bins.sum(-1)%2


def design_matrix(q,model):
    if model not in MODELS:raise ValueError("similarity or affine model required")
    q=np.asarray(q,dtype=np.float64);x,y=q.T
    design=np.zeros((2*len(q),4 if model=="similarity" else 6))
    if model=="similarity":
        design[0::2,:2]=np.stack((x,-y),-1);design[1::2,:2]=np.stack((y,x),-1)
    else:
        design[0::2,:2]=q;design[1::2,2:4]=q
    design[0::2,-2]=1.;design[1::2,-1]=1.
    return design


def loss_gradient(beta,design,y,weights):
    """Stable pseudo-Huber and its ordinary analytical coefficient derivative."""
    residual=(64*(design@beta-y.ravel())).reshape(-1,2)
    square=np.sum(residual*residual,-1);scale=np.sqrt(1+square)
    normalized=weights/weights.sum()
    loss=np.sum(normalized*square/(scale+1))
    gradient=64*design.T@((normalized/scale)[:,None]*residual).ravel()
    return float(loss),gradient


def positive_determinants(matrix,offset=None):
    def positive(value):
        if not np.isfinite(value).all():return False
        a,b,c,d=(Fraction.from_float(float(x)) for x in value.ravel())
        return a*d-b*c>0
    with np.errstate(over="ignore",invalid="ignore"):stored=np.asarray(matrix,dtype=np.float32)
    result=dict(determinant_float64_positive=positive(matrix),determinant_float32_positive=positive(stored),
        determinant_float64=float(np.linalg.det(matrix)),stored_matrix_float32=stored.astype(float).tolist())
    if offset is not None:
        with np.errstate(over="ignore",invalid="ignore"):stored_offset=np.asarray(offset,dtype=np.float32)
        result.update(offset_float64_finite=bool(np.isfinite(offset).all()),offset_float32_finite=bool(np.isfinite(stored_offset).all()),
            stored_offset_float32=stored_offset.astype(float).tolist())
    return result


def fit_robust_affine(q,y,weights,*,model="similarity",max_iterations=1000):
    """IRLS weighted least squares: no ridge, determinant projection, or fallback.

    The production cap is1000 and stationarity tolerance is1e-8. The optional
    smaller cap exists only for failure fixtures; run() never changes it.
    """
    q=np.asarray(q,dtype=np.float64);y=np.asarray(y,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    spatial_parity(q)
    if (y.shape!=q.shape or weights.shape!=(len(q),) or len(q)==0 or not np.isfinite(y).all()
            or not np.isfinite(weights).all() or ((weights<0)|(weights>1)).any() or not (weights>0).any()
            or isinstance(max_iterations,bool) or not isinstance(max_iterations,int) or not 1<=max_iterations<=1000):
        raise ValueError("finite matching confidence arrays with positive-weight support and declared cap required")
    design=design_matrix(q,model);dimension=design.shape[1]
    iterations=0;solves=0;increase=0.;rank=0;design_singular=[]
    def solve(effective):
        nonlocal solves,rank,design_singular
        root=np.sqrt(effective/effective.sum()).repeat(2)
        beta,_,rank,singular=np.linalg.lstsq(design*root[:,None],y.ravel()*root,rcond=None)
        design_singular=singular.tolist()
        solves+=1
        if rank!=dimension or not np.isfinite(beta).all():raise ValueError("rank-deficient or nonfinite weighted least squares")
        return beta
    try:
        beta=solve(weights)
        loss,gradient=loss_gradient(beta,design,y,weights)
        while np.isfinite(loss) and np.isfinite(gradient).all() and np.max(np.abs(gradient))>1e-8 and iterations<max_iterations:
            residual=(64*(design@beta-y.ravel())).reshape(-1,2)
            effective=weights/np.sqrt(1+np.sum(residual*residual,-1))
            if not np.isfinite(effective).all() or (effective<0).any() or not (effective>0).any():raise ValueError("nonfinite IRLS majorization weights")
            beta=solve(effective);old_loss=loss
            loss,gradient=loss_gradient(beta,design,y,weights);iterations+=1
            increase=max(increase,loss-old_loss)
        if model=="similarity":matrix=np.array([[beta[0],-beta[1]],[beta[1],beta[0]]])
        else:matrix=beta[:4].reshape(2,2)
        offset=beta[-2:];determinants=positive_determinants(matrix,offset)
        converged=bool(np.isfinite(loss) and np.isfinite(gradient).all() and np.max(np.abs(gradient))<=1e-8)
        valid=(converged and determinants["determinant_float64_positive"] and determinants["determinant_float32_positive"]
            and determinants["offset_float64_finite"] and determinants["offset_float32_finite"])
        result=dict(status="ok" if valid else "failed",error=None if valid else "stationarity/nonfinite/positive stored determinant requirement failed",
            model=model,coefficients=beta.tolist(),matrix=matrix.tolist(),offset=offset.tolist(),iterations=iterations,
            weighted_ls_solves=solves,rank=int(rank),required_rank=dimension,converged=converged,
            gradient_inf=float(np.max(np.abs(gradient))),training_robust_loss=loss,
            maximum_irls_loss_increase=increase,singular_values=np.linalg.svd(matrix,compute_uv=False).tolist(),
            design_singular_values=design_singular,
            design_singular_values_scope="last normalized weighted-LS design; positive weights determine rank, zero-confidence rows retained",
            **determinants)
        result["training"]=_score(q,y,weights,result)
        return result
    except (ValueError,np.linalg.LinAlgError) as error:
        return dict(status="failed",model=model,error=str(error),iterations=iterations,weighted_ls_solves=solves,
            rank=int(rank),required_rank=dimension,converged=False,design_singular_values=design_singular,
            training_support=_support(q,weights))


def _support(q,weights):
    bins=np.minimum(3,np.floor(4*q).astype(np.int64))
    return dict(count=len(q),confidence_denominator=float(weights.sum()),positive_confidence_points=int((weights>0).sum()),
        occupied_4x4_cells=np.unique(bins,axis=0).tolist(),
        positive_confidence_occupied_4x4_cells=np.unique(bins[weights>0],axis=0).tolist())


def _score(q,y,weights,fit):
    if len(q)==0 or not np.isfinite(weights.sum()) or weights.sum()<=0:
        raise ValueError("held-out/training weighted loss requires positive confidence mass")
    residual=q@np.asarray(fit["matrix"]).T+fit["offset"]-y
    square=np.sum((64*residual)**2,-1)
    errors=512*np.linalg.norm(residual,axis=-1)
    return dict(robust_loss=float(np.sum(weights*square/(np.sqrt(1+square)+1))/weights.sum()),support=_support(q,weights),
        count=len(q),confidence_denominator=float(weights.sum()),
        canvas_pixels=dict(mean=float(errors.mean()),p90=float(np.percentile(errors,90)),maximum=float(errors.max())),
        errors_canvas_px=errors.tolist())


def _pending_fold(parity):
    return dict(training_parity=parity,heldout_parity=1-parity,models={model:dict(status="pending") for model in MODELS},heldout={},pass_=False)


def compare_folds(q,y,weights,ids=None):
    ids=np.arange(len(q)) if ids is None else np.asarray(ids)
    parity=spatial_parity(q);folds=[]
    for train in (0,1):
        selected=parity==train;held=~selected;fold=_pending_fold(train)
        fold.update(training_ids=ids[selected].tolist(),heldout_ids=ids[held].tolist())
        for model in MODELS:
            try:
                fit=fit_robust_affine(q[selected],y[selected],weights[selected],model=model)
                fit["point_ids"]=fold["training_ids"];fold["models"][model]=fit
                if fit["status"]=="ok":
                    if not held.any():raise ValueError("empty predeclared held-out support")
                    fold["heldout"][model]=_score(q[held],y[held],weights[held],fit)
            except (ValueError,np.linalg.LinAlgError) as error:fold["models"][model]=dict(status="failed",error=str(error))
        passed=(all(fold["models"][m]["status"]=="ok" for m in MODELS)
            and set(fold["heldout"])==set(MODELS) and fold["heldout"]["affine"]["robust_loss"]<fold["heldout"]["similarity"]["robust_loss"])
        fold.pop("pass_");fold["pass"]=bool(passed);folds.append(fold)
    return dict(folds=folds,**{"pass":all(f["pass"] for f in folds)})


def load_frozen_matches(row,directory):
    resolve=lambda value:Path(value) if Path(value).is_absolute() else directory/value
    if row.get("raw_matches",{}).get("status")!="ok":raise ValueError("original successful raw machine matches required")
    matrix,offset=_affine(resolve(row["affine"]))
    if matrix.dtype!=np.float32 or offset.dtype!=np.float32:raise ValueError("original float32 initializer required")
    if matrix[0,0]!=matrix[1,1] or matrix[0,1]!=-matrix[1,0]:raise ValueError("original initializer must be declared similarity")
    matrix=matrix.astype(np.float64);offset=offset.astype(np.float64)
    raw=json.loads(resolve(row["raw_matches"]["path"]).read_text(encoding="utf-8"))
    basename=lambda value:Path(str(value).replace("\\","/")).name
    if (raw.get("status")!="ok" or raw.get("targets_manual_landmarks_or_dense_teacher_loaded") is not False
            or raw.get("global_geometric_ransac_used") is not False or raw.get("image_side")!=512
            or any(basename(raw[key])!=basename(row[key]) for key in ("fixed","moving"))
            or not np.array_equal(raw["post_affine_matrix"],matrix) or not np.array_equal(raw["post_affine_offset"],offset)):
        raise ValueError("original raw provenance/raster/affine mismatch")
    q=np.asarray(raw["source_points_unit"],dtype=np.float64);p=np.asarray(raw["target_points_unit"],dtype=np.float64)
    confidence=np.asarray(raw["confidence"],dtype=np.float64);spatial_parity(q);spatial_parity(p)
    if p.shape!=q.shape or confidence.shape!=(len(q),) or not np.isfinite(confidence).all() or ((confidence<0)|(confidence>1)).any():
        raise ValueError("invalid original correspondence/confidence arrays")
    world=p@matrix.T+offset;eligible=((world>=0)&(world<=1)).all(-1)
    return q[eligible],world[eligible],confidence[eligible],dict(raw_matches=len(q),static_world_excluded=int((~eligible).sum()),
        zero_confidence_eligible=int((eligible&(confidence==0)).sum()),used_matches=int(eligible.sum()),point_ids=np.flatnonzero(eligible).tolist(),
        original_matrix=matrix.tolist(),original_offset=offset.tolist(),raw_match_path=str(resolve(row["raw_matches"]["path"])))


def _json_safe(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_safe(v) for v in value]
    return value


def run(predictions,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    manifest,directory=load_manifest(predictions)
    rows=[dict(name=row["name"],status="pending",folds=[_pending_fold(i) for i in (0,1)],all_point_fits=None) for row in manifest["rows"]]
    report=dict(protocol="ONE fixed spatial-parity similarity/full-affine restriction probe, previously viewed ONE MIIT specimen",
        predictions=str(predictions),annotations_read=False,maps_written=False,matcher_rerun=False,probe_complete=False,
        previously_viewed_label_informed_development=True,
        development_scope="probe opens NO annotations, but this branch was chosen after known registration errors and the label-oracle witness; NOT untouched/blind validation",
        loss="confidence-normalized sqrt(1+||64(Aq+b-y)||^2)-1",stationarity_inf_tolerance=1e-8,max_irls_iterations=1000,
        split="(floor(4*qx)+floor(4*qy))%2; endpoint1 clipped into bin3; each parity held out once",
        frozen_evidence="original world y=p@Aold.T+bold; original confidence and original-world eligibility only",
        advancement="ALL six held-out full-affine losses STRICTLY below refitted similarity, all fits full-rank/converged/positive in float64 and stored float32",
        denominator_pairs=3,rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    def persist():output.write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    persist();inputs=[]
    for row,source in zip(rows,manifest["rows"],strict=True):
        try:
            q,y,w,metadata=load_frozen_matches(source,directory);inputs.append((q,y,w))
            result=compare_folds(q,y,w,metadata["point_ids"])
            row.update(result,status="ok" if result["pass"] else "failed",frozen_matches=metadata)
        except (ValueError,KeyError,OSError,np.linalg.LinAlgError) as error:
            inputs.append(None);row.update(status="failed",error=f"{type(error).__name__}: {error}")
            for fold in row["folds"]:
                fold.pop("pass_",None);fold["pass"]=False
                fold["models"]={m:dict(status="failed",error=row["error"]) for m in MODELS}
        persist()
    passed=all(row["status"]=="ok" and row["pass"] for row in rows)
    report["pass"]=passed;report["production_refits_eligible"]=False
    if passed:
        for row,data in zip(rows,inputs,strict=True):
            row["all_point_fits"]={m:fit_robust_affine(*data,model=m) for m in MODELS}
            for fit in row["all_point_fits"].values():
                if fit["status"]=="ok":
                    pnew=np.linalg.solve(np.asarray(fit["matrix"]),(data[1]-fit["offset"]).T).T
                    outside=((pnew<0)|(pnew>1)).any(-1)
                    fit["reexpressed_target_domain"]=dict(outside_unit_count=int(outside.sum()),
                        incompatible_original_match_ids=np.asarray(row["frozen_matches"]["point_ids"])[outside].tolist(),
                        min_xy=pnew.min(0).tolist(),max_xy=pnew.max(0).tolist(),
                        scope="diagnostic ONLY; all original eligible points retained, no rematching/drop/production loader change")
            persist()
        report["production_refits_eligible"]=all(row["all_point_fits"][m]["status"]=="ok" for row in rows for m in MODELS)
    report["probe_complete"]=True;persist();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();report=run(args.predictions,args.output)
    print(json.dumps({k:report[k] for k in ("pass","production_refits_eligible","denominator_pairs")}))


if __name__=="__main__":main()
