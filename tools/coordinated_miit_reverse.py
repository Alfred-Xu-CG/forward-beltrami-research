"""All three fixed MIIT reverse tasks: fresh SG1/fusion300 and shared-A STANDARD.

Only the saved global affine is inverted; no optimized map, forward matches,
landmarks or cycle penalty enter prediction. One previously viewed specimen.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from tools.coordinated_conditional_pipeline import ResetAwarePeaks
from tools.coordinated_matchanything_batch import configuration, validate_table, validate_export
from tools.coordinated_match_fusion import fuse_tables

FORWARD = ((2,3),(7,8),(10,11))
REVERSE = tuple((f,m) for m,f in FORWARD)
METHODS = ("sg1", "fusion", "native_shared")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def inverse_affine(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    if a.shape!=(2,2) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.det(a)<=0:
        raise ValueError("finite positive original affine required")
    ai=np.linalg.inv(a);bi=-ai@b
    ar,br=ai.astype(np.float32),bi.astype(np.float32)
    if not np.isfinite(ar).all() or not np.isfinite(br).all() or np.linalg.det(ar.astype(float))<=0:
        raise ValueError("stored float32 inverse must remain finite and positive")
    corners=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    first=(corners@a.T+b)@ar.astype(float).T+br
    second=(corners@ar.astype(float).T+br)@a.T+b
    return ar,br,dict(original_matrix=a.tolist(),original_offset=b.tolist(),
        inverse_matrix_float64=ai.tolist(),inverse_offset_float64=bi.tolist(),
        stored_dtype="float32",stored_determinant=float(np.linalg.det(ar.astype(float))),
        reverse_after_forward_max_corner_canvas_pixels=float(np.linalg.norm(first-corners,axis=1).max()*512),
        forward_after_reverse_max_corner_canvas_pixels=float(np.linalg.norm(second-corners,axis=1).max()*512),
        optimized_map_inverted=False)


def prepare(case, source_data, output):
    cfg=case["original_configuration"]
    _,moving,_,fixed=case["name"].split("_")
    moving,fixed=int(moving),int(fixed)
    name=f"miit_{fixed}_to_{moving}"
    layout=read(Path(cfg["fixed"]).parent/(case["name"]+"_layout.json"))
    if layout["side"]!=512:
        raise ValueError("existing512 layout required")
    reverse_layout={**copy.deepcopy(layout),"fixed":copy.deepcopy(layout["moving"]),
                    "moving":copy.deepcopy(layout["fixed"]),"role_swap_of":case["name"]}
    paths={role:Path(output)/(name+"_"+role+"512.png") for role in ("fixed","moving")}
    native={role:Path(source_data)/str(section)/"images/image.tif"
            for role,section in (("fixed",moving),("moving",fixed))}
    for role,old_role in (("fixed","moving"),("moving","fixed")):
        with Image.open(cfg[old_role]) as image:
            if image.size!=(512,512) or image.mode!="RGB":
                raise ValueError("unchanged accepted RGB512 canvas required")
        with Image.open(native[role]) as image:
            if list(image.size)!=reverse_layout[role]["original_wh"] or image.getexif().get(274,1)!=1:
                raise ValueError("swapped native TIFF must agree with accepted layout")
        shutil.copyfile(cfg[old_role],paths[role])
        reverse_layout[role]["canvas_png"]=str(paths[role])
        reverse_layout[role]["source"]=str(native[role])
    with np.load(cfg["affine"],allow_pickle=False) as archive:
        a,b,diagnostics=inverse_affine(archive["post_affine_matrix"],archive["post_affine_offset"])
    affine=Path(output)/(name+"_affine.npz")
    np.savez(affine,post_affine_matrix=a,post_affine_offset=b)
    layout_path=Path(output)/(name+"_layout.json");save(layout_path,reverse_layout)
    source=copy.deepcopy(cfg)
    source.update(fixed=str(paths["fixed"]),moving=str(paths["moving"]),affine=str(affine),match_weight=.1)
    # Reuse the strict original recipe validator, not a new hyperparameter recipe.
    configuration(source,Path(output)/(name+"_sg.json"),Path(output)/"unused.npz")
    return dict(name=name,moving_section=fixed,fixed_section=moving,
        fixed=paths["fixed"].name,moving=paths["moving"].name,affine=affine.name,layout=layout_path.name,
        source_fixed=str(native["fixed"]),source_moving=str(native["moving"]),
        accepted512_layouts={r:reverse_layout[r] for r in ("fixed","moving")},
        shared_affine_matrix=a.tolist(),shared_affine_offset=b.tolist(),inverse_audit=diagnostics,
        source_forward_case=case["name"],source_forward_configuration=copy.deepcopy(cfg),
        original_configuration=source,map_direction="fixed_canvas_to_moving_canvas",input_status="ok")


def native_call(row, output, *, device):
    """Use the existing audited shared-native wrapper class and STANDARD config."""
    import sys
    import deeperhistreg as dhr
    from tools.coordinated_dhr_native_shared import shared_pipeline_class
    from tools.coordinated_dhr_released_baseline import configure
    transform=sys.modules[dhr.direct_registration.DeeperHistReg_FullResolution.__module__].w.tc_transform_to_tc_df
    cls=shared_pipeline_class(dhr.direct_registration.DeeperHistReg_FullResolution,transform)
    output=Path(output);output.mkdir()
    base=dhr.configs.default_initial_nonrigid();params=configure(base,device=device,output=output)
    save(output/"released_preset.json",base);save(output/"config.json",params)
    pipeline=None
    try:
        if device=="cuda":
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        pipeline=cls(params)
        pipeline.shared_initialization=(row["shared_affine_matrix"],row["shared_affine_offset"],row["accepted512_layouts"])
        pipeline.run_registration(row["source_moving"],row["source_fixed"],str(output))
        if device=="cuda":torch.cuda.synchronize()
        final=output/"released_dhr/Results_Final"
        for name in ("displacement_field.mha","postprocessing_params.json"):
            if not (final/name).is_file():raise FileNotFoundError(final/name)
        return dict(status="ok",field=(final/"displacement_field.mha").relative_to(output.parent).as_posix(),
            postprocessing_params=(final/"postprocessing_params.json").relative_to(output.parent).as_posix(),
            configuration=(output/"config.json").relative_to(output.parent).as_posix(),
            shared_affine_matrix=row["shared_affine_matrix"],shared_affine_offset=row["shared_affine_offset"],
            preprocessed_shape=list(pipeline.pre_source.shape),field_shape=list(pipeline.current_displacement_field.shape),
            initial_transform=pipeline.initial_transform.detach().cpu().tolist(),
            conjugate_theta64=pipeline.shared_theta64,initial_field_audit=pipeline.shared_initial_audit,
            native_padding_params=pipeline.padding_params,preprocessing_seconds=pipeline.preprocessing_time,
            initial_seconds=pipeline.initial_registration_time,nonrigid_seconds=pipeline.nonrigid_registration_time,
            registration_seconds=pipeline.total_registration_time,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device=="cuda" else None,
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if device=="cuda" else None,
            topology="no hard certificate; original native STANDARD nonrigid pipeline, only initializer replaced")
    finally:
        del pipeline


def run(fusion_predictions, source_data, output, source_root, checkpoint, *, cases=None,
        sg_extractor=None, model_factory=None, optimizer=None, native_runner=None, export_validator=None):
    if sg_extractor is None:
        import deeperhistreg  # noqa:F401
        import superpoint_superglue  # noqa:F401
        from tools.coordinated_image_matches import extract as sg_extractor
    if model_factory is None:
        from tools.coordinated_matchanything import FrozenMatchAnything as model_factory
    if optimizer is None:
        from tools.coordinated_real_case import optimize as optimizer
    native_runner=native_call if native_runner is None else native_runner
    export_validator=validate_export if export_validator is None else export_validator
    if cases is None:
        from tools.coordinated_joint_pose_batch import source_cases
        cases=[r for r in source_cases(fusion_predictions) if r["cohort"]=="miit"]
    if [r["name"] for r in cases]!=[f"miit_{m}_to_{f}" for m,f in FORWARD]:
        raise ValueError("all three original MIIT directions required in order")
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir()
    rows=[dict(name=f"miit_{m}_to_{f}",moving_section=m,fixed_section=f,input_status="pending",
               sg_status="pending",ma_status="pending",methods={method:dict(status="pending") for method in METHODS})
          for m,f in REVERSE]
    manifest=dict(protocol="all3 MIIT reverse directions; inverse saved affine only; SG1/fusion300/native STANDARD shared-A",
        cohort_size=3,attempt_denominator=9,annotations_read=False,prediction_complete=False,rows=rows,
        source_fusion_predictions=str(fusion_predictions),
        scope="one previously viewed specimen, direction robustness only; no blind/independent cohort claim",
        geometry_caveat="reverse boundary-fixed affine-postcomposed family need not equal inverse of forward admissible family",
        cost_scope="fresh reverse prep/matches plus six safe calls and three native calls; native uses original TIFF resolution, not equal raster budget",
        memory_scope="whole serial batch reset-aware allocator maxima; safe per-call report peaks exclude some setup, native peaks start before native pipeline construction",
        timing_scope="process imports excluded; stage batched SG3, MAsetup/extract3/close, safe6, native3; no independent cold/warm latency distribution")
    started=time.perf_counter()
    def persist():
        manifest["elapsed_seconds"]=time.perf_counter()-started
        save(output/"predictions.json",manifest)
    persist()  # ALL NINE pending methods precede image/model inference.
    for case,row in zip(cases,rows,strict=True):
        tick=time.perf_counter()
        try:row.update(prepare(case,source_data,output))
        except Exception as error:
            row.update(input_status="failed",input_error=f"{type(error).__name__}: {error}")
            row.update(sg_status="skipped",ma_status="skipped")
            for method in METHODS:row["methods"][method]=dict(status="failed",error=row["input_error"])
        row["preparation_seconds"]=time.perf_counter()-tick;persist()
    devices={r["original_configuration"]["device"] for r in rows if r["input_status"]=="ok"}
    if len(devices)>1:raise ValueError("one original execution device required")
    device=next(iter(devices),"cpu")
    def sync():
        if str(device).startswith("cuda"):torch.cuda.synchronize()
    with ResetAwarePeaks(device) as peaks:
        for row in rows:
            if row["input_status"]!="ok":continue
            tick=time.perf_counter();cfg=row["original_configuration"]
            row["sg_table"]=row["name"]+"_sg.json"
            try:
                result=sg_extractor(argparse.Namespace(fixed=Path(cfg["fixed"]),moving=Path(cfg["moving"]),
                    affine=Path(cfg["affine"]),output=output/row["sg_table"],image_side=512,device=device))
                sync()
                if result.get("status")!="ok":raise ValueError("insufficient fresh reverse SG; no fallback")
                row["sg_status"]="ok"
            except Exception as error:row.update(sg_status="failed",sg_error=f"{type(error).__name__}: {error}")
            row["sg_complete_seconds"]=time.perf_counter()-tick;persist()
        model=None;tick=time.perf_counter()
        try:
            model=model_factory(Path(source_root),Path(checkpoint),device=device);sync()
            manifest.update(ma_setup_status="ok",ma_setup=model.setup_report)
        except Exception as error:manifest.update(ma_setup_status="failed",ma_setup_error=f"{type(error).__name__}: {error}")
        manifest["ma_setup_seconds"]=time.perf_counter()-tick
        try:
            for row in rows:
                if row["input_status"]!="ok":continue
                tick=time.perf_counter();cfg=row["original_configuration"];row["ma_table"]=row["name"]+"_ma.json"
                try:
                    if manifest["ma_setup_status"]!="ok":raise RuntimeError(manifest["ma_setup_error"])
                    result=model.extract(Path(cfg["fixed"]),Path(cfg["moving"]),Path(cfg["affine"]),output=output/row["ma_table"])
                    sync()
                    if result.get("status")!="ok":raise ValueError("insufficient fresh reverse MA; no fallback")
                    row["ma_status"]="ok"
                except Exception as error:row.update(ma_status="failed",ma_error=f"{type(error).__name__}: {error}")
                row["ma_complete_seconds"]=time.perf_counter()-tick;persist()
        finally:
            try:
                if model is not None:model.close()
                manifest["ma_close_status"]="ok"
            except Exception as error:
                manifest.update(ma_close_status="failed",ma_close_error=f"{type(error).__name__}: {error}")
                for row in rows:
                    if row["input_status"]=="ok":row.update(ma_status="failed",ma_error=manifest["ma_close_error"])
        manifest["extraction_complete"]=True;persist()
        for row in rows:
            if row["input_status"]!="ok":continue
            for method in ("sg1","fusion"):
                tick=time.perf_counter();record=row["methods"][method]
                try:
                    if row["sg_status"]!="ok":raise ValueError("fresh SG failed")
                    matches=output/row["sg_table"]
                    if method=="fusion":
                        if row["ma_status"]!="ok":raise ValueError("fresh MA failed; no SG fallback")
                        matches=output/(row["name"]+"_fused_matches.json")
                        save(matches,fuse_tables(read(output/row["sg_table"]),read(output/row["ma_table"])))
                    cfg=configuration(row["original_configuration"],matches,output/(row["name"]+"_"+method+".npz"))
                    if method=="fusion":cfg.match_weight=.2
                    record.update(output=cfg.output.name,report=cfg.output.with_suffix(".json").name,
                        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()})
                    record["point_metadata"]=validate_table(cfg)
                    result=optimizer(cfg);sync();ratio=export_validator(cfg,result)
                    record.update(status="ok",actual_minimum_corner_ratio=ratio,
                        **{k:result.get(k) for k in ("gradient_steps","failed_trials","peak_allocated_bytes","optimize_seconds","end_to_end_seconds","saved_binary_certificate")})
                except Exception as error:record.update(status="failed",error=f"{type(error).__name__}: {error}")
                record["complete_call_seconds"]=time.perf_counter()-tick;persist()
        for row in rows:
            if row["input_status"]!="ok":continue
            tick=time.perf_counter();record=row["methods"]["native_shared"]
            try:
                record.update(native_runner(row,output/(row["name"]+"_native"),device=device));sync()
                if record.get("status")!="ok":raise ValueError("native call did not succeed")
            except Exception as error:record.update(status="failed",error=f"{type(error).__name__}: {error}")
            record["complete_call_seconds"]=time.perf_counter()-tick;persist()
    manifest.update(prediction_complete=True,all9_terminal=True,whole_batch_memory=peaks.report(),
        successful_calls=sum(r["methods"][m]["status"]=="ok" for r in rows for m in METHODS),
        failed_calls=sum(r["methods"][m]["status"]!="ok" for r in rows for m in METHODS))
    persist()
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fusion-predictions","source-data","output","source-root","checkpoint"):
        parser.add_argument("--"+name,type=Path,required=True)
    torch.set_num_threads(2)
    result=run(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ("successful_calls","failed_calls","elapsed_seconds")}))
    if result["failed_calls"]:raise SystemExit(1)


if __name__=="__main__":main()
