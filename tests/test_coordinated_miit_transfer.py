"""Tiny dependency-injected tests; no matcher/native registration or label reads."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import pytest

from tools import coordinated_miit_transfer as runner


def assets(tmp_path):
    source=tmp_path/"source_data"
    for section in (2,3,7,8,10,11):
        directory=source/str(section)/"images";directory.mkdir(parents=True)
        raster=np.random.default_rng(section).integers(30,240,(28+section,32+section,3),dtype=np.uint8)
        Image.fromarray(raster).save(directory/"image.tif")
        poison=source/str(section)/"landmarks";poison.mkdir()
        (poison/f"{section:02d}.csv").write_text("DO NOT OPEN OR PARSE\n")
    return argparse.Namespace(source_data=source,output=tmp_path/"predictions",device="cpu",threads=1)


def affine_extractor(pairs,report_out,*,device_name,match_threshold):
    assert len(pairs)==1 and match_threshold==.3
    name,fixed,moving,output=pairs[0]
    assert fixed.name==name+"_fixed512.png" and moving.name==name+"_moving512.png"
    matrix=np.array([[.98,-.01],[.01,.98]],dtype=np.float32)
    offset=np.array([.02,.01],dtype=np.float32)
    np.savez(output,post_affine_matrix=matrix,post_affine_offset=offset,affine_source="direct_image_superglue")
    result=dict(landmarks_initial_DHR_affine_full_DHR_field_loaded=False,
        rows=[dict(name=name,fixed=str(fixed),moving=str(moving),status="ok",matrix=matrix.tolist(),offset=offset.tolist())])
    report_out.write_text(json.dumps(result));return result


def matcher(config):
    q=np.array([[x,y] for x in (.25,.5,.75) for y in (.25,.5,.75)])
    with np.load(config.affine) as data:matrix=data["post_affine_matrix"];offset=data["post_affine_offset"]
    raw=dict(status="ok",global_geometric_ransac_used=False,
        targets_manual_landmarks_or_dense_teacher_loaded=False,image_side=config.image_side,
        fixed=str(config.fixed),moving=str(config.moving),post_affine_matrix=matrix.tolist(),post_affine_offset=offset.tolist(),
        source_points_unit=q.tolist(),target_points_unit=q.tolist(),confidence=[.8]*len(q),raw_matches=len(q))
    config.output.write_text(json.dumps(raw));return raw


def optimizer(config):
    from tools.digital_q1_dhr_distill import identity_vertices
    with np.load(config.affine) as data:matrix=data["post_affine_matrix"];offset=data["post_affine_offset"]
    vertices=identity_vertices(config.grid_side,device="cpu").double().numpy()
    np.savez(config.output,vertices=vertices,boundary_reference=vertices,
        post_affine_matrix=matrix,post_affine_offset=offset,interpolation="p1_ac")
    gradients=config.cycles*len(config.levels)*config.inner_steps*(2 if config.method=="analytic" else 1)
    result=dict(landmarks_used=False,configuration=runner._plain(vars(config)),gradient_steps=gradients,
        failed_trials=0,evaluations=gradients+4,objective_evaluations=gradients+12,
        saved_binary_certificate={"valid":True},initial={"total":.3},final={"total":.2},optimize_seconds=.001,
        loading_seconds=.002,feature_seconds=.003,serialization_seconds=.004,certification_seconds=.005)
    config.output.with_suffix(".json").write_text(json.dumps(result));return result


def dhr(config):
    config.output.mkdir();final=config.output/"common_affine_dhr/Results_Final";final.mkdir(parents=True)
    (final/"displacement_field.mha").write_text("mock field; no native run")
    (final/"postprocessing_params.json").write_text("{}")
    (config.output/"config.json").write_text("{}")
    with np.load(config.affine) as data:matrix=data["post_affine_matrix"];offset=data["post_affine_offset"]
    (config.output/"runtime.json").write_text(json.dumps(dict(post_affine_matrix=matrix.tolist(),
        post_affine_offset=offset.tolist(),runtime_seconds=.001)))


def invoke(args,**overrides):
    return runner.run(args,production=False,affine_extractor=overrides.get("affine_extractor",affine_extractor),
        match_extractor=overrides.get("match_extractor",matcher),optimizer=overrides.get("optimizer",optimizer),
        dhr_runner=overrides.get("dhr_runner",dhr))


def test_exact_production_recipe_and_metadata_only_direction(tmp_path):
    args=assets(tmp_path);cohort=runner.pairs(args)
    assert [(p["moving_section"],p["fixed_section"]) for p in cohort]==[(2,3),(7,8),(10,11)]
    for pair in cohort:
        assert pair["source_moving"]==args.source_data/str(pair["moving_section"])/"images/image.tif"
        assert pair["source_fixed"]==args.source_data/str(pair["fixed_section"])/"images/image.tif"
        for method in ("analytic","f2"):
            cfg=runner.make_configuration(pair,method,args)
            assert cfg.grid_side==257 and cfg.image_side==512
            assert cfg.levels==[17,33,65,129,257] and cfg.image_levels==[32,64,128,256,512]
            assert cfg.inner_steps*cfg.cycles*len(cfg.levels)*(2 if method=="analytic" else 1)==300
            assert cfg.cycles==(1 if method=="analytic" else 2)
            assert (cfg.strain_model,cfg.strain_weight,cfg.shape_weight)==("p1_arap",3.,1e-4)
            assert (cfg.loss,cfg.mind_order,cfg.image_weight,cfg.oob_weight)==("mind","transport",1.,1.)
            assert cfg.learning_rate==.004 and cfg.lr_calibration=="edge"
            assert (cfg.precision,cfg.image_precision,cfg.interpolation)==("float64","float32","p1_ac")
            assert cfg.joint_prior_backend=="eager" and cfg.match_p1_sampling=="existing" and cfg.p1_sampling=="frozen"
            assert cfg.match_weight==.1 and cfg.match_robust_scale==8. and cfg.capture_prefix=="none"
            assert cfg.f2_accepted_gain==1. and cfg.f2_floor_safety_fraction==(.95 if method=="f2" else 1.)
            assert cfg.output_selection=="best_full" and cfg.control_hierarchy=="fixed"
            assert cfg.geometry_backend==("stage_cache" if method=="analytic" else "existing")
            assert not any("landmark" in key or "annotation" in key for key in vars(cfg))


def test_cohort_before_work_frames_relative_paths_and_unknown_physical_scale(tmp_path,monkeypatch):
    args=assets(tmp_path);events=[]
    original=runner.make_canvas
    def canvas(*a,**kw):
        initial=json.loads((args.output/"predictions.json").read_text())
        assert len(initial["rows"])==3 and initial["prediction_complete"] is False
        events.append("canvas");return original(*a,**kw)
    def affine(*a,**kw):events.append("affine");return affine_extractor(*a,**kw)
    def predict(cfg):events.append(cfg.method);return optimizer(cfg)
    old_open=Path.open
    def guarded_open(path,*a,**kw):
        if path.suffix==".csv":pytest.fail("annotation must not be opened")
        return old_open(path,*a,**kw)
    monkeypatch.setattr(Path,"open",guarded_open)
    report=runner.run(args,production=False,canvas_maker=canvas,affine_extractor=affine,
        match_extractor=matcher,optimizer=predict,dhr_runner=dhr)
    assert events==["canvas","affine","analytic","f2"]*3
    assert report["prediction_complete"] and report["annotations_read"] is False and report["cohort_size"]==3
    assert report["started_utc"].endswith("+00:00")
    assert report["initializer"]=="direct"
    assert report["method_success_counts"]==dict(analytic=3,f2=3,dhr=3)
    for row in report["rows"]:
        assert row["map_direction"]=="fixed_canvas_to_moving_canvas"
        assert not Path(row["layout"]).is_absolute() and not Path(row["affine"]).is_absolute()
        layout=json.loads((args.output/row["layout"]).read_text())
        assert layout["scale_assumption"]==runner.SCALE_ASSUMPTION and layout["landmarks_used"] is False
        for role in ("fixed","moving"):
            assert layout[role]["canvas_mpp"] is None and layout[role]["original_mpp_xy"] is None
            assert layout[role]["source"]==row["source_"+role]
            assert "original_xy+0.5" in layout[role]["original_center_to_canvas_center"]
        assert row["methods"]["analytic"]["budget_complete"] and row["methods"]["f2"]["budget_complete"]
        for method in ("analytic","f2"):
            assert [row["methods"][method][key] for key in ("loading_seconds","feature_seconds",
                "serialization_seconds","certification_seconds")]==[.002,.003,.004,.005]
        assert row["methods"]["dhr"]["wrapper_observed_peak_allocated_bytes"] is None
    assert not any(record["geometry_certified"] for row in report["rows"] for name,record in row["methods"].items() if name=="dhr")


@pytest.mark.parametrize("failure",["source","affine","matches","optimizer","certificate","floor","nan","reserve","native_affine"])
def test_failures_retained_three_direction_denominator_and_later_calls_continue(tmp_path,failure):
    args=assets(tmp_path)
    if failure=="source":(args.source_data/"2/images/image.tif").unlink()
    def affine(*a,**kw):
        if failure=="affine" and a[0][0][0]=="miit_2_to_3":raise RuntimeError("mock affine failure")
        return affine_extractor(*a,**kw)
    def match(cfg):
        if failure=="matches" and cfg.fixed.name.startswith("miit_2_to_3"):raise RuntimeError("mock raw match failure")
        return matcher(cfg)
    def predict(cfg):
        if cfg.fixed.name.startswith("miit_2_to_3") and cfg.method=="analytic":
            if failure=="optimizer":raise RuntimeError("mock optimizer failure")
            result=optimizer(cfg)
            if failure=="certificate":result["saved_binary_certificate"]["valid"]=False
            if failure=="floor":
                with np.load(cfg.output) as data:saved={key:data[key].copy() for key in data.files}
                saved["vertices"][:]=0.;np.savez(cfg.output,**saved)
            if failure=="nan":result["final"]["total"]=float("nan")
            return result
        result=optimizer(cfg)
        if failure=="reserve" and cfg.fixed.name.startswith("miit_2_to_3") and cfg.method=="f2":
            result["configuration"]["f2_floor_safety_fraction"]=1.
        return result
    def native(cfg):
        dhr(cfg)
        if failure=="native_affine" and cfg.fixed.name.startswith("miit_2_to_3"):
            path=cfg.output/"runtime.json";record=json.loads(path.read_text());record["post_affine_offset"]=[0.,0.]
            path.write_text(json.dumps(record))
    report=invoke(args,affine_extractor=affine,match_extractor=match,optimizer=predict,dhr_runner=native)
    assert report["prediction_complete"] and len(report["rows"])==3
    assert report["rows"][0]["status"]=="partial_failure"
    assert all(row["status"]=="ok" for row in report["rows"][1:])
    assert all(method["status"] in ("ok","failed","skipped") for row in report["rows"] for method in row["methods"].values())
    if failure=="matches":assert report["rows"][0]["methods"]["dhr"]["status"]=="ok"
    if failure=="nan":assert report["rows"][0]["methods"]["analytic"]["final"]["total"] is None
    json.loads((args.output/"predictions.json").read_text(),parse_constant=lambda value:pytest.fail("nonstandard JSON"))


def test_incomplete_budget_valid_exports_explicit_not_hidden(tmp_path):
    args=assets(tmp_path)
    def incomplete(cfg):
        result=optimizer(cfg);result["gradient_steps"]-=1;result["failed_trials"]=1;return result
    report=invoke(args,optimizer=incomplete)
    assert report["method_success_counts"]["analytic"]==3
    assert report["safe_method_incomplete_budget_counts"]==dict(analytic=3,f2=3)


def test_existing_output_refused_before_any_call(tmp_path):
    args=assets(tmp_path);args.output.mkdir();(args.output/"keep.txt").write_text("existing material")
    with pytest.raises(FileExistsError):invoke(args)
    assert (args.output/"keep.txt").read_text()=="existing material"


def test_native_cuda_peak_reset_sync_readback_scope_mocked_without_gpu(tmp_path,monkeypatch):
    args=assets(tmp_path);args.device="cuda";events=[]
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:events.append("sync"))
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda:events.append("reset"))
    def peak():events.append("peak");return 123456
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",peak)
    def native(config):events.append("native");dhr(config)
    report=invoke(args,dhr_runner=native)
    assert events==["sync","reset","native","sync","peak"]*3
    assert all(row["methods"]["dhr"]["wrapper_observed_peak_allocated_bytes"]==123456 for row in report["rows"])


def test_quarter_turn_initializer_dispatch_all_three_pairs_same_downstream_recipe(tmp_path,monkeypatch):
    from tools import coordinated_rotation_initializer as rotation
    args=assets(tmp_path);args.initializer="quarter_turns";calls=[]
    def amended(*a,**kw):
        calls.append(a[0][0][0]);result=affine_extractor(*a,**kw)
        result["rows"][0].update(selected_quarter_turns=2,matcher_calls=4)
        return result
    monkeypatch.setattr(rotation,"extract",amended)
    report=runner.run(args,production=False,match_extractor=matcher,optimizer=optimizer,dhr_runner=dhr)
    assert calls==["miit_2_to_3","miit_7_to_8","miit_10_to_11"]
    assert report["initializer"]=="quarter_turns" and report["prediction_complete"]
    for row in report["rows"]:
        assert row["affine_estimation"]["selected_quarter_turns"]==2
        assert row["affine_estimation"]["matcher_calls"]==4
        for method in ("analytic","f2"):
            cfg=row["methods"][method]["configuration"]
            assert cfg["capture_prefix"]=="none" and cfg["joint_prior_backend"]=="eager"
            assert cfg["match_p1_sampling"]=="existing" and cfg["image_weight"]==1.


def test_unknown_initializer_rejected_before_calls(tmp_path):
    args=assets(tmp_path);args.initializer="unapproved"
    with pytest.raises(ValueError,match="initializer"):invoke(args)
    assert not args.output.exists()
