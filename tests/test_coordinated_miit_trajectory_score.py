import copy
import json

import torch
import numpy as np
from PIL import Image
import pytest

from tools import coordinated_miit_trajectory_score as module
from tools.coordinated_lung_all20 import make_configuration
from tools.coordinated_miit_score import PAIRS
from test_coordinated_miit_score import manifest
import argparse


def save_json(path,value):path.write_text(json.dumps(value),encoding="utf-8")


@pytest.fixture
def cohort(tmp_path):
    original=tmp_path/"original";replay=tmp_path/"replay";native=tmp_path/"native"
    original.mkdir();replay.mkdir();native.mkdir()
    value=manifest();axis=np.linspace(0.,1.,257);x,y=np.meshgrid(axis,axis)
    reference=np.stack((x,y),-1)[None]
    maps=np.repeat(reference[None],10,axis=0)
    window=np.sin(np.pi*x)**2*np.sin(np.pi*y)**2
    window[[0,-1],:]=0.;window[:,[0,-1]]=0.
    for i in range(10):maps[i,0,...,0]+=.0002*(i+1)*window
    totals=[.9,.8,.8]+[.85]*7;selected=1
    a=np.array([[1.1,.02],[-.03,.9]],dtype=np.float32);b=np.array([.01,.02],dtype=np.float32)
    one=dict(original_wh=[16,16],resized_wh=[512,512],padding_xy=[0,0],effective_original_to_canvas_scale_xy=[32.,32.])
    layout=dict(side=512,fixed=one,moving=one)
    for row,(moving,fixed),available in zip(value["rows"],PAIRS,module.AVAILABLE,strict=True):
        row.update(status="ok",layout=row["name"]+"_layout.json",affine=row["name"]+"_affine.npz")
        save_json(original/row["layout"],layout)
        np.savez(original/row["affine"],post_affine_matrix=a,post_affine_offset=b)
        for section in (moving,fixed):
            root=native/str(section);(root/"images").mkdir(parents=True);(root/"landmarks").mkdir()
            Image.new("RGB",(16,16)).save(root/"images"/"image.tif")
            rows=[f"id{i:03d},{i%14},{i//14}" if section==moving or i<available else f"id{i:03d},inf,inf" for i in range(124)]
            (root/"landmarks"/f"{section:02d}.csv").write_text("label,x,y\n"+"\n".join(rows)+"\n")
        for method in module.METHODS:
            stem=row["name"]+"_"+method
            row["methods"][method]=dict(status="ok",output=stem+".npz",report=stem+".json")
            pair={key:original/row[key] for key in ("fixed","moving","affine") if key in row}
            pair.update(name=row["name"],fixed=original/(row["name"]+"_fixed512.png"),moving=original/(row["name"]+"_moving512.png"))
            config=vars(make_configuration(pair,method,argparse.Namespace(output=original,device="cpu",threads=1)))
            config={k:str(v) if hasattr(v,"parts") else v for k,v in config.items()}
            config.update(f2_floor_safety_fraction=.95 if method=="f2" else 1.,joint_prior_backend="eager",match_p1_sampling="existing")
            stages=[dict(cycle=0,level=level,direction=direction,image_side=side,inner_steps=30,accepted_full_total=total)
                for (level,side,direction),total in zip([(l,s,d) for l,s in zip(config["levels"],config["image_levels"],strict=True)
                for d in ([1.,0.],[0.,1.])],totals,strict=True)] if method=="analytic" else [
                dict(cycle=i//5,level=config["levels"][i%5],direction=None,image_side=config["image_levels"][i%5],inner_steps=30,accepted_full_total=t) for i,t in enumerate(totals)]
            report=dict(configuration=config,stages=stages,initial=dict(total=1.),final=dict(total=.8),selected_stage=selected,
                landmarks_used=False,gradient_steps=300,evaluations=310,objective_evaluations=332,failed_trials=0)
            save_json(original/(stem+".json"),report)
            np.savez(original/(stem+".npz"),vertices=maps[selected],boundary_reference=reference,
                post_affine_matrix=a,post_affine_offset=b,interpolation=np.asarray("p1_ac"))
            config=copy.deepcopy(config);config.update(output=str(replay/(stem+"_replay.npz")),device="cuda",trial_diagnostics="existing")
            report=copy.deepcopy(report);report["configuration"]=config
            save_json(replay/(stem+"_replay.json"),report)
            np.savez(replay/(stem+"_replay.npz"),vertices=maps[selected],boundary_reference=reference,
                post_affine_matrix=a,post_affine_offset=b,interpolation=np.asarray("p1_ac"))
            np.savez(replay/(stem+"_replay_trajectory.npz"),vertices=maps,optimizer_seconds=np.arange(1.,11.))
            save_json(replay/(stem+"_replay_trajectory.json"),dict(configuration_source=str(original/(stem+".json")),
                output_map=str(replay/(stem+"_replay.npz")),manual_labels_loaded_during_record=False,
                gradient_steps=300,stages=stages,initial_objective=dict(total=1.),snapshot_copy_overhead_included_in_optimizer_time=True))
    path=original/"predictions.json";save_json(path,value)
    return path,native,replay


def test_strict_objective_ties_identity_and_no_truth_input():
    assert module.selected_prefix([1.,1.1,.9,.9],1.)==(2,.9)
    assert module.selected_prefix([1.,1.],1.)==(None,1.)
    with pytest.raises(ValueError):module.selected_prefix([float("nan")],1.)


def test_complete_cohort_counts_errors_selection_and_affine_exactly_once(cohort):
    path,native,replay=cohort;result=module.score(path,native,replay)
    for row,count in zip(result["rows"],module.AVAILABLE,strict=True):
        assert len(row["nominal_ids"])==124 and len(row["available_ids"])==count
        for method in module.METHODS:
            entry=row["methods"][method];assert entry["status"]=="ok",entry
            assert entry["endpoint_comparison"]["max_absolute"]==0.
            # All synthetic native pixel-center queries land EXACTLY on grid nodes.
            # Independent literal node lookup then ONE post-affine application.
            indices=np.arange(count);query=np.stack(((indices%14+.5)/16,(indices//14+.5)/16),-1)
            a=np.array([[1.1,.02],[-.03,.9]],dtype=np.float32).astype(float)
            b=np.array([.01,.02],dtype=np.float32).astype(float)
            with np.load(replay/(row["name"]+"_"+method+"_replay_trajectory.npz")) as data:
                v=data["vertices"][0,0]
            raw=v[((indices//14+.5)*16).astype(int),((indices%14+.5)*16).astype(int)]
            literal=np.linalg.norm((raw@a.T+b-query)*512,axis=-1)
            np.testing.assert_allclose(list(entry["trajectory"]["stages"][0]["accepted_errors"]["canvas_pixels"]["per_label"].values()),literal,rtol=0,atol=1e-12)
            for i,stage in enumerate(entry["trajectory"]["stages"]):
                assert stage["selected_prefix_stage"]==(0 if i==0 else 1)
                assert (stage["cumulative_gradients"],stage["cumulative_trials"],stage["cumulative_objective_evaluations"])==(30*(i+1),31*(i+1),1+33*(i+1))
                assert len(stage["accepted_errors"]["canvas_pixels"]["per_label"])==count
                assert stage["accepted"]["mean_canvas_px"]==pytest.approx(stage["accepted_errors"]["canvas_pixels"]["mean"],abs=1e-12)
                assert stage["image_selected_prefix"]["p90_canvas_px"]==pytest.approx(stage["image_selected_prefix_errors"]["canvas_pixels"]["p90"],abs=1e-12)
                assert stage["actual_geometry"]["minimum_normalized_corner"]>.001
    assert all(v["scored_pairs"]==3 and len(v["all_three_prefix_table"])==10 for v in result["aggregate"].values())


@pytest.mark.parametrize("change",["annotations","pending","direction"])
def test_manifest_refusal_precedes_any_labels(cohort,monkeypatch,change):
    path,native,replay=cohort;value=json.loads(path.read_text())
    if change=="annotations":value["annotations_read"]=True
    elif change=="pending":value["prediction_complete"]=False
    else:value["rows"][0]["map_direction"]="moving_to_fixed"
    save_json(path,value)
    monkeypatch.setattr(module,"read_points",lambda *a,**k:pytest.fail("labels read"))
    with pytest.raises(ValueError):module.score(path,native,replay)


@pytest.mark.parametrize("bad",["recipe","counter","direction","selection","boundary","labeloff","missing"])
def test_one_failed_method_retained_no_incomplete_aggregate(cohort,bad):
    path,native,replay=cohort;stem="miit_2_to_3_analytic_replay"
    report_path=replay/(stem+".json");report=json.loads(report_path.read_text())
    if bad=="recipe":report["configuration"]["strain_weight"]=2.
    elif bad=="counter":report["objective_evaluations"]=331
    elif bad=="direction":report["stages"][0]["direction"]=[0.,1.]
    elif bad=="selection":report["selected_stage"]=2
    elif bad=="missing":report_path.unlink()
    elif bad=="labeloff":
        file=replay/(stem+"_trajectory.json");data=json.loads(file.read_text());data["manual_labels_loaded_during_record"]=True;save_json(file,data)
    elif bad=="boundary":
        file=replay/(stem+"_trajectory.npz")
        with np.load(file) as saved:vertices=saved["vertices"].copy();times=saved["optimizer_seconds"].copy()
        vertices[4,0,0,3,0]+=.001
        np.savez(file,vertices=vertices,optimizer_seconds=times)
    if bad in ("recipe","counter","direction","selection"):save_json(report_path,report)
    result=module.score(path,native,replay)
    assert result["rows"][0]["methods"]["analytic"]["status"]=="failed"
    assert result["aggregate"]["analytic"]["pair_denominator"]==3
    assert result["aggregate"]["analytic"]["scored_pairs"]==2
    assert result["aggregate"]["analytic"]["all_three_prefix_table"] is None
    assert result["aggregate"]["f2"]["scored_pairs"]==3


def test_missing_annotations_never_intersect_drop(cohort):
    path,native,replay=cohort;file=native/"3"/"landmarks"/"03.csv"
    text=file.read_text();file.write_text(text.replace("id123,inf,inf\n",""))
    result=module.score(path,native,replay)
    assert all(v["scored_pairs"]==2 and v["all_three_prefix_table"] is None for v in result["aggregate"].values())


def test_repeat_endpoint_difference_reported_without_tolerance_gate(cohort):
    path,native,replay=cohort;original=path.parent/"miit_2_to_3_analytic.npz"
    with np.load(original) as data:arrays={k:data[k].copy() for k in data.files}
    arrays["vertices"][0,128,128,0]+=.000001;np.savez(original,**arrays)
    result=module.score(path,native,replay)
    entry=result["rows"][0]["methods"]["analytic"]
    assert entry["status"]=="ok" and entry["endpoint_comparison"]["max_absolute"]>0.
