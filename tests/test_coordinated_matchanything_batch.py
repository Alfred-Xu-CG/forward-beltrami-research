"""Injected extraction/optimization verifies phase order without model/GPU/labels."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import coordinated_matchanything_batch as batch
from tools.coordinated_dhr_existing_inputs import existing_rows
from tools.coordinated_miit_score import load_manifest, PAIRS


def original_configuration(tmp_path):
    return dict(method="analytic",loss="mind",output_selection="best_full",grid_side=257,image_side=512,
        image_levels=[32,64,128,256,512],levels=[17,33,65,129,257],inner_steps=30,cycles=1,
        strain_weight=3.,strain_model="p1_arap",shape_weight=1e-4,match_weight=.1,oob_weight=1.,
        minimum_jacobian=.001,precision="float64",image_precision="float32",interpolation="p1_ac",
        mind_frame="shared_affine",preprocessing="raw_inverted",fixed=str(tmp_path/"fixed.png"),
        moving=str(tmp_path/"moving.png"),affine=str(tmp_path/"affine.npz"),matches=str(tmp_path/"sg.json"),
        output=str(tmp_path/"old.npz"),learning_rate=.004,match_robust_scale=8.,device="cpu")


def test_exact_matches_output_delta_and_unchanged_source(tmp_path):
    source=original_configuration(tmp_path); original=copy.deepcopy(source)
    cfg=vars(batch.configuration(source,tmp_path/"new.json",tmp_path/"new.npz"))
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in cfg.items()}
    assert {k for k in cfg if cfg[k]!=source[k]}=={"matches","output"}
    assert cfg["preprocessing"]=="raw_inverted" and source==original
    for key,value in (("preprocessing","hematoxylin_proxy"),("match_weight",.2),
                      ("match_robust_scale",4.),("seed_initializer","coupled_mind")):
        with pytest.raises(ValueError):batch.configuration({**source,key:value},"new.json","new.npz")


def fake_cases(tmp_path):
    cfg=original_configuration(tmp_path)
    np.savez(tmp_path/"affine.npz",post_affine_matrix=np.eye(2,dtype=np.float32),
             post_affine_offset=np.zeros(2,dtype=np.float32))
    miit=[dict(name=f"miit_{m}_to_{f}",moving_section=m,fixed_section=f,
        map_direction="fixed_canvas_to_moving_canvas",fixed="fixed.png",moving="moving.png",
        affine="affine.npz",layout="layout.json",status="ok",methods={k:dict(status="ok",output=k+".npz")
        for k in ("analytic","f2","dhr")}) for m,f in PAIRS]
    existing=[dict(name=r["name"],status="ok") for r in existing_rows(Path("unused"))]
    cases=[]
    for cohort,rows in (("miit",miit),("existing",existing)):
        source=tmp_path/(cohort+".json")
        source.write_text(json.dumps(dict(rows=rows,prediction_complete=True,annotations_read=False,
            cohort_size=len(rows))),encoding="utf-8")
        cases.extend(dict(name=r["name"],cohort=cohort,source_manifest=str(source),
            source_report="unused.json",original_configuration=copy.deepcopy(cfg)) for r in rows)
    return cases


@pytest.mark.parametrize("failures,insufficient", [({0,24},{12}), (set(),set())])
def test_all25_extraction_before_optimization_failures_and_costs(tmp_path,monkeypatch,failures,insufficient):
    cases=fake_cases(tmp_path)
    monkeypatch.setattr(batch,"source_cases",lambda *_:copy.deepcopy(cases))
    events=[];calls=[];output=tmp_path/"run"
    class Model:
        setup_report={"load_count":1,"scope":"fake"}
        def __init__(self,*args,**kwargs):events.append("setup")
        def extract(self,fixed,moving,affine,*,output):
            index=len(calls);calls.append(index);events.append(f"extract{index}")
            assert not (output.parent/"miit_predictions.json").exists()
            assert not (output.parent/"existing_predictions.json").exists()
            if index in failures:raise ValueError("deliberate extraction failure")
            count=7 if index in insufficient else 8
            record=dict(status="insufficient_matches" if count<8 else "ok",fixed=str(fixed),moving=str(moving),
                image_side=512,post_affine_matrix=[[1.,0.],[0.,1.]],post_affine_offset=[0.,0.],
                source_points_unit=[[.1+.1*i,.2] for i in range(count)],
                target_points_unit=[[.1+.1*i,.3] for i in range(count)],confidence=[1.]*count,
                targets_manual_landmarks_or_dense_teacher_loaded=False,extraction_seconds=.01)
            output.write_text(json.dumps(record),encoding="utf-8")
            return record
        def close(self):events.append("closed")
    def optimize(cfg):
        assert len(calls)==25 and "closed" in events
        assert json.loads((output/"predictions.json").read_text())["extraction_complete"]
        assert not (output/"miit_predictions.json").exists()
        events.append("optimize")
        return dict(gradient_steps=300,failed_trials=0,serialization_seconds=.01,certification_seconds=.02)
    result=batch.run("unused","unused",output,"source","checkpoint",model_factory=Model,
        optimizer=optimize,export_validator=lambda *_:.5)
    assert result["prediction_complete"] and result["extraction_complete"]
    assert events.count("setup")==events.count("closed")==1
    assert events.index("closed")<events.index("optimize")
    assert len(calls)==25 and events.count("optimize")==25-len(failures|insufficient)
    for i,row in enumerate(result["rows"]):
        assert row["status"]==("failed" if i in failures|insufficient else "ok")
        if i in failures|insufficient:assert row["optimization_status"]=="skipped"
        assert row["configuration"]["preprocessing"]=="raw_inverted"
        assert row["configuration"]["matches"]!=row["original_configuration"]["matches"]
    costs=result["cost_totals"]
    assert costs["model_setup_count"]==1 and costs["extraction_attempts"]==25
    assert costs["extraction_calls"]==25
    assert costs["optimizer_calls"]==events.count("optimize")
    assert costs["extraction_complete_call_seconds"]==sum(r["extraction_complete_call_seconds"] for r in result["rows"])
    miit,_=load_manifest(output/"miit_predictions.json")
    existing=json.loads((output/"existing_predictions.json").read_text())
    assert len(miit["rows"])+len(existing["rows"])==25
    assert miit["rows"][0]["raw_matches"]["matcher"]=="MatchAnything ELoFTR"
    assert existing["rows"][-1]["status"]==("failed" if 24 in failures else "ok")


def test_setup_failure_retains_all25_without_optimizer(tmp_path,monkeypatch):
    monkeypatch.setattr(batch,"source_cases",lambda *_:fake_cases(tmp_path))
    def factory(*args,**kwargs):raise RuntimeError("checkpoint unavailable")
    def forbidden(*args):pytest.fail("optimizer must not run after setup failure")
    result=batch.run("unused","unused",tmp_path/"run","source","checkpoint",
        model_factory=factory,optimizer=forbidden)
    assert result["prediction_complete"] and result["model_setup_status"]=="failed"
    assert len(result["rows"])==25 and all(r["status"]=="failed" for r in result["rows"])
    assert result["cost_totals"]["optimizer_calls"]==0
    assert result["cost_totals"]["extraction_calls"]==0
