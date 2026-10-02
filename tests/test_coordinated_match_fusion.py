"""Literal normalization/value/VJP checks plus injected two-arm scheduling."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import coordinated_match_fusion as fusion
from tools.coordinated_real_case import load_image_matches
from tools.coordinated_dhr_existing_inputs import existing_rows
from tools.coordinated_miit_score import PAIRS, load_manifest


def table(count=10,confidence=.8,shift=0.):
    a=np.array([[1.25,.13],[-.07,.92]],dtype=np.float32)
    b=np.array([.11,.09],dtype=np.float32)
    q=np.column_stack((np.linspace(.1,.8,count),np.linspace(.13,.7,count)))
    p=q+shift;p[-2:]=.95
    return dict(fixed="fixed.png",moving="moving.png",image_side=512,
        post_affine_matrix=a.tolist(),post_affine_offset=b.tolist(),
        source_points_unit=q.tolist(),target_points_unit=p.tolist(),confidence=[confidence]*count,
        coordinate_convention=fusion.CONVENTION,targets_manual_landmarks_or_dense_teacher_loaded=False)


def test_preserve_all_rows_and_separate_values_vjps(tmp_path):
    sg,ma=table(),table(15,.2,.015)
    original=copy.deepcopy((sg,ma));combined=fusion.fuse_tables(sg,ma)
    assert (sg,ma)==original
    assert combined["source_points_unit"]==sg["source_points_unit"]+ma["source_points_unit"]
    assert combined["target_points_unit"]==sg["target_points_unit"]+ma["target_points_unit"]
    assert all(r["static_ineligible_rows"]>=2 for r in combined["composition"])
    assert combined["composition"][0]["original_eligible_confidence_mass"]!=combined["composition"][1]["original_eligible_confidence_mass"]
    a=np.asarray(sg["post_affine_matrix"],dtype=np.float32);b=np.asarray(sg["post_affine_offset"],dtype=np.float32)
    points=[]
    for i,record in enumerate((sg,ma,combined)):
        path=tmp_path/f"{i}.json";path.write_text(json.dumps(record),encoding="utf-8")
        obj,meta=load_image_matches(path,a,b,fixed_path=Path("fixed.png"),moving_path=Path("moving.png"),
            image_side=512,device="cpu",dtype=torch.float64,robust_scale=8.)
        points.append(obj)
    assert meta["confidence_denominator"]==pytest.approx(2.,abs=1e-14)
    y,x=torch.meshgrid(torch.linspace(0,1,7,dtype=torch.float64),torch.linspace(0,1,7,dtype=torch.float64),indexing="ij")
    vertices=torch.stack((x+.015*torch.sin(3*x)*torch.sin(3*y),y+.01*x*(1-x)),dim=-1)[None].requires_grad_()
    matrix=torch.tensor(a,dtype=torch.float64)
    separate=.1*points[0](vertices,matrix,"p1_ac")+.1*points[1](vertices,matrix,"p1_ac")
    together=.2*points[2](vertices,matrix,"p1_ac")
    assert float(separate)==pytest.approx(float(together),abs=1e-14)
    separate_vjp=torch.autograd.grad(separate,vertices,retain_graph=True)[0]
    together_vjp=torch.autograd.grad(together,vertices)[0]
    torch.testing.assert_close(separate_vjp,together_vjp,atol=2e-14,rtol=2e-14)


def test_reject_incompatible_tables_without_clipping():
    sg,ma=table(),table(15,.2)
    for key,value in (("fixed","other.png"),("image_side",256),("coordinate_convention","wrong"),
                      ("post_affine_offset",[0.,0.])):
        with pytest.raises(ValueError):fusion.fuse_tables(sg,{**ma,key:value})
    tiny=table(10,.001);tiny["confidence"][-1]=1.
    with pytest.raises(ValueError,match="exceeds1"):fusion.fuse_tables(tiny,ma)
    bad=copy.deepcopy(sg);bad["source_points_unit"][0][0]=float("nan")
    with pytest.raises(ValueError):fusion.fuse_tables(bad,ma)


def original(tmp_path):
    return dict(method="analytic",loss="mind",output_selection="best_full",grid_side=257,image_side=512,
        image_levels=[32,64,128,256,512],levels=[17,33,65,129,257],inner_steps=30,cycles=1,
        strain_weight=3.,strain_model="p1_arap",shape_weight=1e-4,match_weight=.1,oob_weight=1.,
        minimum_jacobian=.001,precision="float64",image_precision="float32",interpolation="p1_ac",
        mind_frame="shared_affine",preprocessing="raw_inverted",fixed="fixed.png",moving="moving.png",
        affine=str(tmp_path/"affine.npz"),matches=str(tmp_path/"sg.json"),output="old.npz",device="cpu")


def test_exact_arm_deltas_and_budget(tmp_path):
    source=original(tmp_path);saved=copy.deepcopy(source)
    for arm,expected in (("fusion",{"matches","match_weight","output"}),("sg2",{"match_weight","output"})):
        cfg=vars(fusion.configuration(source,arm,"new.npz","fusion.json"))
        norm=lambda k,v:str(Path(v)) if k in ("fixed","moving","affine","matches","output") else v
        delta={k for k,v in cfg.items() if norm(k,v)!=norm(k,source[k])}
        assert delta==expected and cfg["match_weight"]==.2
        assert cfg["inner_steps"]*len(cfg["levels"])*2==300
    assert source==saved


def fixtures(tmp_path):
    sg,ma=table(),table(15,.2)
    (tmp_path/"sg.json").write_text(json.dumps(sg),encoding="utf-8")
    np.savez(tmp_path/"affine.npz",post_affine_matrix=np.asarray(sg["post_affine_matrix"],dtype=np.float32),
        post_affine_offset=np.asarray(sg["post_affine_offset"],dtype=np.float32))
    cfg=original(tmp_path)
    miit=[dict(name=f"miit_{m}_to_{f}",moving_section=m,fixed_section=f,map_direction="fixed_canvas_to_moving_canvas",
        fixed="fixed.png",moving="moving.png",affine="affine.npz",layout="layout.json",status="ok",
        methods={k:dict(status="ok",output=k+".npz") for k in ("analytic","f2","dhr")}) for m,f in PAIRS]
    existing=[dict(name=r["name"],status="ok") for r in existing_rows(Path("unused"))]
    cases=[]
    for cohort,rows in (("miit",miit),("existing",existing)):
        path=tmp_path/(cohort+".json")
        path.write_text(json.dumps(dict(rows=rows,prediction_complete=True,annotations_read=False,
            cohort_size=len(rows))),encoding="utf-8")
        cases.extend(dict(name=r["name"],cohort=cohort,source_manifest=str(path),
            source_report="unused.json",original_configuration=copy.deepcopy(cfg)) for r in rows)
    ma_rows=[]
    for i,case in enumerate(cases):
        path=tmp_path/(case["name"]+"_ma.json");path.write_text(json.dumps(ma),encoding="utf-8")
        ma_rows.append(dict(name=case["name"],status="ok",extraction_status="ok",match_table=path.name))
    ma_path=tmp_path/"ma_predictions.json"
    ma_path.write_text(json.dumps(dict(prediction_complete=True,annotations_read=False,rows=ma_rows)),encoding="utf-8")
    return cases,ma_path


def test_all25_composed_then_all50_terminal_before_adapters(tmp_path,monkeypatch):
    cases,ma_path=fixtures(tmp_path)
    # The first fusion composition fails, but the unchanged SG2 arm remains eligible.
    ma=json.loads(ma_path.read_text());ma["rows"][0]["extraction_status"]="failed"
    ma_path.write_text(json.dumps(ma),encoding="utf-8")
    monkeypatch.setattr(fusion,"source_cases",lambda *_:copy.deepcopy(cases))
    output=tmp_path/"run";calls=[]
    def optimize(cfg):
        outer=json.loads((output/"predictions.json").read_text())
        assert outer["composition_complete"] and len(outer["composition_rows"])==25
        for arm in ("fusion","sg2"):
            assert not (output/arm/"miit_predictions.json").exists()
            assert not json.loads((output/arm/"predictions.json").read_text())["prediction_complete"]
        calls.append(cfg)
        assert cfg.match_weight==.2 and cfg.inner_steps*len(cfg.levels)*2==300
        if cfg.output.name.startswith(cases[-1]["name"]):raise ValueError("deliberate final-case optimizer failure")
        return dict(gradient_steps=300,failed_trials=0)
    result=fusion.run("unused","unused",ma_path,output,optimizer=optimize,export_validator=lambda *_:.5)
    assert len(calls)==49 and result["prediction_complete"]
    assert result["arms"]["fusion"]["failed"]==2 and result["arms"]["sg2"]["failed"]==1
    for arm in ("fusion","sg2"):
        manifest=json.loads((output/arm/"predictions.json").read_text())
        assert len(manifest["rows"])==25 and manifest["all50_terminal"]
        scored,_=load_manifest(output/arm/"miit_predictions.json")
        assert len(scored["rows"])==3
    assert json.loads((tmp_path/"sg.json").read_text())==table()
