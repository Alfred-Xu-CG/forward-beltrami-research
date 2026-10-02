"""Role/inverse/frame and nine-attempt lifecycle checks without real labels/GPU."""
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from tools import coordinated_miit_reverse as reverse
from tools import coordinated_miit_reverse_score as scorer


def test_inverse_only_affine_precision_and_two_composition_orders():
    a=np.array([[1.1,.2],[-.13,.9]],dtype=np.float32);b=np.array([.07,-.05],dtype=np.float32)
    ar,br,report=reverse.inverse_affine(a,b)
    # Independent explicit2x2 inverse, not the implementation's linalg routine.
    x,y,z,w=map(float,a.flat);det=x*w-y*z
    ai=np.array([[w,-y],[-z,x]])/det;bi=-ai@b.astype(float)
    np.testing.assert_array_equal(ar,ai.astype(np.float32));np.testing.assert_array_equal(br,bi.astype(np.float32))
    assert report["stored_determinant"]>0 and report["optimized_map_inverted"] is False
    assert report["reverse_after_forward_max_corner_canvas_pixels"]<1e-3
    assert report["forward_after_reverse_max_corner_canvas_pixels"]<1e-3
    with pytest.raises(ValueError):reverse.inverse_affine(np.diag([-1.,1.]),b)


def fixtures(tmp_path):
    native=tmp_path/"native";canvases=tmp_path/"canvases";canvases.mkdir()
    cases=[]
    for moving,fixed in reverse.FORWARD:
        name=f"miit_{moving}_to_{fixed}"
        paths={role:canvases/(name+"_"+role+"512.png") for role in ("fixed","moving")}
        roles={}
        for role,section in (("moving",moving),("fixed",fixed)):
            image_path=native/str(section)/"images/image.tif";image_path.parent.mkdir(parents=True)
            Image.new("RGB",(512,512),(section,section+1,section+2)).save(image_path)
            Image.new("RGB",(512,512),(section,section+1,section+2)).save(paths[role])
            roles[role]=dict(original_wh=[512,512],resized_wh=[512,512],padding_xy=[0,0],
                             effective_original_to_canvas_scale_xy=[1.,1.],source=str(image_path))
        reverse.save(canvases/(name+"_layout.json"),dict(side=512,**roles))
        affine=canvases/(name+"_affine.npz")
        np.savez(affine,post_affine_matrix=np.array([[.99,.03],[-.03,.99]],dtype=np.float32),
                 post_affine_offset=np.array([.01,.02],dtype=np.float32))
        cfg=dict(method="analytic",loss="mind",output_selection="best_full",grid_side=257,image_side=512,
            image_levels=[32,64,128,256,512],levels=[17,33,65,129,257],inner_steps=30,cycles=1,
            strain_weight=3.,strain_model="p1_arap",shape_weight=1e-4,match_weight=.2,oob_weight=1.,
            minimum_jacobian=.001,precision="float64",image_precision="float32",interpolation="p1_ac",
            mind_frame="shared_affine",fixed=str(paths["fixed"]),moving=str(paths["moving"]),
            affine=str(affine),matches="forward_fused_table_MUST_NOT_READ.json",output="forward_map_MUST_NOT_READ.npz",device="cpu")
        cases.append(dict(name=name,cohort="miit",original_configuration=cfg))
    return cases,native


def test_prepare_exact_role_swap_and_only_source_affine(tmp_path):
    cases,native=fixtures(tmp_path);out=tmp_path/"new";out.mkdir()
    row=reverse.prepare(cases[0],native,out)
    assert (row["name"],row["moving_section"],row["fixed_section"])==("miit_3_to_2",3,2)
    assert (out/row["fixed"]).read_bytes()==Path(cases[0]["original_configuration"]["moving"]).read_bytes()
    assert (out/row["moving"]).read_bytes()==Path(cases[0]["original_configuration"]["fixed"]).read_bytes()
    assert Path(row["source_fixed"]).parts[-3]=="2" and Path(row["source_moving"]).parts[-3]=="3"
    layout=reverse.read(out/row["layout"])
    assert layout["fixed"]["source"]==row["source_fixed"] and layout["moving"]["source"]==row["source_moving"]
    assert row["original_configuration"]["match_weight"]==.1


def table(args):
    with np.load(args.affine) as a:
        matrix=a["post_affine_matrix"].tolist();offset=a["post_affine_offset"].tolist()
    q=np.column_stack((np.linspace(.2,.6,10),np.linspace(.3,.7,10))).tolist()
    return dict(status="ok",fixed=str(args.fixed),moving=str(args.moving),image_side=512,
        post_affine_matrix=matrix,post_affine_offset=offset,source_points_unit=q,target_points_unit=q,
        confidence=[.8]*10,targets_manual_landmarks_or_dense_teacher_loaded=False,
        coordinate_convention="fixed unit q -> affine-aligned moving unit p; full moving point is A p+b")


@pytest.mark.parametrize("fail",[False,True])
def test_all9_predeclared_fresh_tables_shared_affine_and_failure_retention(tmp_path,fail):
    cases,native=fixtures(tmp_path);out=tmp_path/"run";events=[]
    def sg(args):
        value=reverse.read(out/"predictions.json")
        assert value["attempt_denominator"]==9 and not value["prediction_complete"]
        assert all(r["methods"][m]["status"]=="pending" for r in value["rows"] for m in reverse.METHODS)
        events.append("sg")
        if fail and len(events)==1:raise ValueError("first SG failure")
        value=table(args);reverse.save(args.output,value);return value
    class MA:
        def __init__(self,*args,**kwargs):
            assert events==["sg"]*3
            self.setup_report={};events.append("setup")
        def extract(self,fixed,moving,affine,*,output):
            value=table(type("Args",(),dict(fixed=fixed,moving=moving,affine=affine)))
            reverse.save(output,value);events.append("ma");return value
        def close(self):events.append("close")
    def optimize(cfg):
        assert "close" in events
        # Actual optimizer rejects if either map OR its report already exists.
        # A table named *_fusion.json would collide with *_fusion.npz's report.
        assert not cfg.output.exists() and not cfg.output.with_suffix(".json").exists()
        assert cfg.matches != cfg.output.with_suffix(".json")
        assert cfg.match_weight in (.1,.2) and cfg.inner_steps*len(cfg.levels)*2==300
        assert "forward_" not in str(cfg.matches) and cfg.fixed.name.endswith("_fixed512.png")
        with np.load(cfg.affine) as a:
            value=reverse.read(cfg.matches)
            np.testing.assert_array_equal(a["post_affine_matrix"],value["post_affine_matrix"])
        events.append("opt")
        return dict(gradient_steps=300,failed_trials=0)
    def native_run(row,folder,*,device):
        value=reverse.read(out/"predictions.json")
        assert all(r["methods"][m]["status"] in ("ok","failed") for r in value["rows"] for m in ("sg1","fusion"))
        with np.load(out/row["affine"]) as a:
            np.testing.assert_array_equal(a["post_affine_matrix"],row["shared_affine_matrix"])
            np.testing.assert_array_equal(a["post_affine_offset"],row["shared_affine_offset"])
        events.append("native")
        if fail and row["name"]=="miit_11_to_10":raise ValueError("native failure")
        return dict(status="ok")
    result=reverse.run("unused",native,out,"source","checkpoint",cases=cases,
        sg_extractor=sg,model_factory=MA,optimizer=optimize,native_runner=native_run,export_validator=lambda *_:.5)
    assert result["all9_terminal"] and result["successful_calls"]==(6 if fail else 9)
    assert result["failed_calls"]==(3 if fail else 0)
    assert events.count("native")==3 and events.count("ma")==3
    assert result["annotations_read"] is False
    assert scorer.load_completed(out)[0]["all9_terminal"]


def test_scorer_refuses_any_pending_before_coordinate_access(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("CSV accessed before completion")
    monkeypatch.setattr(scorer,"read_points",forbidden)
    path=tmp_path/"predictions.json"
    reverse.save(path,dict(prediction_complete=False,annotations_read=False))
    with pytest.raises(ValueError,match="BEFORE labels"):scorer.score(path,tmp_path)
    value=dict(prediction_complete=True,all9_terminal=True,annotations_read=False,attempt_denominator=9,cohort_size=3,
        rows=[dict(name=f"miit_{m}_to_{f}",moving_section=m,fixed_section=f,input_status="ok",
                   methods={k:dict(status="ok") for k in reverse.METHODS}) for m,f in reverse.REVERSE])
    value["rows"][2]["methods"]["native_shared"]["status"]="pending";reverse.save(path,value)
    with pytest.raises(ValueError,match="BEFORE labels"):scorer.score(path,tmp_path)
