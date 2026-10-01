"""Scoring units only; topology is checked by the separate production loader."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import tools.coordinated_score_case as scorer


def test_pixel_units_are_explicit_and_native_error_is_resolution_invariant(tmp_path,monkeypatch):
    source=tmp_path/"source.png"
    Image.new("RGB",(200,160)).save(source)
    affine=tmp_path/"affine.npz"
    np.savez(affine,post_affine_matrix=np.eye(2),post_affine_offset=np.zeros(2))
    mapped=tmp_path/"map.npz"
    np.savez(mapped,interpolation="p1_ac")
    xy=np.stack(np.meshgrid(np.linspace(0,1,3),np.linspace(0,1,3)),axis=-1)
    monkeypatch.setattr(scorer,"load_effective_vertices",lambda path:(xy,dict(composite_representation_valid=True)))
    def labels(path,size):
        assert size==(200,160)
        return {"0":np.array([50.,60.])+(np.array([2.,0.]) if Path(path).name=="moving.csv" else 0)}
    monkeypatch.setattr(scorer,"_landmarks",labels)
    reports=[]
    for side in (512,1024):
        geometry=dict(source=str(source),effective_original_to_canvas_scale_xy=[side/200,side/200],padding_xy=[0,side*.1])
        layout=tmp_path/f"layout{side}.json"
        layout.write_text(json.dumps(dict(side=side,fixed=geometry,moving=geometry)))
        reports.append(scorer.score(layout,Path("fixed.csv"),Path("moving.csv"),
            {"identity":mapped},affine,tmp_path/f"score{side}.json"))
    for report in reports:
        row=report["results"]["identity"]
        assert report["equivalent_pixel_reference_side"]==512
        assert row["mean_native_moving_px"]==pytest.approx(2.)
        assert row["mean_512_equivalent_px"]==pytest.approx(5.12)
        assert row["p90_512_equivalent_px"]==pytest.approx(5.12)
    assert reports[1]["results"]["identity"]["mean_canvas_px"]==pytest.approx(
        2*reports[0]["results"]["identity"]["mean_canvas_px"])


def test_development_score_uses_selected_cases_and_control_filenames(tmp_path,monkeypatch):
    import sys
    import tools.coordinated_score_development as batch
    captured=[]
    def fake_score(layout,fixed,moving,maps,affine,output):
        captured.append((layout,maps,output))
        return dict(landmark_count=69,results={"common_affine":dict(mean_canvas_px=4.,p90_canvas_px=6.,
            mean_512_equivalent_px=2.,p90_512_equivalent_px=3.,mean_native_moving_px=5.)})
    monkeypatch.setattr(batch,"score",fake_score)
    monkeypatch.setattr(sys,"argv",["score","--data-root",str(tmp_path),"--maps-dir",str(tmp_path/"maps"),
        "--canvas-dir",str(tmp_path/"native"),"--cases","rat_kidney","--grid-side","513","--methods","analytic"])
    batch.main()
    assert len(captured)==1
    assert captured[0][0]==tmp_path/"native"/"rat_kidney_layout.json"
    assert captured[0][1]["analytic"].name=="rat_kidney_analytic_mind_edge513.npz"


@pytest.mark.parametrize("side",[512,1024])
def test_saved_dhr_pixel_field_resolution_and_local_signs(tmp_path,monkeypatch,side):
    import SimpleITK as sitk
    source=tmp_path/"source.png";Image.new("RGB",(side,side)).save(source)
    geometry=dict(source=str(source),effective_original_to_canvas_scale_xy=[1.,1.],padding_xy=[0,0])
    layout=tmp_path/"layout.json";layout.write_text(json.dumps(dict(side=side,fixed=geometry,moving=geometry)))
    affine=tmp_path/"affine.npz";np.savez(affine,post_affine_matrix=np.eye(2),post_affine_offset=np.zeros(2))
    mapped=tmp_path/"map.npz";np.savez(mapped,interpolation="p1_ac")
    vertices=np.stack(np.meshgrid(np.linspace(0,1,3),np.linspace(0,1,3)),axis=-1)
    monkeypatch.setattr(scorer,"load_effective_vertices",lambda path:(vertices,dict(composite_representation_valid=True)))
    monkeypatch.setattr(scorer,"_landmarks",lambda path,size:{"0":np.array([side*.3,side*.4])})
    params=tmp_path/"params.json"
    params.write_text(json.dumps(dict(pad_1=[[0,0],[0,0]],pad_2=[[0,0],[0,0]],
        source_resample_ratio=1.,target_resample_ratio=1.,initial_resample_ratio=1.)))
    field=np.zeros((2,side,side),dtype=np.float32);field[0]=side*.01
    field_path=tmp_path/"translation.mha";sitk.WriteImage(sitk.GetImageFromArray(field),str(field_path))
    report=scorer.score(layout,Path("fixed.csv"),Path("moving.csv"),{"identity":mapped},affine,
        tmp_path/"score.json",dhr=[("native",field_path,params)])
    assert report["results"]["native"]["mean_512_equivalent_px"]==pytest.approx(5.12,abs=1e-4)
    assert report["native_dhr_geometry"]["native"]["nonpositive_corner_count"]==0
    field[0]=-2*np.arange(side)[None,:]
    folded=tmp_path/"folded.mha";sitk.WriteImage(sitk.GetImageFromArray(field),str(folded))
    report=scorer.score(layout,Path("fixed.csv"),Path("moving.csv"),{"identity":mapped},affine,
        tmp_path/"fold_score.json",dhr=[("native",folded,params)])
    assert report["native_dhr_geometry"]["native"]["cells_with_nonpositive_corner"]==(side-1)**2
