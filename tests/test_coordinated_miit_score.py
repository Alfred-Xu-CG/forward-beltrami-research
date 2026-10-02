import json
from pathlib import Path

import numpy as np
import pytest

from tools.coordinated_miit_score import (
    PAIRS, load_manifest, metrics, read_points, score, validate_layout,
)
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit,canvas_unit_to_original_pixel


def layout():
    def item(wh,resized,pad):
        return dict(original_wh=wh,resized_wh=resized,padding_xy=pad,
            effective_original_to_canvas_scale_xy=(np.asarray(resized)/wh).tolist())
    return dict(side=512,fixed=item([1000,900],[512,461],[0,25]),
        moving=item([950,1100],[442,512],[35,0]))


def manifest():
    return dict(prediction_complete=True,annotations_read=False,rows=[dict(
        name=f"miit_{m}_to_{f}",moving_section=m,fixed_section=f,
        map_direction="fixed_canvas_to_moving_canvas",status="partial_failure",
        methods={key:dict(status="failed") for key in ("analytic","f2","dhr")})
        for m,f in PAIRS])


def test_half_pixel_two_axis_roundtrip_and_metric():
    value=validate_layout(layout())
    xy=np.asarray([[0.,0.],[700.25,500.75]])
    q=original_pixel_to_canvas_unit(xy,value["moving"],512)
    assert np.allclose(canvas_unit_to_original_pixel(q,value["moving"],512),xy,atol=1e-13)
    displacement=np.asarray([2.,3.])
    predicted=original_pixel_to_canvas_unit(xy+displacement,value["moving"],512)
    result=metrics(predicted,xy,value["moving"],["a","b"])
    assert result["native_moving_pixels"]["mean"]==pytest.approx(np.linalg.norm(displacement))
    assert result["canvas_pixels"]["mean"]==pytest.approx(np.linalg.norm(
        displacement*np.asarray(value["moving"]["effective_original_to_canvas_scale_xy"])))


def test_layout_rejects_bad_scale():
    value=layout();value["fixed"]["effective_original_to_canvas_scale_xy"][0]=0
    with pytest.raises(ValueError,match="scale"):validate_layout(value)


@pytest.mark.parametrize("header",["label,x,y",",label,x,y"])
def test_lowercase_csv_with_optional_dataframe_index(tmp_path,header):
    path=tmp_path/"points.csv"
    rows=["id0,0,0","id1,10.5,20.5"]
    if header.startswith(","):rows=[f"{i},{row}" for i,row in enumerate(rows)]
    path.write_text(header+"\n"+"\n".join(rows)+"\n")
    result=read_points(path,[100,200],expected_count=2)
    assert set(result)=={"id0","id1"}
    assert np.array_equal(result["id1"],[10.5,20.5])


@pytest.mark.parametrize("text,match",[
    ("label,x,y\na,0,0\na,1,1\n","unique"),
    ("label,x,y\na,0,0\n,1,1\n","nonempty"),
    ("label,x,y\na,0,0\n   ,1,1\n","nonempty"),
    ("label,x,y\na,nan,0\nb,1,1\n","finite"),
    ("label,x,y\na,1000,0\nb,1,1\n","native-image"),
    ("label,x,y\na,0,0\n","all 2"),
    ("label,X,Y\na,0,0\nb,1,1\n","columns"),
])
def test_no_bad_or_missing_label_dropping(tmp_path,text,match):
    path=tmp_path/"points.csv";path.write_text(text)
    with pytest.raises(ValueError,match=match):read_points(path,[100,100],expected_count=2)


@pytest.mark.parametrize("change",["pending","annotations","direction","cohort"])
def test_rejects_unfinished_or_reversed_cohort_before_labels(tmp_path,change):
    value=manifest()
    if change=="pending":value["prediction_complete"]=False
    elif change=="annotations":value["annotations_read"]=True
    elif change=="direction":value["rows"][0]["map_direction"]="moving_canvas_to_fixed_canvas"
    else:value["rows"].reverse()
    path=tmp_path/"predictions.json";path.write_text(json.dumps(value))
    with pytest.raises(ValueError):score(path,tmp_path/"NO_COORDINATES")


def test_completed_failures_retained_no_fallback(tmp_path):
    value=manifest()
    for row in value["rows"]:
        row["layout"]="missing_layout.json";row["affine"]="missing_affine.npz"
    path=tmp_path/"predictions.json";path.write_text(json.dumps(value))
    loaded,_=load_manifest(path)
    assert len(loaded["rows"])==3
    result=score(path,tmp_path/"NO_COORDINATES")
    for method in result["aggregate"].values():
        assert method["failed_pairs"]==3 and method["all_three"] is None
    assert all(row["methods"]["common_affine"]["status"]=="failed" for row in result["rows"])


def test_successful_all124_p1_and_native_branches_no_double_affine(tmp_path,monkeypatch):
    from PIL import Image
    import torch
    import tools.coordinated_miit_score as module
    value=manifest()
    one=dict(original_wh=[16,16],resized_wh=[512,512],padding_xy=[0,0],
        effective_original_to_canvas_scale_xy=[32.,32.])
    current=dict(side=512,fixed=one,moving=one)
    a=np.eye(2,dtype=np.float32);b=np.asarray([.01,.02],dtype=np.float32)
    t=np.linspace(0,1,5);xx,yy=np.meshgrid(t,t)
    effective=np.stack([xx,yy],axis=-1)+b
    monkeypatch.setattr(module,"_affine",lambda path:(a,b))
    monkeypatch.setattr(module,"_safe_map",lambda *args:(effective,dict(interpolation="p1_ac")))
    monkeypatch.setattr(module,"_native",lambda *args:("field","params",dict(native=True)))
    calls=[]
    def native(field,params,**kwargs):
        assert field=="field" and params=="params"
        query=kwargs["query"];calls.append(query.clone())
        return query+torch.tensor(b)
    monkeypatch.setattr(module,"dhr_map_at_unit_queries",native)
    for section in {x for pair in PAIRS for x in pair}:
        root=tmp_path/str(section)
        (root/"images").mkdir(parents=True);(root/"landmarks").mkdir()
        Image.new("RGB",(16,16)).save(root/"images"/"image.tif")
        content="label,x,y\n"+"".join(f"id{i},{i%14},{i//14}\n" for i in range(124))
        (root/"landmarks"/f"{section:02d}.csv").write_text(content)
    for row in value["rows"]:
        row.update(status="ok",layout=row["name"]+"_layout.json",affine="affine.npz")
        (tmp_path/row["layout"]).write_text(json.dumps(current))
        row["methods"]={key:dict(status="ok") for key in ("analytic","f2","dhr")}
    path=tmp_path/"predictions.json";path.write_text(json.dumps(value))
    result=score(path,tmp_path)
    assert len(calls)==3
    expected=np.linalg.norm(b.astype(float))*512
    for row in result["rows"]:
        assert row["scored_landmarks"]==124
        for key,record in row["methods"].items():
            assert record["status"]=="ok",record
            assert len(record["metrics"]["canvas_pixels"]["per_label"])==124
            assert record["metrics"]["canvas_pixels"]["mean"]==pytest.approx(expected,rel=2e-6)
    assert all(v["scored_pairs"]==3 for v in result["aggregate"].values())
