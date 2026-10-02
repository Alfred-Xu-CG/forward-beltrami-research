import json

import torch
import torch.nn.functional as F
import numpy as np
import pytest

from tools import coordinated_miit_feature_frame_probe as module
from tools.digital_mind_objective_probe import self_similarity,OFFSETS
from test_coordinated_miit_score import manifest
from test_coordinated_miit_trajectory_score import cohort
from PIL import Image


def texture(side=64):
    g=torch.Generator().manual_seed(5301)
    # Directionally anisotropic, deterministic texture; no learned model.
    raw=torch.rand((1,1,side,side),generator=g)
    return F.avg_pool2d(F.pad(raw,(2,2,0,0),mode="replicate"),(1,5),stride=1)


@pytest.mark.parametrize("k",[1,2,3])
def test_exact_quarter_turn_spatial_and_channel_frame(k):
    moving=texture();fixed=torch.rot90(moving,k,(-2,-1))
    matrices=(np.array([[0.,-1.],[1.,0.]]),-np.eye(2),np.array([[0.,1.],[-1.,0.]]))
    offsets=(np.array([1.,0.]),np.ones(2),np.array([0.,1.]))
    a,b=matrices[k-1],offsets[k-1]
    # Independently transform actual offset vectors to identify channels.
    permutation=[OFFSETS.index(tuple((a@np.array(r)).astype(int))) for r in OFFSETS]
    expected=torch.rot90(self_similarity(moving)[0],k,(-2,-1))[:,permutation]
    actual=self_similarity(fixed)[0]
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=7e-7)
    q=np.array([[13.5/64,20.5/64],[30.5/64,42.5/64]])
    costs,flags=module.probe_arrays(fixed,moving,a,b,q,q,q)
    assert costs["raw_truth"].min()>.01
    np.testing.assert_array_equal(costs["affine_truth"],np.zeros(2))
    assert flags["common"].all()


def test_identity_raw_and_affine_features_same_and_all_costs_retained():
    moving=texture();fixed=texture().roll(2,-1)
    q=np.array([[.3,.4],[.6,.55],[.001,.2]])
    truth=q+np.array([.03,-.01]);current=q+np.array([.01,.01])
    costs,flags=module.probe_arrays(fixed,moving,np.eye(2),np.zeros(2),q,truth,current)
    for name in ("truth","current","margin"):
        np.testing.assert_array_equal(costs["raw_"+name],costs["affine_"+name])
    assert len(costs["raw_margin"])==3 and not flags["common"][2]
    assert module.summarize(costs,np.ones(3,dtype=bool))["count"]==3
    assert module.summarize(costs,flags["common"])["count"]==2
    assert module.summarize(costs,np.zeros(3,dtype=bool))==dict(count=0,statistics=None)


def test_exact_existing_prepared_png_pipeline_quarter_turn(tmp_path):
    raw=(texture()[0,0].numpy()*255).round().astype(np.uint8)
    moving_path=tmp_path/"moving.png";fixed_path=tmp_path/"fixed.png"
    Image.fromarray(raw).save(moving_path);Image.fromarray(np.rot90(raw).copy()).save(fixed_path)
    fixed,moving,_,_=module.load_registration_evidence(fixed_path,moving_path,64,
        preprocessing="raw_inverted",dtype=torch.float32,device="cpu")
    assert torch.equal(fixed,torch.rot90(moving,1,(-2,-1)))
    a=np.array([[0.,-1.],[1.,0.]]);b=np.array([1.,0.])
    q=np.array([[20.5/64,30.5/64]])
    costs,flags=module.probe_arrays(fixed,moving,a,b,q,q,q)
    assert costs["raw_truth"][0]>.01 and costs["affine_truth"][0]==0.
    assert flags["common"][0]


def test_support_pixel_centers_subpixels_boundary_and_exact_rotation():
    side=64
    q=np.array([[3.5/side,3.5/side],[60.5/side,60.5/side],[(3.5-1e-5)/side,.5],
        [(60.5+1e-5)/side,.5],[3.7/side,.5],[.001,.001]])
    valid,lo,hi=module.footprint_support(q,side)
    np.testing.assert_array_equal(valid,[True,True,False,False,True,False])
    np.testing.assert_array_equal(lo[0],[0,0]);np.testing.assert_array_equal(hi[1],[63,63])
    a=np.array([[0.,-1.],[1.,0.]]);b=np.array([1.,0.])
    np.testing.assert_array_equal(module.affine_footprint_support(q,a,b,side),valid)
    # Aligned center is inside original moving, but its FULL footprint is not.
    center=np.array([[.8,.5]])
    assert ((center@np.eye(2)*1.5+[-.25,-.25])<1).all()
    assert not module.affine_footprint_support(center,1.5*np.eye(2),[-.25,-.25],side)[0]


def test_affine_extreme_support_agrees_all_footprint_pixels_literal():
    q=np.array([[.2,.3],[.5,.5],[.8,.75]])
    angle=.4;a=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])*1.1
    b=np.array([.17,-.24]);side=64
    valid,lower,upper=module.footprint_support(q,side)
    literal=[]
    for okay,lo,hi in zip(valid,lower.astype(int),upper.astype(int),strict=True):
        pixels=np.array([[x,y] for y in range(lo[1],hi[1]+1) for x in range(lo[0],hi[0]+1)])
        indices=(((pixels+.5)/side)@a.T+b)*side-.5
        literal.append(okay and (np.floor(indices)>=0).all() and (np.ceil(indices)<=side-1).all())
    np.testing.assert_array_equal(module.affine_footprint_support(q,a,b,side),literal)


def test_affine_once_and_fixed_descriptor_sampling_literal():
    moving=texture();fixed=moving.roll(1,-1)
    a=np.array([[1.1,.03],[-.02,.9]]);b=np.array([-.03,.04])
    q=np.array([[.3,.4],[.6,.7]]);target=np.array([[.32,.41],[.66,.65]])
    truth=np.linalg.solve(a,(target-b).T).T
    current=q.copy()
    costs,_=module.probe_arrays(fixed,moving,a,b,q,truth,current)
    def literal(field,points):
        grid=torch.tensor(2*points-1,dtype=torch.float32).reshape(1,1,-1,2)
        return F.grid_sample(field,grid,align_corners=False,mode="bilinear",padding_mode="zeros")[0,:,0].T
    fixed_values=literal(self_similarity(fixed)[0],q)
    raw_values=literal(self_similarity(moving)[0],target)
    np.testing.assert_array_equal(costs["raw_truth"],(fixed_values-raw_values).abs().mean(1).numpy())


@pytest.mark.parametrize("change",["annotations","pending","direction"])
def test_manifest_refused_before_labels(tmp_path,monkeypatch,change):
    value=manifest()
    if change=="annotations":value["annotations_read"]=True
    elif change=="pending":value["prediction_complete"]=False
    else:value["rows"][0]["map_direction"]="moving_to_fixed"
    path=tmp_path/"predictions.json";path.write_text(json.dumps(value))
    monkeypatch.setattr(module,"read_points",lambda *a,**kw:pytest.fail("labels read"))
    with pytest.raises(ValueError):module.probe(path,tmp_path)


def test_complete_available_cohort_preserves328_ids_and_common_support(cohort,monkeypatch):
    path,native,_=cohort;manifest_value=json.loads(path.read_text())
    for row in manifest_value["rows"]:
        row["fixed"]="fixed.png";row["moving"]="moving.png"
        report_path=path.parent/row["methods"]["analytic"]["report"]
        report=json.loads(report_path.read_text());report["configuration"]["image_precision"]="float32"
        report_path.write_text(json.dumps(report))
    path.write_text(json.dumps(manifest_value))
    # Mock ONLY I/O to small prepared intensities; formulas/descriptor/support run actual.
    def load(*args,**kwargs):
        assert kwargs["dtype"]==torch.float32 and kwargs["preprocessing"]=="raw_inverted"
        image=texture();return image.roll(1,-1),image,torch.ones_like(image),dict(name="raw_inverted")
    monkeypatch.setattr(module,"load_registration_evidence",load)
    result=module.probe(path,native)
    assert result["scored_pairs"]==3,result
    assert result["full_cohort"]["count"]==328
    assert result["common_support"]["count"]<328
    for row,count in zip(result["rows"],module.AVAILABLE,strict=True):
        assert len(row["nominal_ids"])==124 and len(row["per_id"])==count
        assert row["full"]["count"]==count
        assert set(next(iter(row["per_id"].values()))["support"])=={"fixed","raw_truth","raw_current","affine_truth","affine_current","common"}


def test_missing_nominal_id_is_failure_not_intersection_drop(cohort,monkeypatch):
    path,native,_=cohort;file=native/"3"/"landmarks"/"03.csv"
    file.write_text(file.read_text().replace("id123,inf,inf\n",""))
    monkeypatch.setattr(module,"load_registration_evidence",lambda *a,**kw:(texture(),texture(),None,{}))
    result=module.probe(path,native)
    assert result["rows"][0]["status"]=="failed"
    assert result["full_cohort"] is None and result["pair_denominator"]==3


@pytest.mark.parametrize("bad",["dtype","shape","matrix","nonfinite"])
def test_array_guards(bad):
    image=texture();a=np.eye(2);b=np.zeros(2);q=np.array([[.5,.5]])
    if bad=="dtype":image=image.double()
    elif bad=="shape":image=image[...,:63]
    elif bad=="matrix":a[0,0]=-1
    else:q[0,0]=float("nan")
    with pytest.raises(ValueError):module.probe_arrays(image,image,a,b,q,q,q)
