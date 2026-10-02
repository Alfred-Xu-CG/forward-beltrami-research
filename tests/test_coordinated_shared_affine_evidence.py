"""Exploratory frame correction correctness, NOT anatomical efficacy evidence."""
import argparse

import pytest
import torch
import numpy as np

from tools import coordinated_real_case as app
from tools.coordinated_shared_affine_features import prepare_shared_affine_intensity
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from tools.coordinated_lung_all20 import make_configuration
from test_coordinated_lung_all20 import assets
from test_coordinated_descriptor_order import literal_descriptor


def identity(rows=9,columns=None,batch=1):
    columns=rows if columns is None else columns
    y,x=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64),
        torch.linspace(0,1,columns,dtype=torch.float64),indexing="ij")
    return torch.stack((x,y),-1)[None].repeat(batch,1,1,1)


def image(dtype=torch.float32,batch=1,height=16,width=None):
    return torch.rand(batch,1,height,height if width is None else width,
        generator=torch.Generator().manual_seed(751),dtype=dtype)*.8+.1


def evidence(fixed,moving,a,b,**kwargs):
    return app.Evidence(fixed,moving,a,b,"mind",3.,1.,1e-4,
        interpolation="p1_ac",strain_model="p1_arap",**kwargs)


def test_default_and_explicit_original_bitwise_value_all_parts_vertex_gradient():
    f=image();m=f.roll(1,-1);a=torch.tensor([[1.02,.03],[-.02,.97]],dtype=torch.float64)
    b=torch.tensor([.03,-.02],dtype=torch.float64);base=identity()
    base[:,1:-1,1:-1]+=torch.randn(1,7,7,2,generator=torch.Generator().manual_seed(752),dtype=base.dtype)*.002
    results=[]
    for kwargs in ({},{"mind_frame":"original"}):
        e=evidence(f,m,a,b,**kwargs);v=base.clone().requires_grad_();total,parts=e(v)
        results.append((total,{k:x.detach() for k,x in parts.items()},torch.autograd.grad(total,v)[0]))
        assert e.mind_frame_metadata is None
    assert torch.equal(results[0][0],results[1][0]) and torch.equal(results[0][2],results[1][2])
    assert all(torch.equal(results[0][1][k],results[1][1][k]) for k in results[0][1])


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_identity_frame_at_identity_grid_matches_original_exactly(dtype):
    f=image(dtype);m=f.roll(1,-1);a=torch.eye(2,dtype=torch.float64);b=torch.zeros(2,dtype=torch.float64)
    old=evidence(f,m,a,b);new=evidence(f,m,a,b,mind_frame="shared_affine")
    old_total,old_parts=old(identity());new_total,new_parts=new(identity())
    assert torch.equal(old.moving_feature,new.moving_feature)
    assert torch.equal(old_total,new_total)
    assert all(torch.equal(old_parts[k],new_parts[k]) for k in old_parts)
    assert new.mind_frame_metadata["coordinate_dtype"]=="torch.float64"
    assert new.mind_frame_metadata["normalized_sampling_grid_dtype"]==str(dtype)


@pytest.mark.parametrize("k",[1,2,3])
def test_exact_quarter_turn_known_image_corrected_objective(k):
    m=image();f=torch.rot90(m,k,(-2,-1))
    matrices=([[0.,-1.],[1.,0.]],[[-1.,0.],[0.,-1.]],[[0.,1.],[-1.,0.]])
    offsets=([1.,0.],[1.,1.],[0.,1.])
    a=torch.tensor(matrices[k-1],dtype=torch.float64);b=torch.tensor(offsets[k-1],dtype=torch.float64)
    old=evidence(f,m,a,b);new=evidence(f,m,a,b,mind_frame="shared_affine")
    _,parts_old=old(identity());_,parts_new=new(identity())
    assert parts_old["image"].item()>.02
    assert parts_new["image"].item()<1e-7
    assert parts_new["oob"].item()==0.
    assert torch.equal(new.fixed_feature,new.moving_feature)


def test_prewarp_matches_literal_descriptor_and_original_support_not_center_only():
    m=image(torch.float64,height=8);a=torch.tensor([[1.3,.1],[-.08,.9]],dtype=torch.float64)
    b=torch.tensor([-.15,.03],dtype=torch.float64);mask=torch.ones_like(m)
    warped,metadata=prepare_shared_affine_intensity(m,a,b,mask)
    e=evidence(m,m,a,b,mind_frame="shared_affine")
    torch.testing.assert_close(e.moving_feature,literal_descriptor(warped),rtol=1e-12,atol=1e-13)
    assert metadata["shared_affine_descriptor_support"]["full_domain_fraction"]<metadata["fixed_descriptor_support"]["full_domain_fraction"]
    assert metadata["mask_policy"].startswith("ALL")
    assert "ones" in metadata["boundary_difference"] and "NOT" in metadata["support_scope"]


def test_only_image_changes_original_oob_points_priors_mask_and_full_y_fd():
    f=image(torch.float64);m=f.roll(2,-1);a=torch.tensor([[1.12,.04],[-.03,.95]],dtype=torch.float64)
    b=torch.tensor([.025,-.018],dtype=torch.float64)
    p=torch.tensor([[.2,.25],[.6,.65]],dtype=torch.float64)
    points=ImageCorrespondences(p,p+.01,torch.ones(2,dtype=torch.float64))
    mask=torch.ones_like(f);mask[...,:2,:]=.25
    v=identity()+torch.tensor([.0183,.0127],dtype=torch.float64)
    objects=[evidence(f,m,a,b,mind_frame=frame,matches=points,match_weight=.1,fixed_mask=mask)
        for frame in ("original","shared_affine")]
    results=[e(v) for e in objects]
    for key in ("oob","outside_fraction","strain","shape","match"):
        assert torch.equal(results[0][1][key],results[1][1][key]),key
    assert torch.equal(objects[0].mask,objects[1].mask)
    assert torch.equal(objects[0].denominator,objects[1].denominator)
    assert results[1][1]["oob"].item()>0.
    trainable=v.clone().requires_grad_();loss,_=objects[1](trainable)
    gradient,=torch.autograd.grad(loss,trainable)
    direction=torch.randn(v.shape,generator=torch.Generator().manual_seed(753),dtype=v.dtype)*.1
    step=1e-6
    numerical=(objects[1](v+step*direction)[0]-objects[1](v-step*direction)[0])/(2*step)
    torch.testing.assert_close((gradient*direction).sum(),numerical,rtol=2e-7,atol=2e-8)
    assert torch.isfinite(gradient).all()


def test_rectangular_b2_setup_support_weighting_and_gradients():
    m=image(batch=2,height=8,width=12);a=torch.eye(2,dtype=torch.float32);b=torch.zeros(2,dtype=torch.float32)
    mask=torch.zeros_like(m);mask[:,:,3:5,3:9]=1.
    e=evidence(m,m,a,b,mind_frame="shared_affine",fixed_mask=mask)
    assert e.mind_frame_metadata["raster_hw"]==[8,12]
    expected=2*6/(8*12)
    assert e.mind_frame_metadata["shared_affine_descriptor_support"]["full_domain_fraction"]==pytest.approx(expected)
    assert e.mind_frame_metadata["shared_affine_descriptor_support"]["fixed_mask_weighted_fraction"]==1.
    v=identity(5,7,batch=2).requires_grad_()
    assert e.matrix.dtype==e.offset.dtype==torch.float64
    total,_=e(v);grad,=torch.autograd.grad(total,v)
    assert torch.isfinite(grad).all()


def test_copies_frozen_affine_lifetime_and_promotes_values_faithfully():
    m=image();a=torch.tensor([[1.02,.03],[-.02,.97]],dtype=torch.float32)
    b=torch.tensor([.03,-.02],dtype=torch.float32)
    old_a=a.double().clone();old_b=b.double().clone()
    e=evidence(m,m,a,b,mind_frame="shared_affine")
    before=e(identity())[0].clone()
    a.zero_();b.fill_(42.)
    assert torch.equal(e.matrix,old_a) and torch.equal(e.offset,old_b)
    assert torch.equal(before,e(identity())[0])
    assert "STATIC" in e.mind_frame_metadata["support_query_scope"]


@pytest.mark.parametrize("bad",["frame","loss","order","matrix_grad","offset_grad","moving_grad"])
def test_unsupported_evidence_no_silent_trainability_drop(bad):
    m=image();a=torch.eye(2,dtype=torch.float64);b=torch.zeros(2,dtype=torch.float64)
    kwargs=dict(mind_frame="shared_affine",mind_order="transport");loss="mind"
    if bad=="frame":kwargs["mind_frame"]="unknown"
    elif bad=="loss":loss="local_ncc"
    elif bad=="order":kwargs["mind_order"]="after_warp"
    elif bad=="matrix_grad":a.requires_grad_()
    elif bad=="offset_grad":b.requires_grad_()
    else:m.requires_grad_()
    with pytest.raises(ValueError):app.Evidence(m,m,a,b,loss,0.,0.,**kwargs)


def configuration(tmp_path,method):
    source=assets(tmp_path)
    pair=dict(name="he_to_cc10",fixed=source.canvas/"cc10_fixed512.png",moving=source.canvas/"cc10_moving512.png",affine=source.affines_from/"he_to_cc10_affine.npz")
    cfg=make_configuration(pair,method,argparse.Namespace(output=tmp_path,device="cpu",threads=1),production=False)
    cfg.matches=None;cfg.match_weight=0.;cfg.f2_floor_safety_fraction=.95
    return cfg


@pytest.mark.parametrize("method",["analytic","f2"])
def test_tiny_whole_shared_optimizer_budget_topology_and_per_scale_setup(tmp_path,method):
    cfg=configuration(tmp_path,method);cfg.mind_frame="shared_affine"
    report=app.optimize(cfg)
    assert report["gradient_steps"]==4 and report["evaluations"]==8 and report["objective_evaluations"]==18
    assert report["failed_trials"]==0 and report["saved_binary_certificate"]["valid"]
    assert report["mind_frame"]==report["configuration"]["mind_frame"]=="shared_affine"
    assert set(report["mind_frame_by_resolution"])=={"8","16"}
    assert all(item["intensity_prewarp_count"]==item["descriptor_construction_count"]==1 for item in report["mind_frame_by_resolution"].values())
    preprocessing=report["image_preprocessing"]
    assert preprocessing["original_moving_features_no_affine_prewarp"] is False
    assert preprocessing["original_moving_raster_no_affine_prewarp"] is True
    assert preprocessing["moving_descriptor_frame"]=="shared_affine"
    assert "once per raster scale" in preprocessing["moving_descriptor_prewarp_scope"]


def test_tiny_default_explicit_original_whole_path_bitwise(tmp_path):
    cfg=configuration(tmp_path,"analytic");default=app.optimize(cfg)
    with np.load(cfg.output) as data:old=data["vertices"].copy()
    cfg.output=tmp_path/"explicit.npz";cfg.mind_frame="original";explicit=app.optimize(cfg)
    for key in ("final","initial","stages","trace","failures","gradient_steps","objective_evaluations"):
        assert default[key]==explicit[key],key
    with np.load(cfg.output) as data:np.testing.assert_array_equal(old,data["vertices"])
    assert default["mind_frame_by_resolution"] is None
    assert default["image_preprocessing"]==explicit["image_preprocessing"]==dict(
        name="raw_inverted",mask_source="original inverted grayscale >.04",
        original_moving_features_no_affine_prewarp=True)


@pytest.mark.parametrize("change",[
    {"mind_frame":"unknown"},{"method":"radial"},{"control_hierarchy":"nested_p1"},
    {"coordinate_mode":"joint"},{"proposal_filter_steps":1},{"capture_prefix":"mind_discrete"},
    {"fine_patch_cells":2},{"trial_diagnostics":"packed"},{"loss":"local_ncc"},
    {"mind_order":"after_warp"},{"image_weight":0.},{"precision":"float32"},
    {"image_precision":"float64"},{"interpolation":"p1_bd"},
])
def test_unsupported_new_application_mode_before_loading(tmp_path,monkeypatch,change):
    cfg=configuration(tmp_path,"analytic");cfg.mind_frame="shared_affine"
    for key,value in change.items():setattr(cfg,key,value)
    monkeypatch.setattr(app,"load_registration_evidence",lambda *a,**kw:pytest.fail("loaded unsupported config"))
    with pytest.raises(ValueError):app.optimize(cfg)
