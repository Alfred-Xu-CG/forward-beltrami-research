import copy
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.coordinated_ngf import unit_canvas_gradient,FrozenPostwarpNGF,squared_dot_ngf_errors
from tools import coordinated_real_case as app
from tools.coordinated_miit_multiscale_pilot import configuration as pilot_configuration
from test_coordinated_shared_affine_evidence import identity,image,configuration


def matrices(h,w):
    gx=np.zeros((h*w,h*w));gy=gx.copy()
    for y in range(h):
        for x in range(w):
            row=y*w+x
            lo,hi=(0,1) if x==0 else ((w-2,w-1) if x==w-1 else (x-1,x+1))
            gx[row,y*w+lo]=-w/(hi-lo);gx[row,y*w+hi]=w/(hi-lo)
            lo,hi=(0,1) if y==0 else ((h-2,h-1) if y==h-1 else (y-1,y+1))
            gy[row,lo*w+x]=-h/(hi-lo);gy[row,hi*w+x]=h/(hi-lo)
    return torch.tensor(gx),torch.tensor(gy)


@pytest.mark.parametrize("hw",[(2,2),(4,7),(9,5)])
def test_stencil_literal_matrix_transpose_constants_and_ramps(hw):
    h,w=hw;gx,gy=matrices(h,w)
    im=torch.rand(1,1,h,w,generator=torch.Generator().manual_seed(91),dtype=torch.float64,requires_grad=True)
    actual=unit_canvas_gradient(im)
    expected=torch.stack((gx@im.flatten(),gy@im.flatten()),0).reshape(1,2,h,w)
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-14)
    cot=torch.randn(actual.shape,generator=torch.Generator().manual_seed(92),dtype=im.dtype)
    grad,=torch.autograd.grad((actual*cot).sum(),im)
    literal=gx.T@cot[:,0].flatten()+gy.T@cot[:,1].flatten()
    torch.testing.assert_close(grad.flatten(),literal,atol=1e-14,rtol=1e-14)
    assert torch.equal(unit_canvas_gradient(torch.ones_like(im)),torch.zeros_like(actual))
    y,x=torch.meshgrid((torch.arange(h,dtype=im.dtype)+.5)/h,(torch.arange(w,dtype=im.dtype)+.5)/w,indexing="ij")
    ramp=unit_canvas_gradient((2*x-3*y+.4)[None,None])
    torch.testing.assert_close(ramp[:,0],torch.full_like(ramp[:,0],2.),atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(ramp[:,1],torch.full_like(ramp[:,1],-3.),atol=1e-14,rtol=1e-14)


def test_frozen_rule_floor_and_adaptive_branch():
    s=8;x=(torch.arange(s,dtype=torch.float64)+.5)/s
    fixed=(.8*x+.1)[None,None,None].expand(1,1,s,s).clone()
    moving=torch.ones_like(fixed)*.4;mask=torch.ones_like(fixed)
    ngf=FrozenPostwarpNGF(fixed,moving,mask)
    assert float(ngf.epsilon_fixed)==pytest.approx(.08)
    assert float(ngf.epsilon_moving)==pytest.approx(s/255)
    old=(ngf.epsilon_fixed.clone(),ngf.epsilon_moving.clone())
    ngf.errors(fixed.roll(1,-1))
    assert torch.equal(old[0],ngf.epsilon_fixed) and torch.equal(old[1],ngf.epsilon_moving)
    e=ngf.errors(moving.requires_grad_()).mean();g,=torch.autograd.grad(e,moving)
    assert float(e)==1 and torch.equal(g,torch.zeros_like(g))


def test_literal_formula_sign_and_derivative_unit_scaling():
    a=torch.randn(1,2,6,6,generator=torch.Generator().manual_seed(8),dtype=torch.float64)
    b=torch.randn(1,2,6,6,generator=torch.Generator().manual_seed(9),dtype=torch.float64,requires_grad=True)
    ef=torch.tensor(.2,dtype=a.dtype);em=torch.tensor(.3,dtype=a.dtype)
    value=squared_dot_ngf_errors(a,b,ef,em)
    assert torch.equal(value,squared_dot_ngf_errors(a,-b,ef,em))
    torch.testing.assert_close(value,squared_dot_ngf_errors(a/6,b/6,ef/6,em/6),rtol=1e-14,atol=1e-14)
    grad,=torch.autograd.grad(value.sum(),b)
    c=(a*b).sum(1,keepdim=True);aa=a.square().sum(1,keepdim=True)+ef**2;bb=b.square().sum(1,keepdim=True)+em**2
    literal=-2*(c*a/(aa*bb)-c.square()*b/(aa*bb.square()))
    torch.testing.assert_close(grad,literal,rtol=1e-14,atol=1e-14)
    same=squared_dot_ngf_errors(a,a,ef,ef)
    assert bool((same>0).all())  # finite epsilon does NOT imply zero identity loss


def make_evidence(f,m,a,b):
    return app.Evidence(f,m,a,b,"ngf",3.,1.,1e-4,interpolation="p1_ac",strain_model="p1_arap")


def test_full_nonuniform_map_value_and_gradient_literal_afterwarp():
    f=image(torch.float64,height=19);m=f.roll(2,-1)
    a=torch.tensor([[1.02,.037],[-.024,.99]],dtype=f.dtype);offset=torch.tensor([.007,-.011],dtype=f.dtype)
    e=make_evidence(f,m,a,offset)
    v=(identity(7)+torch.randn(1,7,7,2,generator=torch.Generator().manual_seed(352),dtype=f.dtype)*.002).requires_grad_()
    total,parts=e(v)
    from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_pixel_centers
    query=p1_map_at_pixel_centers(v,19,19,diagonal="ac")@a.T+offset
    w=F.grid_sample(m,2*query-1,align_corners=False,padding_mode="zeros")
    gx,gy=matrices(19,19)
    ga=torch.stack((gx@f.flatten(),gy@f.flatten()),0).reshape(1,2,19,19)
    gb=torch.stack((gx@w.flatten(),gy@w.flatten()),0).reshape(1,2,19,19)
    literal=((1-(ga*gb).sum(1,keepdim=True).square()/
        ((ga.square().sum(1,keepdim=True)+e.ngf.epsilon_fixed**2)*
         (gb.square().sum(1,keepdim=True)+e.ngf.epsilon_moving**2)))*e.mask).sum()/e.denominator
    torch.testing.assert_close(parts["image"],literal,rtol=1e-13,atol=1e-13)
    g1,=torch.autograd.grad(parts["image"],v,retain_graph=True);g2,=torch.autograd.grad(literal,v,retain_graph=True)
    torch.testing.assert_close(g1,g2,rtol=1e-12,atol=1e-12)
    g,=torch.autograd.grad(total,v)
    d=torch.randn(v.shape,generator=torch.Generator().manual_seed(353),dtype=v.dtype)*.01
    h=1e-7;fd=(e(v+h*d)[0]-e(v-h*d)[0])/(2*h)
    torch.testing.assert_close((g*d).sum(),fd,rtol=5e-6,atol=2e-8)


def test_known_rotation_queries_original_intensity_once_and_preserves_affine():
    m=image(torch.float64,height=16);f=torch.rot90(m,1,(-2,-1))
    a=torch.tensor([[0.,-1.],[1.,0.]],dtype=m.dtype);b=torch.tensor([1.,0.],dtype=m.dtype)
    e=make_evidence(f,m,a,b);_,parts=e(identity())
    expected=e.ngf.errors(f).mean()
    torch.testing.assert_close(parts["image"],expected,rtol=1e-13,atol=1e-13)
    assert float(parts["oob"])==0 and torch.equal(e.matrix,a)


@pytest.mark.parametrize("method",["analytic","f2"])
def test_tiny_complete_ngf_budget_topology_and_frozen_per_scale_metadata(tmp_path,method):
    cfg=configuration(tmp_path,method);cfg.loss="ngf"
    r=app.optimize(cfg)
    assert r["gradient_steps"]==4 and r["evaluations"]==8 and r["objective_evaluations"]==18
    assert r["failed_trials"]==0 and r["saved_binary_certificate"]["valid"]
    assert set(r["ngf_by_resolution"])=={"8","16"}
    assert r["mind_frame_by_resolution"] is None and r["supplied_tissue_annotation"] is False
    assert all(e["numerator_epsilon_added"] is False for e in r["ngf_by_resolution"].values())


def test_pilot_configuration_changes_only_declared_data_term_fields(tmp_path):
    cfg=configuration(tmp_path,"analytic");before={k:str(v) if hasattr(v,"__fspath__") else v for k,v in vars(cfg).items()}
    report={"configuration":copy.deepcopy(before)}
    actual=pilot_configuration(report,tmp_path/"ngf.npz","continuation",data_term="ngf")
    assert report["configuration"]==before
    assert actual.loss=="ngf" and actual.mind_frame=="original" and actual.mind_order=="transport"
    for key in ("learning_rate","levels","cycles","inner_steps","strain_weight","match_weight","shape_weight"):
        assert getattr(actual,key)==before[key]
    with pytest.raises(ValueError):pilot_configuration(report,tmp_path/"bad.npz","simultaneous_multiscale",data_term="ngf")
