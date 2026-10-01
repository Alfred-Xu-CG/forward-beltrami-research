"""Single-forward manual patch diagnostics and first-order full cascade VJP."""
import pytest
import torch
import numpy as np

from qcopt.neural_bijection.dense.coordinated_explicit_vjp import (
    explicit_coordinated_candidate, explicit_coordinated_candidate_with_diagnostics,
)
from qcopt.neural_bijection.dense.coordinated_patch_cascade import CoordinatedPatchCascade
from qcopt.neural_bijection.dense.coordinated_patches import CoordinatedPatchQ1Pass
from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update


def grid(rows,columns=None,dtype=torch.float64):
    columns=rows if columns is None else columns
    yy,xx=torch.meshgrid(torch.arange(rows,dtype=dtype)/(rows-1),
                        torch.arange(columns,dtype=dtype)/(columns-1),indexing="ij")
    return torch.stack((xx,yy),-1)[None]


def generic_corner_margin(vertices,reference,eta=.001):
    def corners(tensor):
        v=tensor.detach().double().numpy()
        a,b,c,d=v[:,:-1,:-1],v[:,:-1,1:],v[:,1:,1:],v[:,1:,:-1]
        return np.stack([np.linalg.det(np.stack((q-p,r-p),-1))
                         for p,q,r in ((a,b,d),(a,b,c),(d,b,c),(a,c,d))],-1)
    return (corners(vertices)/corners(reference)-eta).reshape(vertices.shape[0],-1).min(1)


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("case",["active","zero","ties"])
def test_full_four_pass_b2_rectangular_full_y_proposal_gradients(mode,dtype,case):
    generator=torch.Generator().manual_seed(297)
    reference=grid(13,17,dtype)*torch.tensor((2.,3.),dtype=dtype)
    anchor=reference.repeat(2,1,1,1)
    if case=="active":
        anchor[:,1:-1,1:-1]+=.002*torch.randn(2,11,15,2,generator=generator,dtype=dtype)
        # Separately declared .1 f32 comparison; the original .3 f32 proposal
        # contacts the rounded eta floor and remains a regression below.
        amplitude=.1 if dtype==torch.float32 else .3
        raw=amplitude*torch.randn(anchor.shape[:-1],generator=generator,dtype=dtype)
    elif case=="zero":raw=torch.zeros(anchor.shape[:-1],dtype=dtype)
    else:
        yy,xx=torch.meshgrid(torch.arange(13),torch.arange(17),indexing="ij")
        raw=torch.where((xx+yy)%2==0,1.,-1.).to(dtype)[None].repeat(2,1,1)
    anchor.requires_grad_();raw.requires_grad_()
    options=dict(patch_cells=4,direction=(.6,.8),mode=mode)
    ordinary=CoordinatedPatchCascade(13,17,**options)(anchor,raw,reference=reference)
    manual=CoordinatedPatchCascade(13,17,**options,backward_backend="manual")(anchor,raw,reference=reference)
    for field in ("vertices","patch_scales","patch_gauges","normalized_margin_min","pass_margin_min"):
        torch.testing.assert_close(getattr(manual,field),getattr(ordinary,field),rtol=0,atol=0)
    assert all(not getattr(manual,field).requires_grad for field in
               ("patch_scales","patch_gauges","normalized_margin_min","pass_margin_min"))
    weight=torch.randn(anchor.shape,generator=generator,dtype=dtype)
    mg=torch.autograd.grad(manual.vertices,(anchor,raw),weight)
    og=torch.autograd.grad(ordinary.vertices,(anchor,raw),weight)
    for left,right in zip(mg,og):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left,right,rtol=3e-5 if dtype==torch.float32 else 2e-11,
                                   atol=3e-6 if dtype==torch.float32 else 2e-11)
    np.testing.assert_allclose(manual.normalized_margin_min.numpy(),
                               generic_corner_margin(manual.vertices,reference),rtol=2e-12,atol=2e-13)
    for index in (0,-1):
        assert torch.equal(manual.vertices[:,index],anchor[:,index])
        assert torch.equal(manual.vertices[:,:,index],anchor[:,:,index])


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_unique_active_full_y_and_proposal_directional_fd(mode):
    generator=torch.Generator().manual_seed(297)
    reference=grid(13,17)
    anchor=reference.clone()
    anchor[:,1:-1,1:-1]+=.002*torch.randn(1,11,15,2,generator=generator,dtype=anchor.dtype)
    anchor.requires_grad_()
    raw=(.2*torch.randn(anchor.shape[:-1],generator=generator,dtype=anchor.dtype)).requires_grad_()
    layer=CoordinatedPatchCascade(13,17,4,direction=(.6,.8),mode=mode,backward_backend="manual")
    current=anchor
    for op in layer.passes:
        patch=current.reshape(1,-1,2)[:,op.patch_ids].reshape(-1,5,5,2)
        refpatch=reference.reshape(1,-1,2)[:,op.patch_ids].reshape(-1,5,5,2)
        proposal=op._proposal_patches(raw).reshape(-1,5,5)
        slack,delta,_=op.operator._constraints(patch,proposal,refpatch)
        top=((-delta).clamp_min(0)/slack).sort(1,descending=True).values
        assert (top[:,0]-top[:,1]>1e-6).all() and (top[:,0]>1).all()
        current=op(current,raw,reference=reference).vertices
    weights=torch.randn(anchor.shape,generator=generator,dtype=anchor.dtype)
    result=layer(anchor,raw,reference=reference)
    gradients=torch.autograd.grad(result.vertices,(anchor,raw),weights)
    for index,variable in enumerate((anchor,raw)):
        tangent=torch.randn(variable.shape,generator=generator,dtype=variable.dtype)
        plus,minus=[anchor.detach(),raw.detach()],[anchor.detach(),raw.detach()]
        h=1e-6
        plus[index],minus[index]=variable.detach()+h*tangent,variable.detach()-h*tangent
        fd=((layer(*plus,reference=reference).vertices-layer(*minus,reference=reference).vertices)*weights).sum()/(2*h)
        torch.testing.assert_close((gradients[index]*tangent).sum(),fd,rtol=3e-6,atol=3e-7)


def test_single_forward_diagnostics_actual_global_source_normalization(monkeypatch):
    reference=grid(17,25)*torch.tensor((3.,2.),dtype=torch.float64)
    anchor=reference.clone().requires_grad_()
    raw=torch.full(anchor.shape[:-1],.3,dtype=anchor.dtype,requires_grad=True)
    ordinary=CoordinatedPatchQ1Pass(17,25,8,direction=(.6,.8))(anchor,raw,reference=reference)
    calls=[]
    constraints=CoordinatedQ1Update._constraints
    def counted(self,*args):
        calls.append(1)
        return constraints(self,*args)
    def forbidden(*args,**kwargs):
        raise AssertionError("manual must not call ordinary forward, even for diagnostics")
    monkeypatch.setattr(CoordinatedQ1Update,"_constraints",counted)
    monkeypatch.setattr(CoordinatedQ1Update,"forward",forbidden)
    manual=CoordinatedPatchQ1Pass(17,25,8,direction=(.6,.8),backward_backend="manual")(anchor,raw,reference=reference)
    assert len(calls)==1  # All gathered patches batched into ONE safe forward.
    for field in ("vertices","amplitude","patch_scales","patch_gauges","patch_alpha_max","normalized_margin_min"):
        torch.testing.assert_close(getattr(manual,field),getattr(ordinary,field),rtol=0,atol=0)
        if field!="vertices":assert not getattr(manual,field).requires_grad
    assert manual.normalized_margin_min.item()==pytest.approx(generic_corner_margin(manual.vertices,reference)[0])
    yg,zg=torch.autograd.grad(manual.vertices.sum(),(anchor,raw))
    assert torch.isfinite(yg).all() and torch.isfinite(zg).all()


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("backend",["autograd","torch_manual"])
def test_new_aux_api_and_old_seven_input_tensor_api(mode,backend):
    anchor=grid(5,7).requires_grad_()
    raw=torch.randn(1,5,7,dtype=anchor.dtype,generator=torch.Generator().manual_seed(9)).requires_grad_()
    kwargs=dict(mode=mode,direction=(.6,.8),backward_backend=backend,validate=False)
    old=explicit_coordinated_candidate(anchor,raw,**kwargs)
    candidate,scale,gauge,margin=explicit_coordinated_candidate_with_diagnostics(anchor,raw,**kwargs)
    ordinary=CoordinatedQ1Update((.6,.8),mode=mode)(anchor,raw,validate=False)
    for expected,actual in zip((ordinary.vertices,ordinary.scale,ordinary.gauge,ordinary.normalized_margin_min),
                               (candidate,scale,gauge,margin)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert isinstance(old,torch.Tensor)
    assert all(not t.requires_grad and t.grad_fn is None for t in (scale,gauge,margin))
    with pytest.raises(RuntimeError):torch.autograd.grad(scale.sum(),raw)
    upstream=torch.randn_like(anchor)
    for first,second in zip(torch.autograd.grad(old,(anchor,raw),upstream),
                            torch.autograd.grad(candidate,(anchor,raw),upstream)):
        torch.testing.assert_close(first,second,rtol=0,atol=0)


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_aux_reports_actual_rounded_floor_with_validate_false(mode):
    anchor=grid(3).float()
    anchor[0,1,1,0]=torch.nextafter(torch.tensor(.25),torch.tensor(1.))
    raw=torch.zeros(1,3,3);raw[0,1,1]=-1.
    kwargs=dict(mode=mode,minimum_jacobian=.5,backward_backend="torch_manual")
    _,_,_,margin=explicit_coordinated_candidate_with_diagnostics(anchor,raw,validate=False,**kwargs)
    assert margin.item()==0
    with pytest.raises(RuntimeError,match="rounded"):
        explicit_coordinated_candidate_with_diagnostics(anchor,raw,**kwargs)


@pytest.mark.parametrize("case",["reference","trial","backend","higher_gradient"])
def test_manual_unsupported_features_rejected(case):
    anchor=grid(9).requires_grad_()
    raw=torch.ones(anchor.shape[:-1],dtype=anchor.dtype,requires_grad=True)
    kwargs=dict(backward_backend="manual")
    if case=="reference":kwargs["reference"]=anchor.clone()
    elif case=="trial":kwargs["alpha_trial"]=torch.tensor(1.,requires_grad=True)
    elif case=="backend":kwargs["backward_backend"]="wrong"
    if case=="higher_gradient":
        result=CoordinatedPatchQ1Pass(9,patch_cells=4,**kwargs)(anchor,raw)
        first=torch.autograd.grad(result.vertices.sum(),raw,create_graph=True)[0]
        with pytest.raises(RuntimeError):torch.autograd.grad(first.sum(),raw)
    else:
        with pytest.raises(ValueError):
            reference=kwargs.pop("reference",None);trial=kwargs.pop("alpha_trial",1.)
            CoordinatedPatchQ1Pass(9,patch_cells=4,**kwargs)(anchor,raw,reference=reference,alpha_trial=trial)


def test_original_float32_radial_seed297_amplitude03_floor_rejection_preserved():
    generator=torch.Generator().manual_seed(297)
    reference=grid(13,17,torch.float32)*torch.tensor((2.,3.))
    anchor=reference.repeat(2,1,1,1)
    anchor[:,1:-1,1:-1]+=.002*torch.randn(2,11,15,2,generator=generator)
    raw=.3*torch.randn(anchor.shape[:-1],generator=generator)
    trusted=[]
    for backend in ("ordinary","manual"):
        layer=CoordinatedPatchCascade(13,17,4,direction=(.6,.8),backward_backend=backend)
        with pytest.raises(RuntimeError,match="rounded candidate"):
            layer(anchor,raw,reference=reference)
        result=layer(anchor,raw,reference=reference,validate=False)
        assert (result.pass_margin_min[:,:3]>0).all()
        assert result.pass_margin_min[0,3]<0
        trusted.append(result)
    torch.testing.assert_close(trusted[0].vertices,trusted[1].vertices,rtol=0,atol=0)
    torch.testing.assert_close(trusted[0].pass_margin_min,trusted[1].pass_margin_min,rtol=0,atol=0)
