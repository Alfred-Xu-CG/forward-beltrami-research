import importlib.util

import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants


def api():
    name="qcopt.neural_bijection.dense.coordinated_joint_stage"
    assert importlib.util.find_spec(name) is not None,"joint stage not implemented"
    from qcopt.neural_bijection.dense.coordinated_joint_stage import FrozenAnchorJointCoordinatedUpdate
    return FrozenAnchorJointCoordinatedUpdate


def grid(rows,columns,dtype=torch.float64):
    yy,xx=torch.meshgrid(torch.linspace(0,1,rows,dtype=dtype),torch.linspace(0,1,columns,dtype=dtype),indexing="ij")
    return torch.stack((xx,yy),-1)[None]


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_unique_active_both_legs_full_latent_gradients_and_fd(mode):
    cls=api();generator=torch.Generator().manual_seed(808)
    anchor=grid(4,7).expand(2,-1,-1,-1).clone()
    anchor[:,1:-1,1:-1]+=.003*torch.randn(2,2,5,2,generator=generator,dtype=anchor.dtype)
    reference=grid(4,7);reference[:,1:-1,1:-1]+=.002*torch.randn(1,2,5,2,generator=generator,dtype=anchor.dtype)
    x=(.6*torch.randn(2,4,7,generator=generator,dtype=anchor.dtype)).requires_grad_()
    y=(.6*torch.randn(2,4,7,generator=generator,dtype=anchor.dtype)).requires_grad_()
    kwargs=dict(reference=reference,mode=mode,minimum_jacobian=.01,theta=.91)
    ordinary=cls(anchor,backend="ordinary",**kwargs)
    cached=cls(anchor,backend="cached_manual",**kwargs)
    expected=ordinary(x,y);actual=cached(x,y)
    for name in ("vertices","scales","gauges","normalized_margin_min","substep_margin_min"):
        torch.testing.assert_close(getattr(actual,name),getattr(expected,name),rtol=0,atol=0)
    assert not actual.scales.requires_grad and not actual.gauges.requires_grad
    assert not actual.normalized_margin_min.requires_grad
    assert not actual.substep_margin_min.requires_grad
    assert (actual.scales<1).all()
    # Independently identify the two selected branches on CURRENT geometry.
    lx=CoordinatedQ1Update((1.,0.),mode=mode,minimum_jacobian=.01,theta=.91)
    ly=CoordinatedQ1Update((0.,1.),mode=mode,minimum_jacobian=.01,theta=.91)
    intermediate=lx(anchor,x,reference=reference).vertices
    for layer,current,proposal in ((lx,anchor,x),(ly,intermediate,y)):
        slack,delta,_=layer._constraints(current,layer._mask(proposal),reference)
        ratios=(-delta).clamp_min(0)/slack
        assert ((ratios==ratios.amax(1)[:,None]).sum(1)==1).all()
    upstream=torch.randn(actual.vertices.shape,generator=generator,dtype=anchor.dtype)
    eg=torch.autograd.grad(expected.vertices,(x,y),upstream)
    ag=torch.autograd.grad(actual.vertices,(x,y),upstream)
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,rtol=2e-10,atol=2e-10)
    for index,variable in enumerate((x,y)):
        tangent=torch.randn(variable.shape,generator=generator,dtype=variable.dtype);h=2e-7
        plus=[x.detach(),y.detach()];minus=list(plus)
        plus[index]=variable.detach()+h*tangent;minus[index]=variable.detach()-h*tangent
        fd=((cached(*plus).vertices-cached(*minus).vertices)*upstream).sum()/(2*h)
        torch.testing.assert_close(fd,(ag[index]*tangent).sum(),rtol=2e-5,atol=2e-7)
    q=q1_corner_determinants(actual.vertices)/q1_corner_determinants(reference)
    torch.testing.assert_close(actual.normalized_margin_min,(q-.01).flatten(1).amin(1),rtol=0,atol=0)
    assert (q>.01).all()
    for edge in (0,-1):
        assert torch.equal(actual.vertices[:,edge],anchor[:,edge])
        assert torch.equal(actual.vertices[:,:,edge],anchor[:,:,edge])


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("backend",["ordinary","cached_manual"])
def test_unchecked_substep_margins_are_actual_rounded_coordinates(mode,backend):
    cls=api();generator=torch.Generator().manual_seed(108)
    anchor=grid(4,7,torch.float32).expand(2,-1,-1,-1).clone()
    reference=anchor.clone()
    anchor[:,1:-1,1:-1]+=.002*torch.randn(2,2,5,2,generator=generator)
    x=(.3*torch.randn(2,4,7,generator=generator)).requires_grad_()
    y=(.3*torch.randn(2,4,7,generator=generator)).requires_grad_()
    options=dict(mode=mode,minimum_jacobian=.01,theta=.91)
    result=cls(anchor,reference=reference,backend=backend,**options)(x,y,validate=False)
    intermediate=CoordinatedQ1Update((1.,0.),**options)(
        anchor,x,reference=reference,validate=False).vertices
    qref=q1_corner_determinants(reference.double())
    expected=torch.stack(tuple((q1_corner_determinants(current.double())/qref-.01)
        .flatten(1).amin(1) for current in (intermediate,result.vertices)),1)
    torch.testing.assert_close(result.substep_margin_min,expected,rtol=0,atol=0)
    torch.testing.assert_close(result.normalized_margin_min,expected[:,1],rtol=0,atol=0)
    assert result.substep_margin_min.shape==(2,2)
    assert not result.substep_margin_min.requires_grad
    assert (result.substep_margin_min>0).all()
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(result.vertices.square().sum(),(x,y)))


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("amplitude",[0.,1e-305,.1])
def test_zero_tiny_and_tied_fields_match_ordinary(mode,amplitude):
    cls=api();anchor=grid(3,3)
    x=torch.zeros(1,3,3,dtype=anchor.dtype);y=x.clone()
    x[0,1,1]=amplitude;y[0,1,1]=.7*amplitude
    x.requires_grad_();y.requires_grad_()
    old=cls(anchor,mode=mode,backend="ordinary")(x,y)
    new=cls(anchor,mode=mode,backend="cached_manual")(x,y)
    torch.testing.assert_close(new.vertices,old.vertices,rtol=0,atol=0)
    a=torch.autograd.grad(old.vertices.square().sum(),(x,y))
    b=torch.autograd.grad(new.vertices.square().sum(),(x,y))
    for one,two in zip(a,b):
        assert torch.isfinite(two).all()
        torch.testing.assert_close(one,two,rtol=2e-12,atol=2e-12)
    if amplitude==0:torch.testing.assert_close(new.vertices,anchor,rtol=0,atol=0)


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_float32_full_two_latent_gradient_and_trial_batch(mode):
    cls=api();anchor=grid(4,7,torch.float32).expand(2,-1,-1,-1).clone()
    generator=torch.Generator().manual_seed(93)
    x=(.2*torch.randn(2,4,7,generator=generator)).requires_grad_()
    y=(.2*torch.randn(2,4,7,generator=generator)).requires_grad_()
    trial=torch.tensor([.8,.9],dtype=torch.float64)
    old=cls(anchor,mode=mode,backend="ordinary")(x,y,alpha_trial=trial)
    new=cls(anchor,mode=mode,backend="cached_manual")(x,y,alpha_trial=trial)
    torch.testing.assert_close(new.vertices,old.vertices,rtol=0,atol=0)
    actual_margin=(q1_corner_determinants(new.vertices.double())/
        q1_corner_determinants(grid(4,7))-.001).flatten(1).amin(1)
    torch.testing.assert_close(new.normalized_margin_min,actual_margin,rtol=0,atol=0)
    upstream=torch.randn(new.vertices.shape,generator=generator)
    a=torch.autograd.grad(old.vertices,(x,y),upstream)
    b=torch.autograd.grad(new.vertices,(x,y),upstream)
    for one,two in zip(a,b):torch.testing.assert_close(one,two,rtol=3e-6,atol=3e-6)


@pytest.mark.parametrize("backend",["ordinary","cached_manual"])
def test_copied_constants_and_default_actual_eta_rounding_rejection(backend):
    cls=api();anchor=grid(4,7);reference=anchor.clone();before=anchor.clone()
    module=cls(anchor,reference=reference,backend=backend)
    anchor.add_(10);reference.mul_(2)
    result=module(torch.zeros(1,4,7,dtype=before.dtype),torch.zeros(1,4,7,dtype=before.dtype))
    torch.testing.assert_close(result.vertices,before,rtol=0,atol=0)
    assert module.extra_no_grad_diagnostic_passes==(backend=="cached_manual")
    tensors=[module._anchor,module._reference]
    if backend=="cached_manual":
        cache=module._x_cache
        assert cache._anchor.untyped_storage().data_ptr()!=module._anchor.untyped_storage().data_ptr()
        assert cache._reference.untyped_storage().data_ptr()!=module._reference.untyped_storage().data_ptr()
        tensors.extend((cache._anchor,cache._reference,cache._layer._cached_qref,
            cache._layer._cached_slack,cache._layer._k1,cache._layer._k2))
    unique={(str(t.device),t.untyped_storage().data_ptr()):t.untyped_storage().nbytes() for t in tensors}
    assert module.resident_constant_bytes==sum(unique.values())>before.numel()*before.element_size()
    anchor=grid(3,3,torch.float32)
    anchor[0,1,1,0]=torch.nextafter(torch.tensor(.25),torch.tensor(1.))
    module=cls(anchor,minimum_jacobian=.5,backend=backend)
    x=torch.zeros(1,3,3);x[0,1,1]=-1.;y=torch.zeros_like(x)
    with pytest.raises(RuntimeError,match="rounded candidate"):
        module(x,y)


@pytest.mark.parametrize("case",["anchor_grad","reference_grad","reference_shape","reference_dtype","reference_device","backend","boundary","proposal_shape","proposal_dtype","trial_grad","trial_shape","trial_negative"])
def test_guards(case):
    cls=api();anchor=grid(4,7);reference=anchor.clone();kwargs={}
    x=torch.zeros(1,4,7,dtype=torch.float64);y=x.clone();call={}
    if case=="anchor_grad":anchor.requires_grad_()
    elif case=="reference_grad":reference.requires_grad_()
    elif case=="reference_shape":reference=grid(3,7)
    elif case=="reference_dtype":reference=reference.float()
    elif case=="reference_device":reference=reference.to("meta")
    elif case=="backend":kwargs["backend"]="wrong"
    elif case=="boundary":kwargs["boundary"]="sliding"
    elif case=="proposal_shape":y=y[:,:,:-1]
    elif case=="proposal_dtype":y=y.float()
    elif case=="trial_grad":call["alpha_trial"]=torch.tensor(1.,requires_grad=True)
    elif case=="trial_shape":call["alpha_trial"]=torch.ones(2)
    elif case=="trial_negative":call["alpha_trial"]=-1.
    with pytest.raises(ValueError):cls(anchor,reference=reference,**kwargs)(x,y,**call)
