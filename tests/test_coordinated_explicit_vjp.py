import importlib.util
from functools import partial
import pytest
import torch
from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update

BACKWARD_BACKEND="autograd"


@pytest.fixture(params=["autograd","torch_manual"],autouse=True)
def selected_backend(request,monkeypatch):
    monkeypatch.setattr(__import__(__name__,fromlist=["BACKWARD_BACKEND"]),"BACKWARD_BACKEND",request.param)


def function():
    name="qcopt.neural_bijection.dense.coordinated_explicit_vjp"
    assert importlib.util.find_spec(name) is not None,"candidate VJP not implemented"
    from qcopt.neural_bijection.dense.coordinated_explicit_vjp import explicit_coordinated_candidate
    return explicit_coordinated_candidate if BACKWARD_BACKEND=="autograd" else partial(explicit_coordinated_candidate,backward_backend=BACKWARD_BACKEND)


def grid(rows,columns):
    y,x=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64),torch.linspace(0,1,columns,dtype=torch.float64),indexing="ij")
    return torch.stack((x,y),-1)[None]


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("shape",[(4,5),(5,7)])
def test_full_geometry_and_proposal_gradients_unique_branch_fd(mode,shape):
    f=function();generator=torch.Generator().manual_seed(947)
    base=grid(*shape).expand(2,-1,-1,-1).clone()
    base[:,1:-1,1:-1]+=torch.randn(base[:,1:-1,1:-1].shape,generator=generator,dtype=base.dtype)*.008
    reference=grid(*shape)
    reference[:,1:-1,1:-1]+=torch.randn(reference[:,1:-1,1:-1].shape,generator=generator,dtype=base.dtype)*.004
    raw=torch.randn(base.shape[:-1],generator=generator,dtype=base.dtype)*.8
    base.requires_grad_();raw.requires_grad_()
    options=dict(direction=(.6,.8),mode=mode,minimum_jacobian=.01,theta=.91,reference=reference)
    layer=CoordinatedQ1Update(options['direction'],mode=mode,minimum_jacobian=.01,theta=.91)
    ordinary=layer(base,raw,reference=reference)
    slack,delta,_=layer._constraints(base,layer._mask(raw),reference)
    values=((-delta).clamp_min(0)/slack).sort(dim=1,descending=True).values
    assert ((values[:,0]-values[:,1])>1e-6).all() and (ordinary.gauge>1).all()
    manual=f(base,raw,**options)
    torch.testing.assert_close(manual,ordinary.vertices,rtol=0,atol=0)
    weight=torch.randn(base.shape,generator=generator,dtype=base.dtype)
    actual=torch.autograd.grad(manual,(base,raw),weight)
    expected=torch.autograd.grad(ordinary.vertices,(base,raw),weight)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=3e-12,atol=3e-12)
    for index,v in enumerate((base,raw)):
        direction=torch.randn(v.shape,generator=generator,dtype=v.dtype);h=1e-6
        plus=[base.detach(),raw.detach()];minus=list(plus)
        plus[index]=v.detach()+h*direction;minus[index]=v.detach()-h*direction
        fd=((f(*plus,**options)-f(*minus,**options))*weight).sum()/(2*h)
        torch.testing.assert_close(fd,(actual[index]*direction).sum(),rtol=3e-6,atol=3e-8)


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("amplitude",[0.,1.,1e-305])
def test_exact_ties_zero_and_inactive_huge_step_match_autograd(mode,amplitude):
    f=function();base=grid(3,3).requires_grad_();raw=torch.zeros(1,3,3,dtype=base.dtype)
    raw[0,1,1]=amplitude;raw.requires_grad_()
    layer=CoordinatedQ1Update(mode=mode)
    ordinary=layer(base,raw)
    if amplitude==1:
        slack,delta,_=layer._constraints(base,layer._mask(raw),grid(3,3))
        ratio=(-delta).clamp_min(0)/slack
        assert (ratio==ratio.max()).sum()>1
    actual=f(base,raw,mode=mode)
    weight=torch.arange(base.numel(),dtype=base.dtype).reshape_as(base)/base.numel()
    a=torch.autograd.grad(actual,(base,raw),weight)
    b=torch.autograd.grad(ordinary.vertices,(base,raw),weight)
    for first,second in zip(a,b):
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first,second,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_first_order_neural_cascade_and_float32(mode):
    f=function();torch.manual_seed(91)
    base=grid(4,6).float().requires_grad_()
    z1=(torch.randn(base.shape[:-1])*.3).requires_grad_();z2=(torch.randn_like(z1)*.2).requires_grad_()
    layer=CoordinatedQ1Update((.8,.6),mode=mode)
    reference=layer(layer(base,z1).vertices,z2).vertices
    output=f(f(base,z1,direction=(.8,.6),mode=mode),z2,direction=(.8,.6),mode=mode)
    torch.testing.assert_close(output,reference,rtol=0,atol=0)
    weight=torch.randn_like(base)
    a=torch.autograd.grad(output,(base,z1,z2),weight)
    b=torch.autograd.grad(reference,(base,z1,z2),weight)
    for first,second in zip(a,b):torch.testing.assert_close(first,second,rtol=2e-5,atol=2e-6)


def test_rejects_reference_trial_gradients_and_invalid_actual_margin():
    f=function();base=grid(3,4);raw=torch.zeros(base.shape[:-1],dtype=base.dtype)
    for kwargs in (dict(reference=base.clone().requires_grad_()),dict(alpha_trial=torch.tensor(1.,requires_grad=True)),dict(boundary="sliding")):
        with pytest.raises(ValueError):f(base,raw,**kwargs)
    bad=base.clone();bad[0,1,1]=-1
    with pytest.raises(ValueError):f(bad,raw)
    raw.requires_grad_();first=torch.autograd.grad(f(base,raw).sum(),raw,create_graph=True)[0]
    with pytest.raises(RuntimeError):torch.autograd.grad(first.sum(),raw)


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_many_exact_ties_are_processed_in_chunks_not_dropped(mode):
    f=function();base=grid(65,65).requires_grad_()
    yy,xx=torch.meshgrid(torch.arange(65),torch.arange(65),indexing="ij")
    raw=torch.where((xx+yy)%2==0,1.,-1.).double()[None].requires_grad_()
    layer=CoordinatedQ1Update(mode=mode)
    slack,delta,_=layer._constraints(base,layer._mask(raw),grid(65,65))
    ratios=(-delta).clamp_min(0)/slack
    assert (ratios==ratios.max()).sum()>4096
    weight=torch.ones_like(base)
    ordinary=layer(base,raw).vertices;actual=f(base,raw,mode=mode)
    a=torch.autograd.grad(actual,(base,raw),weight)
    b=torch.autograd.grad(ordinary,(base,raw),weight)
    for first,second in zip(a,b):torch.testing.assert_close(first,second,rtol=2e-12,atol=2e-12)


def test_analytic_exact_branch_equality_zero_trial_and_mixed_batch():
    f=function();base=grid(3,3).expand(3,-1,-1,-1).clone().requires_grad_()
    raw=torch.zeros(3,3,3,dtype=base.dtype);raw[:,1,1]=torch.tensor([1.,1.,1e-305],dtype=base.dtype)
    raw.requires_grad_();trial=torch.tensor([.475,0.,1.],dtype=base.dtype)
    layer=CoordinatedQ1Update(mode="analytic",minimum_jacobian=0)
    ordinary=layer(base,raw,alpha_trial=trial)
    assert ordinary.gauge[0]*trial[0]==layer.theta
    actual=f(base,raw,mode="analytic",minimum_jacobian=0,alpha_trial=trial)
    weight=torch.arange(base.numel(),dtype=base.dtype).reshape_as(base)
    a=torch.autograd.grad(actual,(base,raw),weight)
    b=torch.autograd.grad(ordinary.vertices,(base,raw),weight)
    for first,second in zip(a,b):
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first,second,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_rounded_positive_faces_at_extra_margin_floor_are_rejected(mode):
    f=function();base=grid(3,3).float()
    base[0,1,1,0]=torch.nextafter(torch.tensor(.25),torch.tensor(1.))
    raw=torch.zeros(base.shape[:-1]);raw[0,1,1]=-1.
    layer=CoordinatedQ1Update(mode=mode,minimum_jacobian=.5)
    rounded=layer(base,raw,validate=False)
    assert rounded.normalized_margin_min==0 # faces positive, but NOT > eta
    torch.testing.assert_close(f(base,raw,mode=mode,minimum_jacobian=.5,validate=False),rounded.vertices,rtol=0,atol=0)
    with pytest.raises(RuntimeError,match="rounded candidate"):
        f(base,raw,mode=mode,minimum_jacobian=.5)


def test_unique_active_storage_has_no_full_constraint_trajectory():
    f=function();torch.manual_seed(115)
    base=grid(5,7).requires_grad_();raw=torch.randn(1,5,7,dtype=base.dtype,requires_grad=True)
    saved=[]
    def pack(tensor):
        saved.append(tensor)
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        output=f(base,raw,direction=(.6,.8),validate=False)
    assert len(saved)==11
    assert saved[0] is base and saved[1] is raw
    assert all(tensor.numel()<=1 for tensor in saved[2:])
    assert output.requires_grad


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_zero_beta_with_huge_nonzero_proposal_has_no_gauge_nan(mode):
    f=function();base=grid(3,3).requires_grad_()
    raw=torch.zeros(1,3,3,dtype=base.dtype);raw[0,1,1]=1e300;raw.requires_grad_()
    weight=torch.zeros_like(base);weight[...,1]=1 # orthogonal to e => beta=0
    layer=CoordinatedQ1Update(mode=mode)
    # At enormous g, radial 1/(1+g) rounds to the closed eta floor. Both
    # default checks MUST reject it; trusted backward is tested separately.
    validate=mode!="radial"
    if not validate:
        with pytest.raises(RuntimeError):layer(base,raw)
        with pytest.raises(RuntimeError):f(base,raw,mode=mode)
    ordinary=layer(base,raw,validate=validate).vertices
    actual=f(base,raw,mode=mode,validate=validate)
    a=torch.autograd.grad(actual,(base,raw),weight)
    b=torch.autograd.grad(ordinary,(base,raw),weight)
    for first,second in zip(a,b):
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first,second,rtol=0,atol=0)
