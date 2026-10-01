import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update
from qcopt.neural_bijection.dense.coordinated_stage_cache import FrozenAnchorCoordinatedUpdate


def grid(rows,columns,dtype=torch.float64):
    y,x=torch.meshgrid(torch.linspace(0,1,rows,dtype=dtype),torch.linspace(0,1,columns,dtype=dtype),indexing="ij")
    return torch.stack((x,y),-1)[None]


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_fixed_stage_matches_all_public_outputs_and_proposal_gradients(mode,dtype):
    torch.manual_seed(808)
    anchor=grid(4,7,dtype).expand(2,-1,-1,-1).clone()
    anchor[:,1:-1,1:-1]+=.003*torch.randn_like(anchor[:,1:-1,1:-1])
    reference=grid(4,7)
    reference[:,1:-1,1:-1]+=.002*torch.randn_like(reference[:,1:-1,1:-1])
    raw=(.6*torch.randn(anchor.shape[:-1],dtype=dtype)).requires_grad_()
    kwargs=dict(direction=(.6,.8),mode=mode,minimum_jacobian=.01,theta=.91)
    old=CoordinatedQ1Update(**kwargs)(anchor,raw,reference=reference)
    cached=FrozenAnchorCoordinatedUpdate(anchor,reference=reference,**kwargs)
    new=cached(raw)
    names=("vertices","amplitude","scale","gauge","alpha_max","normalized_margin_min")
    for name in names:torch.testing.assert_close(getattr(old,name),getattr(new,name),rtol=0,atol=0)
    weights={name:torch.randn_like(getattr(old,name)) for name in names}
    loss=lambda result:sum((getattr(result,name)*weights[name]).sum() for name in names)
    a,=torch.autograd.grad(loss(old),raw);b,=torch.autograd.grad(loss(new),raw)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert cached.resident_constant_bytes>anchor.numel()*anchor.element_size()


@pytest.mark.parametrize("mode",["radial","analytic"])
@pytest.mark.parametrize("amplitude",[0.,1e-305])
def test_zero_and_tiny_proposal_candidate_gradient(mode,amplitude):
    anchor=grid(3,3);raw=torch.zeros(1,3,3,dtype=anchor.dtype)
    raw[0,1,1]=amplitude;raw.requires_grad_()
    old=CoordinatedQ1Update(mode=mode)(anchor,raw)
    new=FrozenAnchorCoordinatedUpdate(anchor,mode=mode)(raw)
    torch.testing.assert_close(old.vertices,new.vertices,rtol=0,atol=0)
    a,=torch.autograd.grad(old.vertices.sum(),raw);b,=torch.autograd.grad(new.vertices.sum(),raw)
    assert torch.isfinite(b).all()
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_constants_are_cloned_and_trainable_inputs_rejected():
    anchor=grid(4,7);reference=anchor.clone();before=anchor.clone()
    cached=FrozenAnchorCoordinatedUpdate(anchor,reference=reference)
    anchor.add_(10);reference.mul_(2)
    result=cached(torch.zeros(1,4,7,dtype=before.dtype))
    torch.testing.assert_close(result.vertices,before,rtol=0,atol=0)
    with pytest.raises(ValueError,match="constants"):
        FrozenAnchorCoordinatedUpdate(before.requires_grad_())
    with pytest.raises(ValueError,match="constants"):
        FrozenAnchorCoordinatedUpdate(before.detach(),reference=before)


@pytest.mark.parametrize("mode",["radial","analytic"])
def test_actual_rounded_output_margin_not_prediction(mode):
    anchor=grid(3,3).float();anchor[0,1,1,0]=torch.nextafter(torch.tensor(.25),torch.tensor(1.))
    raw=torch.zeros(1,3,3);raw[0,1,1]=-1.
    cache=FrozenAnchorCoordinatedUpdate(anchor,mode=mode,minimum_jacobian=.5)
    assert cache(raw,validate=False).normalized_margin_min==0
    with pytest.raises(RuntimeError,match="rounded candidate"):
        cache(raw)
