import importlib.util
import pytest
import torch


def decoder_type():
    name="qcopt.neural_bijection.dense.coordinated_multilevel"
    assert importlib.util.find_spec(name) is not None,"multilevel decoder not implemented"
    from qcopt.neural_bijection.dense.coordinated_multilevel import CoordinatedMultilevelDecoder
    return CoordinatedMultilevelDecoder


def grid(rows,columns):
    y,x=torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64),torch.linspace(0,1,columns,dtype=torch.float64),indexing="ij")
    return torch.stack((x,y),-1)[None]


@pytest.mark.parametrize("backend",["manual","checkpointed_manual"])
def test_full_chain_non_square_values_all_gradients_and_fd(backend):
    cls=decoder_type();generator=torch.Generator().manual_seed(981)
    reference=grid(5,7);initial=reference.expand(2,-1,-1,-1).clone()
    initial[:,1:-1,1:-1]+=torch.randn(2,3,5,2,generator=generator,dtype=initial.dtype)*.008
    initial.requires_grad_()
    latents=[(torch.randn(2,2,l-2,l-2,generator=generator,dtype=initial.dtype)*(.01 if l==3 else .4)).requires_grad_() for l in [3,5]]
    existing=cls(5,7,[3,5],reference=reference,backend="existing",amplitude=.15)
    module=cls(5,7,[3,5],reference=reference,backend=backend,amplitude=.15)
    expected=existing(initial,latents);actual=module(initial,latents)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    weight=torch.randn(actual.shape,generator=generator,dtype=actual.dtype)
    variables=[initial,*latents]
    eg=torch.autograd.grad(expected,variables,weight);ag=torch.autograd.grad(actual,variables,weight)
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,rtol=2e-10,atol=2e-10)
    for index,variable in enumerate(variables):
        tangent=torch.randn(variable.shape,generator=generator,dtype=variable.dtype);h=2e-7
        plus=[v.detach() for v in variables];minus=list(plus)
        plus[index]=variable.detach()+h*tangent;minus[index]=variable.detach()-h*tangent
        fd=((module(plus[0],plus[1:])-module(minus[0],minus[1:]))*weight).sum()/(2*h)
        torch.testing.assert_close(fd,(ag[index]*tangent).sum(),rtol=2e-5,atol=2e-6)
    assert torch.equal(actual[:,0],initial[:,0])
    assert torch.equal(actual[:,:,-1],initial[:,:,-1])
    assert module.stage_count==4 and module.full_control_latent is False


@pytest.mark.parametrize("backend",["existing","manual","checkpointed_manual"])
def test_identity_latent_only_and_copied_reference(backend):
    cls=decoder_type();reference=grid(5,5)
    module=cls(5,5,[3,5],reference=reference,backend=backend,diagonal="bd")
    reference.zero_()
    latents=[torch.zeros(2,2,l-2,l-2,dtype=torch.float64,requires_grad=True) for l in [3,5]]
    actual=module(None,latents)
    torch.testing.assert_close(actual,grid(5,5).expand(2,-1,-1,-1),rtol=0,atol=0)
    gradients=torch.autograd.grad(actual.sum(),latents)
    assert all(torch.isfinite(g).all() and (g!=0).any() for g in gradients)
    assert module.full_control_latent is True and module.diagonal=="bd"


@pytest.mark.parametrize("backend",["manual","checkpointed_manual"])
def test_float32_full_chain_comparison(backend):
    cls=decoder_type();generator=torch.Generator().manual_seed(98)
    reference=grid(5,7).float();initial=reference.clone().requires_grad_()
    latents=[torch.randn(1,2,l-2,l-2,generator=generator)*.2 for l in [3,5]]
    latents=[v.requires_grad_() for v in latents]
    ordinary=cls(5,7,[3,5],reference=reference,backend="existing",amplitude=.15)
    tested=cls(5,7,[3,5],reference=reference,backend=backend,amplitude=.15)
    expected=ordinary(initial,latents);actual=tested(initial,latents)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    upstream=torch.randn(actual.shape,generator=generator)
    eg=torch.autograd.grad(expected,[initial,*latents],upstream)
    ag=torch.autograd.grad(actual,[initial,*latents],upstream)
    for a,b in zip(ag,eg):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize("case",["levels","nesting","backend","reference_grad","diagonal","latent_shape","latent_count","dtype"])
def test_preconditions(case):
    cls=decoder_type();reference=grid(5,7);kwargs={}
    levels=[3,5]
    if case=="levels":levels=[3,9]
    elif case=="nesting":levels=[3,4]
    elif case=="backend":kwargs["backend"]="bad"
    elif case=="reference_grad":reference.requires_grad_()
    elif case=="diagonal":kwargs["diagonal"]="bad"
    latents=[torch.zeros(1,2,l-2,l-2,dtype=torch.float64) for l in [3,5]]
    if case=="latent_shape":latents[1]=latents[1][:,:1]
    elif case=="latent_count":latents=latents[:1]
    elif case=="dtype":latents=[v.float() for v in latents]
    with pytest.raises(ValueError):
        cls(5,7,levels,reference=reference,**kwargs)(grid(5,7),latents)
