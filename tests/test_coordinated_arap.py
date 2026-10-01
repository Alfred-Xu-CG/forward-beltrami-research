import importlib.util

import pytest
import torch
import numpy as np


def api():
    name="qcopt.neural_bijection.dense.coordinated_arap"
    assert importlib.util.find_spec(name) is not None,"P1 ARAP not implemented"
    from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
    return p1_arap_energy


def grid(rows,columns,dtype=torch.float64):
    yy,xx=torch.meshgrid(torch.arange(rows,dtype=dtype)/(rows-1),
        torch.arange(columns,dtype=dtype)/(columns-1),indexing="ij")
    return torch.stack((xx,yy),-1)[None]


def independent_jacobians(vertices,diagonal):
    """Generic triangle edge matrices and inverse source edge matrices."""
    table=vertices.detach().double().numpy()
    batch,rows,columns,_=table.shape
    source=grid(rows,columns).numpy()[0]
    result=[]
    for b in range(batch):
        for r in range(rows-1):
            for c in range(columns-1):
                ids=((r,c),(r,c+1),(r+1,c+1),(r+1,c))
                triangles=((0,1,2),(0,2,3)) if diagonal=="ac" else ((0,1,3),(1,2,3))
                for triangle in triangles:
                    points=[ids[i] for i in triangle]
                    target_edges=np.stack([table[b,*points[i]]-table[b,*points[0]] for i in (1,2)],axis=1)
                    source_edges=np.stack([source[points[i]]-source[points[0]] for i in (1,2)],axis=1)
                    result.append(target_edges@np.linalg.inv(source_edges))
    return np.asarray(result)


def independent_energy(vertices,diagonal):
    jacobians=independent_jacobians(vertices,diagonal)
    u,_,vt=np.linalg.svd(jacobians)
    rotations=u@vt
    assert (np.linalg.det(jacobians)>0).all() and (np.linalg.det(rotations)>0).all()
    return .5*np.square(jacobians-rotations).sum(axis=(-1,-2)).mean()


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_nonsquare_batch_independent_triangle_svd_and_fd(diagonal,dtype):
    generator=torch.Generator().manual_seed(772)
    vertices=grid(4,7,dtype).expand(2,-1,-1,-1).clone()
    matrix=torch.tensor([[1.2,.17],[-.09,.83]],dtype=dtype)
    vertices=vertices@matrix.T+.01*torch.randn(vertices.shape,generator=generator,dtype=dtype)
    vertices.requires_grad_()
    result=api()(vertices,diagonal=diagonal)
    reference=torch.tensor(independent_energy(vertices,diagonal),dtype=dtype)
    torch.testing.assert_close(result,reference,rtol=2e-6 if dtype==torch.float32 else 2e-13,
        atol=2e-7 if dtype==torch.float32 else 2e-14)
    gradient,=torch.autograd.grad(result,vertices)
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)>0
    tangent=torch.randn(vertices.shape,generator=generator,dtype=dtype)
    h=1e-4 if dtype==torch.float32 else 2e-7
    fd=(independent_energy(vertices.detach()+h*tangent,diagonal)-
        independent_energy(vertices.detach()-h*tangent,diagonal))/(2*h)
    torch.testing.assert_close((gradient*tangent).sum(),torch.tensor(fd,dtype=dtype),
        rtol=2e-3 if dtype==torch.float32 else 2e-7,atol=2e-3 if dtype==torch.float32 else 2e-8)
    assert result.ndim==0 and result.dtype==dtype and result.device==vertices.device


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("matrix",[[[1.,0.],[0.,1.]],[[1.3,0.],[0.,1.3]],[[1.4,0.],[0.,.7]],[[1.,.4],[0.,1.]]])
def test_affine_source_area_normalization_and_full_gradient(diagonal,matrix):
    matrix=torch.tensor(matrix,dtype=torch.float64)
    vertices=(grid(3,6)@matrix.T).expand(2,-1,-1,-1).clone().requires_grad_()
    jacobians=independent_jacobians(vertices,diagonal)
    u,_,vt=np.linalg.svd(matrix.numpy());rotation=u@vt
    # All uniform source triangles have area .5/(2*5); total per batch is1.
    weighted_integral=.5*np.square(jacobians-rotation).sum(axis=(-1,-2)).sum()*(.5/(2*5))/2
    actual=api()(vertices,diagonal)
    torch.testing.assert_close(actual,torch.tensor(weighted_integral),rtol=2e-13,atol=2e-14)
    assert torch.autograd.gradcheck(lambda value:api()(value,diagonal),vertices,eps=1e-6,atol=2e-6,rtol=2e-5)
    if torch.equal(matrix,torch.eye(2,dtype=torch.float64)):
        gradient,=torch.autograd.grad(actual,vertices)
        assert gradient.abs().max()<2e-15


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_left_rotation_translation_invariance_and_covariant_gradient(diagonal):
    generator=torch.Generator().manual_seed(51)
    vertices=(grid(4,6)+.004*torch.randn(1,4,6,2,generator=generator,dtype=torch.float64)).requires_grad_()
    angle=torch.tensor(.71,dtype=torch.float64)
    rotation=torch.stack((torch.stack((angle.cos(),-angle.sin())),torch.stack((angle.sin(),angle.cos()))))
    transformed=(vertices.detach()@rotation.T+torch.tensor([1.7,-2.3])).requires_grad_()
    first=api()(vertices,diagonal);second=api()(transformed,diagonal)
    torch.testing.assert_close(first,second,rtol=2e-12,atol=2e-14)
    gradient,=torch.autograd.grad(first,vertices)
    changed,=torch.autograd.grad(second,transformed)
    torch.testing.assert_close(changed,gradient@rotation.T,rtol=2e-11,atol=2e-13)
    assert gradient.sum((0,1,2)).abs().max()<2e-14


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_connected_rotation_second_derivative_at_repeated_singular_values(diagonal):
    vertices=(1.2*grid(2,2)).requires_grad_()
    assert torch.autograd.gradgradcheck(lambda value:api()(value,diagonal),vertices,eps=1e-6,atol=2e-6,rtol=2e-5)


def test_validates_only_declared_actual_faces_and_unchecked_bad_map():
    ac=grid(2,2);ac[0,1,1]=torch.tensor([.2,.2])
    assert torch.isfinite(api()(ac,"ac"))
    with pytest.raises(ValueError,match="positive"):
        api()(ac,"bd")
    bd=grid(2,2);bd[0,0,1]=torch.tensor([.2,.4])
    assert torch.isfinite(api()(bd,"bd"))
    with pytest.raises(ValueError,match="positive"):
        api()(bd,"ac")
    bad=torch.zeros(1,2,2,2,dtype=torch.float64)
    with pytest.raises(ValueError,match="positive"):
        api()(bad)
    assert not torch.isfinite(api()(bad,validate=False))


@pytest.mark.parametrize("case",["not_tensor","shape","empty_batch","small_rows","small_columns","wrong_components",
    "integer","half","diagonal","validate_int","validate_tensor","nan","inf","inverted"])
def test_guards(case):
    vertices=grid(3,4);kwargs={}
    if case=="not_tensor":vertices=[]
    elif case=="shape":vertices=vertices[0]
    elif case=="empty_batch":vertices=vertices[:0]
    elif case=="small_rows":vertices=vertices[:,:1]
    elif case=="small_columns":vertices=vertices[:,:,:1]
    elif case=="wrong_components":vertices=vertices[...,:1]
    elif case=="integer":vertices=vertices.long()
    elif case=="half":vertices=vertices.half()
    elif case=="diagonal":kwargs["diagonal"]="other"
    elif case=="validate_int":kwargs["validate"]=1
    elif case=="validate_tensor":kwargs["validate"]=torch.tensor(True)
    elif case=="nan":vertices[0,1,1,0]=float("nan")
    elif case=="inf":vertices[0,1,1,0]=float("inf")
    elif case=="inverted":vertices[...,0].neg_()
    with pytest.raises(ValueError):api()(vertices,**kwargs)
