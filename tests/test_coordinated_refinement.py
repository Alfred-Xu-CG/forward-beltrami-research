import numpy as np
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries


def generic_corners(v):
    v=v.detach().numpy()
    a,b,c,d=v[:,:-1,:-1],v[:,:-1,1:],v[:,1:,1:],v[:,1:,:-1]
    det=lambda p,q,r:(q-p)[...,0]*(r-p)[...,1]-(q-p)[...,1]*(r-p)[...,0]
    return np.stack([det(*t) for t in ((a,b,d),(a,b,c),(d,b,c),(a,c,d))],-1)*(v.shape[1]-1)*(v.shape[2]-1)


def independent_p1(v, queries, diagonal):
    v=v.detach().numpy();rows,columns=v.shape[1:3];out=[]
    for x,y in queries:
        row=min(rows-2,int(y*(rows-1)));col=min(columns-2,int(x*(columns-1)))
        source=np.array([[col/(columns-1),row/(rows-1)],[(col+1)/(columns-1),row/(rows-1)],
            [(col+1)/(columns-1),(row+1)/(rows-1)],[col/(columns-1),(row+1)/(rows-1)]])
        target=np.stack((v[0,row,col],v[0,row,col+1],v[0,row+1,col+1],v[0,row+1,col]))
        for ids in (((0,1,2),(0,2,3)) if diagonal=="ac" else ((0,1,3),(1,2,3))):
            weights=np.linalg.solve(np.vstack((source[list(ids)].T,np.ones(3))),[x,y,1.])
            if weights.min()>=-1e-12:
                out.append(weights@target[list(ids)]);break
        else:raise AssertionError("source triangle not found")
    return np.asarray(out)


@pytest.mark.parametrize("diagonal",["ac","bd"])
@pytest.mark.parametrize("factor",[2,3,5])
def test_irregular_convex_quads_same_function_and_normalized_corners(diagonal,factor):
    # Two irregular legal convex polygons, unrelated to any optimizer decoder.
    for quad in ([[[-.3,.1],[1.2,-.2]],[[.1,1.4],[1.5,1.1]]],
                 [[[0.,0.],[1.4,.2]],[[-.1,.9],[.6,1.8]]]):
        v=torch.tensor([quad],dtype=torch.float64)
        before=generic_corners(v);assert before.min()>0
        fine=refine_p1_vertices(v,factor,diagonal)
        np.testing.assert_allclose(generic_corners(fine).min(),before.min(),rtol=2e-14,atol=2e-14)
        q=np.random.default_rng(710).uniform(0,1,(71,2))
        np.testing.assert_allclose(independent_p1(fine,q,diagonal),independent_p1(v,q,diagonal),atol=2e-15,rtol=2e-14)
        torch.testing.assert_close(fine[:,::factor,::factor],v,rtol=0,atol=0)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_non_square_batched_grid_and_vertex_vjp(diagonal):
    torch.manual_seed(73)
    y,x=torch.meshgrid(torch.linspace(0,1,4,dtype=torch.float64),torch.linspace(0,1,6,dtype=torch.float64),indexing="ij")
    base=torch.stack((x,y),-1)[None].repeat(2,1,1,1)
    v=(base+.002*torch.randn_like(base)).requires_grad_()
    fine=refine_p1_vertices(v,3,diagonal)
    assert fine.shape==(2,10,16,2)
    q=torch.rand(1,23,2,dtype=torch.float64)
    old=p1_map_at_queries(v,q,diagonal);new=p1_map_at_queries(fine,q,diagonal)
    torch.testing.assert_close(old,new,rtol=1e-13,atol=1e-14)
    upstream=torch.randn_like(old)
    oldgrad,=torch.autograd.grad((old*upstream).sum(),v,retain_graph=True)
    newgrad,=torch.autograd.grad((new*upstream).sum(),v)
    torch.testing.assert_close(oldgrad,newgrad,rtol=1e-12,atol=1e-13)


@pytest.mark.parametrize("factor",[0,-1,1.5,True])
def test_invalid_integer_factors_rejected(factor):
    with pytest.raises(ValueError):refine_p1_vertices(torch.zeros(1,2,2,2,dtype=torch.float64),factor)


def test_factor_one_copy_and_invalid_geometry():
    v=torch.randn(1,3,4,2,dtype=torch.float64,requires_grad=True)
    fine=refine_p1_vertices(v,1)
    torch.testing.assert_close(fine,v,rtol=0,atol=0)
    assert fine.data_ptr()!=v.data_ptr()
    with pytest.raises(ValueError):refine_p1_vertices(v.detach()*float("nan"),2)
    with pytest.raises(ValueError):refine_p1_vertices(v.detach().long(),2)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_rounding_does_not_inherit_an_unrepresentably_thin_margin(diagonal):
    # One-ULP height is positive before refinement, but its midpoint cannot be
    # represented at this translated ordinate. Fresh actual-corner checks are
    # mandatory; the theorem is explicitly NOT a rounded-table certificate.
    lo=torch.tensor(.5,dtype=torch.float64)
    hi=torch.nextafter(lo,torch.tensor(1.,dtype=torch.float64))
    v=torch.tensor([[[[0.,lo],[1.,lo]],[[0.,hi],[1.,hi]]]],dtype=torch.float64)
    assert generic_corners(v).min()>0
    assert generic_corners(refine_p1_vertices(v,2,diagonal)).min()==0
