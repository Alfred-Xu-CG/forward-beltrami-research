import math
import pytest
import torch
from torch.nn import functional as F
from qcopt.neural_bijection.dense.coordinated_stiffness import (
    dst1_orthonormal,DirichletGalerkinStiffness,optimize_stiffness_fiber)
from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants


def identity(side):
    y,x=torch.meshgrid(torch.linspace(0,1,side,dtype=torch.float64),
                       torch.linspace(0,1,side,dtype=torch.float64),indexing='ij')
    return torch.stack((x,y),-1)[None]


def tridiagonal(size):
    return 2*torch.eye(size,dtype=torch.float64)-torch.diag(torch.ones(size-1,dtype=torch.float64),1)-torch.diag(torch.ones(size-1,dtype=torch.float64),-1)


def literal_matrices(fine,coarse):
    x=torch.arange(1,fine-1,dtype=torch.float64)/(fine-1)
    centers=torch.arange(1,coarse-1,dtype=torch.float64)/(coarse-1)
    b=(1-(x[:,None]-centers[None,:]).abs()*(coarse-1)).clamp_min(0)
    t=tridiagonal(fine-2); eye=torch.eye(fine-2,dtype=torch.float64)
    k=torch.kron(t,eye)+torch.kron(eye,t)
    p=torch.kron(b,b)
    return p,k,p.T@k@p


@pytest.mark.parametrize('size',[1,2,3,4,7,14])
def test_dst_literal_orthonormal_and_roundtrip(size):
    torch.manual_seed(size); x=torch.randn(3,size,dtype=torch.float64)
    indices=torch.arange(1,size+1,dtype=x.dtype)
    sine=math.sqrt(2/(size+1))*torch.sin(math.pi*indices[:,None]*indices[None,:]/(size+1))
    torch.testing.assert_close(dst1_orthonormal(x),x@sine,atol=5e-15,rtol=5e-15)
    torch.testing.assert_close(dst1_orthonormal(dst1_orthonormal(x)),x,atol=5e-15,rtol=5e-15)


@pytest.mark.parametrize('fine,coarse',[(3,3),(5,3),(9,3),(9,5),(9,9),(7,4),(13,4)])
def test_exact_raw_prolongation_galerkin_spectrum_inverse(fine,coarse):
    torch.manual_seed(fine+coarse); p,k,galerkin=literal_matrices(fine,coarse)
    metric=DirichletGalerkinStiffness(fine,coarse)
    c=torch.randn(1,coarse-2,coarse-2,dtype=torch.float64)
    torch.testing.assert_close(metric.prolong(c)[:,1:-1,1:-1].flatten(),p@c.flatten(),rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(metric.apply(c).flatten(),3*galerkin@c.flatten(),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(metric.solve(c).flatten(),torch.linalg.solve(3*galerkin,c.flatten()),rtol=1e-12,atol=1e-12)
    assert (c*metric.solve(c)).sum()>0
    torch.testing.assert_close(metric.apply(metric.solve(c)),c,rtol=1e-12,atol=1e-12)


def jacobians(y):
    side=y.shape[1]; a,b=y[:,:-1,:-1],y[:,:-1,1:]
    d,c=y[:,1:,:-1],y[:,1:,1:]
    return torch.stack((torch.stack((b-a,c-b),-1),torch.stack((c-d,d-a),-1)),-3)*(side-1)


def test_frozen_rotation_majorizer_touches_value_gradient_and_has_exact_k():
    torch.manual_seed(91); base=identity(5)
    base[:,1:-1,1:-1]+=torch.randn(1,3,3,2,dtype=base.dtype)*.012
    j=jacobians(base); u,s,vh=torch.linalg.svd(j); rotations=(u@vh).detach()
    frozen=lambda y:.5*(jacobians(y)-rotations).square().sum((-1,-2)).mean()
    y=base.clone().requires_grad_(True)
    value=p1_arap_energy(y,validate=True)
    torch.testing.assert_close(value,frozen(y),atol=1e-15,rtol=1e-13)
    g=torch.autograd.grad(value,y)[0]; gf=torch.autograd.grad(frozen(y),y)[0]
    torch.testing.assert_close(g,gf,atol=2e-15,rtol=1e-12)
    e=torch.tensor([1.,0.],dtype=base.dtype)
    energy=lambda c:frozen(base+F.pad(c.reshape(1,3,3),(1,1,1,1))[...,None]*e)
    h=torch.autograd.functional.hessian(energy,torch.zeros(9,dtype=base.dtype))
    _,k,_=literal_matrices(5,5)
    torch.testing.assert_close(h,k,atol=2e-14,rtol=1e-13)
    changed=base.clone(); changed[:,1:-1,1:-1]+=.003*torch.randn(1,3,3,2,dtype=base.dtype)
    assert p1_arap_energy(changed)<=frozen(changed)+1e-15


def test_active_constraint_original_anchor_and_boundary_are_preserved():
    base=identity(9); base[:,1:-1,1:-1,0]+=.005
    target=base.clone(); target[:,1:-1,1:-1,0]+=2.
    objective=lambda y:((y-target).square().sum()/2,{})
    output=optimize_stiffness_fiber(base,objective,coefficient_side=5,maximum_gradients=8)
    assert output.counts['accepted_steps']>0 and output.trace[0]['initial_alpha']<1
    q0=q1_corner_determinants(base)*64; floor=.001+.05*(q0-.001)
    assert torch.all(q1_corner_determinants(output.vertices)*64>floor)
    torch.testing.assert_close(output.vertices[:,0],base[:,0],rtol=0,atol=0)
    torch.testing.assert_close(output.vertices[:,:,0],base[:,:,0],rtol=0,atol=0)
    metric=DirichletGalerkinStiffness(9,5)
    literal=base+metric.prolong(output.coefficients)[...,None]*torch.tensor([1.,0.])
    torch.testing.assert_close(output.vertices,literal,rtol=0,atol=0)
    for step in output.trace:
        assert step['directional_derivative']<0
        for trial in step['trials']:
            if trial['accepted']: assert trial['total']<=trial['armijo_rhs']
    assert output.counts['objective_evaluations']==output.counts['gradient_steps']+output.counts['trial_evaluations']


def test_deliberate_overshoot_rejects_without_changing_map():
    base=identity(5); target=base.clone(); target[:,1:-1,1:-1,0]+=.001
    objective=lambda y:(1e8*(y-target).square().sum()/2,{})
    output=optimize_stiffness_fiber(base,objective,coefficient_side=5,maximum_gradients=2,max_backtracks=0)
    assert output.stop_reason=='line_search_exhausted' and output.counts['accepted_steps']==0
    torch.testing.assert_close(output.vertices,base,rtol=0,atol=0)


def test_batch_one_and_aligned_grids_are_explicit():
    with pytest.raises(ValueError): DirichletGalerkinStiffness(8,4)
    with pytest.raises(ValueError): optimize_stiffness_fiber(identity(5).expand(2,-1,-1,-1),lambda x:(x.sum(),{}),coefficient_side=5)
