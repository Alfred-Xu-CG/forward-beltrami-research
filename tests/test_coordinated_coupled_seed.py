import itertools
import pytest
import torch
from torch.nn import functional as F

from qcopt.neural_bijection.dense.coordinated_coupled_seed import (
    finite_displacement_cost,couple_cost_volume,construct_safe_seed,build_coupled_mind_seed)
from qcopt.neural_bijection.dense.coordinated_stiffness import DirichletGalerkinStiffness
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants


def identity(side):
    axis=torch.arange(side,dtype=torch.float64)/(side-1)
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    return torch.stack((x,y),-1)[None]


def volume(mask=None):
    generator=torch.Generator().manual_seed(206)
    fixed=torch.rand(1,3,8,8,generator=generator,dtype=torch.float64)
    moving=torch.rand(1,3,8,8,generator=generator,dtype=torch.float64)
    mask=torch.ones(1,1,8,8,dtype=torch.float64) if mask is None else mask
    a=torch.tensor([[1.13,.07],[-.04,.97]],dtype=torch.float64)
    b=torch.tensor([-.08,.02],dtype=torch.float64)
    result=finite_displacement_cost(fixed,moving,mask,a,b,proposal_side=5,label_radius=2,label_batch=3)
    return result,(fixed,moving,mask,a,b)


def sample(image,query):
    return F.grid_sample(image,(2*query-1)[None,None],align_corners=False,padding_mode='zeros')[0,:,0]


def test_literal_cost_query_coordinates_original_oob_weights_and_label_order():
    mask=torch.ones(1,1,8,8,dtype=torch.float64);mask[:,:,:4,:4]=.3
    v,(fixed,moving,mask,a,b)=volume(mask)
    expected_labels=[(0,0)]+[p for p in itertools.product(range(-2,3),repeat=2) if p!=(0,0)]
    assert v.labels.tolist()==[list(p) for p in expected_labels]
    patch=torch.tensor(list(itertools.product((-1,0,1),repeat=2)),dtype=torch.float64)/8
    expected=torch.empty_like(v.costs);weights=[]
    for node,(iy,ix) in enumerate(itertools.product(range(1,4),repeat=2)):
        q=torch.tensor([ix/4,iy/4],dtype=torch.float64)+patch
        w=sample(mask,q)[0];weights.append(w.sum()/9)
        for label,k in enumerate(expected_labels):
            query=q+torch.tensor(k,dtype=torch.float64)/8
            image=(sample(moving,query)-sample(fixed,q)).abs().mean(0)
            original=query@a.T+b
            outside=(original.clamp(max=0).square()+(original-1).clamp(min=0).square()).sum(-1)
            expected[label,node]=((image+outside)*w).sum()/w.sum()
    torch.testing.assert_close(v.costs,expected,rtol=0,atol=3e-16)
    torch.testing.assert_close(v.weights,torch.stack(weights),rtol=0,atol=0)
    assert v.diagnostics['cost_volume_bytes']==25*9*8


def test_empty_nodes_keep_constant_coupling_mass_and_first_ties():
    mask=torch.zeros(1,1,8,8,dtype=torch.float64);mask[:,:,1:3,1:3]=1
    v,_=volume(mask)
    assert (v.weights==0).any() and torch.equal(v.costs[:,v.weights==0],torch.zeros_like(v.costs[:,v.weights==0]))
    coupled=couple_cost_volume(v,fine_side=17)
    assert torch.isfinite(coupled.displacement).all()
    zero=torch.zeros(1,2,8,8,dtype=torch.float64);one=torch.ones(1,1,8,8,dtype=torch.float64)
    tie=finite_displacement_cost(zero,zero,one,torch.eye(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),proposal_side=5,label_radius=1)
    result=couple_cost_volume(tie,fine_side=17)
    assert torch.count_nonzero(result.selected_labels)==0 and torch.count_nonzero(result.displacement)==0


def test_screened_solution_literal_dense_and_descent_at_each_fixed_coupling():
    v,_=volume();result=couple_cost_volume(v,fine_side=17)
    diag=result.diagnostics
    assert diag['two_component_screened_solves']==diag['label_minimizations']==13
    assert diag['scalar_screened_systems']==26 and len(diag['blocks'])==12
    for block in diag['blocks']:
        assert block['after_labels']['total']<=block['before']['total']+1e-14
        assert block['after_continuous']['total']<=block['after_labels']['total']+1e-14
    metric=DirichletGalerkinStiffness(17,5,weight=1.,dtype=torch.float64,device='cpu')
    basis=torch.eye(9,dtype=torch.float64).reshape(9,3,3)
    dense=metric.apply(basis).reshape(9,9).T
    z=v.labels[result.selected_labels.reshape(-1)].double().T
    expected=torch.linalg.solve(2*torch.eye(9,dtype=torch.float64)+3*v.denominator/64*dense,2*z.T).T.reshape(2,3,3)
    torch.testing.assert_close(result.displacement,expected,rtol=5e-14,atol=5e-15)


def test_safe_constructor_exact_identity_boundary_and_one_frozen_target():
    reference=identity(17)
    zero=construct_safe_seed(reference,torch.zeros(2,3,3,dtype=torch.float64),image_side=16)
    assert torch.equal(zero.vertices,reference)
    displacement=torch.zeros(2,3,3,dtype=torch.float64)
    displacement[0]=.5;displacement[1]=-.25
    result=construct_safe_seed(reference,displacement,image_side=16)
    target=reference+F.interpolate(F.pad(displacement,(1,1,1,1))[None],size=(17,17),mode='bilinear',align_corners=True).permute(0,2,3,1)/16
    torch.testing.assert_close(result.vertices,target,rtol=0,atol=0)
    assert len(result.diagnostics['steps'])==16 and all(s['scale']==1. for s in result.diagnostics['steps'])
    assert result.diagnostics['actual_minimum_corner_ratio']>.001
    for actual,expected in ((result.vertices[:,0],reference[:,0]),(result.vertices[:,-1],reference[:,-1]),
                            (result.vertices[:,:,0],reference[:,:,0]),(result.vertices[:,:,-1],reference[:,:,-1])):
        assert torch.equal(actual,expected)
    assert torch.count_nonzero(result.coefficients_pixels[:,0])==0


def test_strong_folded_proposal_never_exports_unsafe_rounded_map():
    reference=identity(17)
    axis=torch.arange(3);checker=(-1.)**(axis[:,None]+axis[None,:])
    displacement=torch.stack((8*checker,7*checker)).double()
    try:
        result=construct_safe_seed(reference,displacement,image_side=16)
    except (RuntimeError,ValueError):
        return  # Explicit numerical strict-margin failure is the declared policy.
    assert (q1_corner_determinants(result.vertices)/q1_corner_determinants(reference)>.001).all()
    assert result.diagnostics['raw_target_nonpositive_corner_count']>0


def test_reject_unfrozen_inputs_empty_support_nonidentity_and_wrong_production_shape():
    _,(f,m,mask,a,b)=volume()
    with pytest.raises(ValueError):finite_displacement_cost(f.requires_grad_(),m,mask,a,b)
    with pytest.raises(ValueError):finite_displacement_cost(f.detach(),m,mask*0,a,b)
    with pytest.raises(ValueError):construct_safe_seed(identity(17)+.01,torch.zeros(2,3,3,dtype=torch.float64))
    with pytest.raises(ValueError):build_coupled_mind_seed(f.detach(),m,mask,a,b,identity(17))
