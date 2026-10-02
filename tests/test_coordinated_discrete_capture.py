"""Literal finite-label costs check frames, ties and immutable evidence."""
import itertools
import math
import torch
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(dtype=torch.float64):
    y,x=torch.meshgrid(torch.arange(128,dtype=dtype),torch.arange(128,dtype=dtype),indexing='ij')
    moving=torch.stack(((x>=66).to(dtype),(y+.5)/128))[None]
    fixed=torch.stack(((x>=64).to(dtype),(y+.5)/128))[None]
    return fixed,moving,torch.ones(1,1,128,128,dtype=dtype),torch.eye(2,dtype=dtype),torch.zeros(2,dtype=dtype)


def sample(field,q):
    """Independent align-false, zero-padding bilinear raster formula."""
    x,y=q[0]*128-.5,q[1]*128-.5
    j,i=math.floor(x),math.floor(y);tx,ty=x-j,y-i
    result=np.zeros(field.shape[0],dtype=np.float64)
    for r,c,w in ((i,j,(1-tx)*(1-ty)),(i,j+1,tx*(1-ty)),(i+1,j,(1-tx)*ty),(i+1,j+1,tx*ty)):
        if 0<=r<128 and 0<=c<128:result+=w*field[:,r,c]
    return result


def literal(f,m,mask,A,b,q):
    offsets=list(itertools.product((-1,0,1),repeat=2))
    labels=[(0,0)]+[k for k in itertools.product(range(-4,5),repeat=2) if k!=(0,0)]
    points=[q+np.asarray(s)/128 for s in offsets]
    weights=np.array([sample(mask,p)[0] for p in points]);den=weights.sum()
    costs=[]
    for label in labels:
        terms=[]
        for p in points:
            t=A@p+b+np.asarray(label)/128
            distance=np.maximum(-t,0)+np.maximum(t-1,0)
            terms.append(np.abs(sample(f,p)-sample(m,t)).mean()+np.sum(distance**2))
        costs.append(float(weights@terms/den) if den else 0.)
    winner=int(np.argmin(costs))
    return labels[winner],costs[winner],den


def test_positive_two_moving_pixel_shift_captured_despite_zero_local_derivative():
    from qcopt.neural_bijection.dense.coordinated_discrete_capture import discrete_capture_proposal
    fixed,moving,mask,A,b=inputs()
    q=torch.tensor([.5,.5],dtype=torch.float64)
    offsets=torch.tensor(list(itertools.product((-1,0,1),repeat=2)),dtype=torch.float64)/128
    displacement=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    grid=(2*(q+offsets+displacement)-1)[None,None]
    warped=torch.nn.functional.grid_sample(moving,grid,align_corners=False,padding_mode='zeros')
    f=torch.nn.functional.grid_sample(fixed,(2*(q+offsets)-1)[None,None],align_corners=False,padding_mode='zeros')
    local=(f-warped).abs().mean()
    assert local.item()==pytest.approx(.25)
    assert torch.equal(torch.autograd.grad(local,displacement)[0],torch.zeros(2,dtype=torch.float64))
    result=discrete_capture_proposal(fixed,moving,mask,A,b)
    assert result.chosen_labels[0,16,16].tolist()==[2,0]
    assert result.chosen_cost[0,16,16].item()==0
    assert result.proposal[0,16,16].tolist()==[2/128,0]
    assert not result.proposal.requires_grad and result.proposal.shape==(1,33,33,2)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_literal_cost_and_anisotropic_affine_inverse_not_pixel_axis_division(dtype):
    from qcopt.neural_bijection.dense.coordinated_discrete_capture import discrete_capture_proposal
    gen=torch.Generator().manual_seed(29)
    fixed=torch.rand(1,3,128,128,generator=gen,dtype=dtype)
    moving=torch.rand(1,3,128,128,generator=gen,dtype=dtype)
    mask=torch.rand(1,1,128,128,generator=gen,dtype=dtype)
    A=torch.tensor([[.8,.1],[.05,.9]],dtype=torch.float64);b=torch.tensor([.03,-.02],dtype=torch.float64)
    old=[v.clone() for v in (fixed,moving,mask,A,b)]
    result=discrete_capture_proposal(fixed,moving,mask,A,b,geometry_dtype=torch.float64)
    for i,j in [(9,11),(16,17),(2,3)]:
        label,cost,den=literal(fixed[0].numpy(),moving[0].numpy(),mask[0].numpy(),A.numpy(),b.numpy(),np.array([j/32,i/32]))
        assert result.chosen_labels[0,i,j].tolist()==list(label)
        assert result.chosen_cost[0,i,j].item()==pytest.approx(cost,abs=3e-6 if dtype==torch.float32 else 3e-14)
        assert result.patch_denominator[0,i,j].item()==pytest.approx(den,abs=2e-6 if dtype==torch.float32 else 3e-14)
        d=np.asarray(label)/128;det=.8*.9-.1*.05
        p=np.array([(.9*d[0]-.1*d[1])/det,(-.05*d[0]+.8*d[1])/det])
        assert result.proposal[0,i,j].numpy()==pytest.approx(p,abs=2e-16)
        assert result.proposal[0,i,j].numpy()@A.numpy().T==pytest.approx(d,abs=2e-16)
    assert all(torch.equal(a,b) for a,b in zip(old,(fixed,moving,mask,A,b),strict=True))


def test_zero_first_ties_empty_mask_and_exact_zero_boundary():
    from qcopt.neural_bijection.dense.coordinated_discrete_capture import discrete_capture_proposal
    fixed,moving,mask,A,b=inputs();fixed.zero_();moving.zero_()
    result=discrete_capture_proposal(fixed,moving,mask,A,b)
    assert torch.count_nonzero(result.chosen_labels)==0 and torch.count_nonzero(result.proposal)==0
    # Unit coordinates0/1 hit HALF pixels under align_corners=False: no silent
    # align-true interpretation or source-rectangle query clipping.
    assert result.patch_denominator[0,0,0].item()==2.25
    assert result.patch_denominator[0,0,16].item()==4.5
    mask.zero_();empty=discrete_capture_proposal(fixed,moving,mask,A,b)
    assert empty.empty_patch.all() and torch.count_nonzero(empty.chosen_cost)==0
    assert torch.count_nonzero(empty.proposal)==0
    assert empty.diagnostics['label_count']==81 and empty.diagnostics['patch_sample_count']==9
    shifted=discrete_capture_proposal(*inputs())
    for edge in (shifted.proposal[:,0],shifted.proposal[:,-1],shifted.proposal[:,:,0],shifted.proposal[:,:,-1]):
        assert torch.count_nonzero(edge)==0


def test_original_mask_denominator_includes_outside_moving_and_corner_oob_sum_of_squares():
    from qcopt.neural_bijection.dense.coordinated_discrete_capture import discrete_capture_proposal
    fixed,moving,mask,A,b=inputs();fixed.zero_();moving.zero_();b=torch.tensor([.7,.7],dtype=torch.float64)
    result=discrete_capture_proposal(fixed,moving,mask,A,b)
    label,cost,den=literal(fixed[0].numpy(),moving[0].numpy(),mask[0].numpy(),A.numpy(),b.numpy(),np.array([.5,.5]))
    assert label==(-4,-4) and result.chosen_labels[0,16,16].tolist()==[-4,-4]
    assert den==result.patch_denominator[0,16,16].item()==9
    assert result.chosen_cost[0,16,16].item()==pytest.approx(cost,abs=1e-15)
    assert result.chosen_cost[0,16,16]>0  # none of the nine moving queries is in bounds
    assert 'no moving-overlap' in result.diagnostics['mask_scope']


def test_nonzero_equal_minima_follow_declared_xy_lexicographic_order():
    from qcopt.neural_bijection.dense.coordinated_discrete_capture import discrete_capture_proposal
    fixed,moving,mask,A,b=inputs();fixed[:,1].zero_();moving[:,1].zero_()
    result=discrete_capture_proposal(fixed,moving,mask,A,b)
    # Every y-label at x=+2 is equally perfect here; zero isn't a minimum.
    assert result.chosen_labels[0,16,16].tolist()==[2,-4]
    assert result.chosen_cost[0,16,16].item()==0


def test_production_eight_channel_tensorized_sampler_shapes(monkeypatch):
    from qcopt.neural_bijection.dense.coordinated_discrete_capture import discrete_capture_proposal
    fixed,moving,mask,A,b=inputs(torch.float32)
    fixed=fixed.repeat(1,4,1,1);moving=moving.repeat(1,4,1,1)
    calls=[];original=torch.nn.functional.grid_sample
    def observed(image,grid,**kwargs):
        calls.append((tuple(image.shape),tuple(grid.shape),dict(kwargs)))
        return original(image,grid,**kwargs)
    monkeypatch.setattr(torch.nn.functional,'grid_sample',observed)
    result=discrete_capture_proposal(fixed,moving,mask,A,b,geometry_dtype=torch.float64)
    assert [(a,b) for a,b,_ in calls]==[((1,8,128,128),(1,1089,9,2)),
        ((1,1,128,128),(1,1089,9,2)),((81,8,128,128),(81,1089,9,2))]
    assert all(k==dict(mode='bilinear',padding_mode='zeros',align_corners=False) for _,_,k in calls)
    assert result.proposal.dtype==torch.float64 and result.chosen_cost.dtype==torch.float64
    assert result.patch_denominator.dtype==torch.float32


@pytest.mark.parametrize('which',['fixed','moving','mask','matrix','offset'])
def test_trainable_inputs_rejected_no_argmin_vjp_claim(which):
    from qcopt.neural_bijection.dense.coordinated_discrete_capture import discrete_capture_proposal
    values=list(inputs());values[['fixed','moving','mask','matrix','offset'].index(which)].requires_grad_()
    with pytest.raises(ValueError,match='gradient'):discrete_capture_proposal(*values)


@pytest.mark.parametrize('bad',['size','batch','channels','mask_range','nan','singular','negative_det','geometry_dtype'])
def test_fixed_recipe_and_finite_positive_affine_guards(bad):
    from qcopt.neural_bijection.dense.coordinated_discrete_capture import discrete_capture_proposal
    values=list(inputs());kwargs={}
    if bad=='size':values[0]=values[0][...,:127]
    elif bad=='batch':values[0]=values[0].expand(2,-1,-1,-1)
    elif bad=='channels':values[1]=values[1][:,:1]
    elif bad=='mask_range':values[2][0,0,0,0]=1.1
    elif bad=='nan':values[1][0,0,0,0]=float('nan')
    elif bad=='singular':values[3].zero_()
    elif bad=='negative_det':values[3][0,0]=-1
    else:kwargs['geometry_dtype']=torch.float16
    with pytest.raises(ValueError):discrete_capture_proposal(*values,**kwargs)
