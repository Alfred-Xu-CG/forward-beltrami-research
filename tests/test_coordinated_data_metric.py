import copy
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.coordinated_data_metric import (
    FrozenScalarP1Rows, DataAwareMetric, pcg, optimize_data_metric_fiber,
    optimize_timed_adam_fiber,
)
from qcopt.neural_bijection.dense.coordinated_fixed_sampling import FrozenP1Evaluator
from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers
from tools.coordinated_real_case import Evidence
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences


def identity(side=5):
    y,x=torch.meshgrid(torch.linspace(0,1,side,dtype=torch.float64),torch.linspace(0,1,side,dtype=torch.float64),indexing='ij')
    return torch.stack((x,y),-1)[None]


def evidence(side=9, points=True):
    torch.manual_seed(943)
    fixed=.1+.8*torch.rand(1,1,side,side)
    moving=.1+.8*torch.rand_like(fixed)
    A=torch.tensor([[1.15,.07],[-.03,.94]],dtype=torch.float64)
    b=torch.tensor([-.06,.02],dtype=torch.float64)
    matches=None
    if points:
        q=torch.tensor([[0.,0.],[.13,.27],[.42,.66],[1.,1.]],dtype=torch.float64)
        matches=ImageCorrespondences(q,(q+.02).clamp(0,1),torch.tensor([.1,.3,.4,.2],dtype=torch.float64))
    return Evidence(fixed,moving,A,b,'mind',3.,1.,1e-4,fixed_mask=torch.ones_like(fixed),
        interpolation='p1_ac',matches=matches,match_weight=.2 if points else 0.,
        strain_model='p1_arap',mind_frame='shared_affine')


def test_rows_literal_forward_transpose_and_boundary_trace():
    q=torch.tensor([[[0.,0.],[.5,0.],[1.,1.],[.13,.27],[.5,.5],[.99,.45]]],dtype=torch.float64)
    rows=FrozenScalarP1Rows(5,q)
    c=torch.randn(1,3,3,dtype=torch.float64)
    fine=F.pad(c,(1,1,1,1))
    y=torch.stack((fine,torch.zeros_like(fine)),-1)
    expected=FrozenP1Evaluator(5,5,q,'ac')(y)[...,0].reshape(-1)
    assert torch.allclose(rows.apply(c),expected,atol=1e-15,rtol=1e-15)
    cot=torch.randn(len(expected),dtype=torch.float64)
    assert torch.allclose((rows.apply(c)*cot).sum(),(c*rows.transpose(cot)).sum(),atol=1e-15,rtol=1e-15)
    dense=torch.stack([rows.apply(torch.eye(9,dtype=torch.float64)[i].reshape(1,3,3)) for i in range(9)],1)
    assert torch.allclose(rows.row_norm_squared,dense.square().sum(1),atol=1e-15)
    assert torch.equal(rows.row_norm_squared[:3],torch.zeros(3,dtype=torch.float64))


def test_metric_actual_sampler_slopes_point_curvature_oob_and_explicit_spd():
    item=evidence();Y=identity();Y[:,1:-1,1:-1,0]+=.012
    metric=DataAwareMetric(item,Y,(1.,0.))
    query=metric.query.detach().requires_grad_()
    warped=F.grid_sample(item.moving_feature,(2*query-1).to(torch.float32),align_corners=False,padding_mode='zeros')
    slopes=[]
    for channel in range(8):
        slopes.append(torch.autograd.grad(warped[:,channel].sum(),query,retain_graph=True)[0][...,0])
    j=torch.stack(slopes,1)
    expected=(j.square()/(item.fixed_feature-warped).abs().double().clamp_min(1e-3)).sum(1).reshape(-1)/8/item.denominator.double()
    assert torch.allclose(metric.image_diagonal,expected.detach(),atol=1e-14,rtol=1e-14)
    qworld=metric.query@item.matrix.T+item.offset
    ae=item.matrix[:,0]
    expected_oob=2*((qworld<0)|(qworld>1)).double().mul(ae.square()).sum(-1).reshape(-1)/item.denominator.double()
    assert torch.equal(metric.oob_diagonal,expected_oob)
    c=torch.zeros(1,3,3,dtype=torch.float64,requires_grad=True)
    point_energy=lambda v:item.match_weight*item.matches(Y+F.pad(v,(1,1,1,1))[...,None]*Y.new_tensor([1.,0.]),item.matrix,'p1_ac')
    hp=torch.autograd.functional.hessian(point_energy,c).reshape(9,9)
    eye=torch.eye(9,dtype=torch.float64)
    L=torch.stack([metric.point_rows.apply(v.reshape(1,3,3)) for v in eye],1)
    assert torch.allclose(hp,L.T@(metric.point_diagonal[:,None]*L),atol=2e-12,rtol=2e-12)
    H=torch.stack([metric.apply(v.reshape(1,3,3)).reshape(-1) for v in eye],1)
    assert torch.allclose(H,H.T,atol=1e-12,rtol=1e-13)
    assert float(torch.linalg.eigvalsh(H).min())>0
    K=torch.stack([metric.stiffness.apply(v.reshape(1,3,3)).reshape(-1) for v in eye],1)
    assert abs(metric.gamma-float(torch.trace(H-K)/9))<1e-12
    rhs=torch.randn(1,3,3,dtype=torch.float64)
    x,status=pcg(metric.apply,rhs,metric.precondition,rtol=.1,max_steps=20)
    assert status['converged'] and status['relative_residual']<=.1
    assert torch.linalg.vector_norm(metric.apply(x)-rhs)/torch.linalg.vector_norm(rhs)<=.100000001
    _,capped=pcg(metric.apply,rhs,metric.precondition,rtol=1e-16,max_steps=1)
    assert not capped['converged'] and capped['reason']=='iteration_cap'


class Tick:
    def __init__(self):self.t=0.
    def __call__(self):self.t+=.1;return self.t


@pytest.mark.parametrize('arm',['metric','adam'])
def test_timed_stages_retain_floor_boundary_and_honest_counts(arm):
    Y=identity();item=evidence()
    if arm=='metric':out=optimize_data_metric_fiber(Y,item,seconds=.25,clock=Tick())
    else:out=optimize_timed_adam_fiber(Y,item,learning_rate=.00025,seconds=.25,clock=Tick())
    assert out.elapsed_seconds>=.25 and out.stop_reason=='time_budget'
    assert out.counts['gradient_steps']>=1
    assert out.final_objective<=out.initial_objective
    assert torch.equal(out.vertices[:,0],Y[:,0]) and torch.equal(out.vertices[:,-1],Y[:,-1])
    assert torch.equal(out.vertices[:,:,0],Y[:,:,0]) and torch.equal(out.vertices[:,:,-1],Y[:,:,-1])
    assert out.minimum_contracted_slack>0
    if arm=='metric':assert out.counts['descriptor_vjps']==8*out.counts['metric_refreshes']


def test_prefix_hook_raw_map_and_default_exact(tmp_path):
    from tests.test_coordinated_shared_affine_evidence import configuration
    from tools.coordinated_real_case import optimize
    args=configuration(tmp_path,'analytic');args.mind_frame='shared_affine'
    full=[];ordinary=optimize(args,lambda y,s,t:full.append((y.clone(),s)))
    args.output=tmp_path/'prefix.npz';prefix=[]
    short=optimize(args,lambda y,s,t:prefix.append((y.clone(),s)),maximum_stages=2)
    assert short['partial_prefix']['accepted_stages']==2 and short['gradient_steps']==2
    assert short['trace']==ordinary['trace'][:len(short['trace'])]
    for (a,sa),(b,sb) in zip(full,prefix):assert torch.equal(a,b) and sa==sb
    assert short['final']['total']==min([short['initial']['total']]+[s['accepted_full_total'] for s in short['stages']])
    args.output=tmp_path/'default_again.npz';again=optimize(args,maximum_stages=None)
    for key in ('initial','final','trace','stages','gradient_steps','failed_trials','objective_evaluations'):
        assert ordinary[key]==again[key]
    assert 'partial_prefix' not in ordinary and 'partial_prefix' not in again


def test_original_adam_active_contracted_bound_is_legal():
    from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
    Y=identity()
    def objective(vertices):
        value=-vertices[:,1:-1,1:-1,0].sum()
        return value,dict(image=value)
    out=optimize_timed_adam_fiber(Y,objective,learning_rate=1.,seconds=.25,clock=Tick())
    assert out.stop_reason=='time_budget' and out.counts['failed_trials']==0
    assert out.final_objective<out.initial_objective
    assert abs(out.minimum_contracted_slack)<1e-13
    assert float((q1_corner_determinants(out.vertices)*16).min())>.001


def test_metric_armijo_rejects_rounded_equal_objective_thirteen_trials():
    original=evidence()
    class ConstantValueEvidence(type(original)):
        def __call__(self,vertices):
            value=vertices[:,1:-1,1:-1,0].sum()
            loss=1.+value-value.detach()
            return loss,dict(image=loss)
    item=object.__new__(ConstantValueEvidence);item.__dict__.update(original.__dict__)
    Y=identity()
    out=optimize_data_metric_fiber(Y,item,seconds=.25,clock=Tick())
    assert out.stop_reason=='line_search_exhausted'
    assert torch.equal(out.vertices,Y) and out.final_objective==out.initial_objective
    assert out.counts['accepted_steps']==0 and out.counts['backtracks']==12
    assert len(out.trace[0]['trials'])==13
