"""Independent tiny-matrix and actual-P1 checks for the fixed-budget fiber."""
import math
import pytest
import torch

from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from qcopt.neural_bijection.dense.coordinated_distortion_budget import (
    DistortionBudgetMetric,solve_budget_direction,optimize_distortion_budget_fiber)
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants


def identity(side):
    y,x=torch.meshgrid(torch.linspace(0,1,side,dtype=torch.float64),
        torch.linspace(0,1,side,dtype=torch.float64),indexing='ij')
    return torch.stack((x,y),-1)[None]


def literal_matrices(fine,coarse):
    x=torch.arange(1,fine-1,dtype=torch.float64)/(fine-1)
    centers=torch.arange(1,coarse-1,dtype=torch.float64)/(coarse-1)
    b=(1-(x[:,None]-centers[None,:]).abs()*(coarse-1)).clamp_min(0)
    n=fine-2;eye=torch.eye(n,dtype=torch.float64)
    t=2*eye-torch.diag(torch.ones(n-1,dtype=torch.float64),1)-torch.diag(torch.ones(n-1,dtype=torch.float64),-1)
    k=torch.kron(t,eye)+torch.kron(eye,t);p=torch.kron(b,b)
    return p,p.T@k@p,p.T@p/(fine*fine)


@pytest.mark.parametrize('fine,coarse',[(3,3),(5,3),(9,5),(9,9),(7,4),(13,4)])
def test_exact_galerkin_stiffness_and_physical_mass(fine,coarse):
    p,k,m=literal_matrices(fine,coarse);metric=DistortionBudgetMetric(fine,coarse)
    torch.manual_seed(fine+coarse);c=torch.randn(1,coarse-2,coarse-2,dtype=torch.float64)
    torch.testing.assert_close(metric.stiffness.apply(c).flatten(),k@c.flatten(),rtol=2e-12,atol=2e-13)
    torch.testing.assert_close(metric.apply_mass(c).flatten(),m@c.flatten(),rtol=2e-12,atol=2e-14)
    torch.testing.assert_close(metric.solve_mass(c).flatten(),torch.linalg.solve(m,c.flatten()),rtol=2e-12,atol=2e-12)
    assert float((c*metric.apply_mass(c)).sum())==pytest.approx(float(metric.prolong(c).square().mean()),rel=2e-13)


def independent_dense_solution(g,r,k,m,slack,step):
    tau=step/math.sqrt(float(g@torch.linalg.solve(m,g)))
    def solve(mu):
        d=torch.linalg.solve(m/tau+mu*k,-g-mu*r)
        return d,float(r@d+.5*d@k@d-slack)
    d,value=solve(0.)
    if value<=0:return d,0.,tau
    lo,hi=0.,1.
    while solve(hi)[1]>0:hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if solve(mid)[1]>0:lo=mid
        else:hi=mid
    return solve(hi)[0],hi,tau


@pytest.mark.parametrize('slack',[0.,.0001,10.])
def test_active_and_inactive_qp_against_independent_dense_solve(slack):
    metric=DistortionBudgetMetric(7,4);_,k,m=literal_matrices(7,4)
    g=torch.tensor([.8,-.4,.3,.2],dtype=torch.float64).reshape(1,2,2)
    r=torch.tensor([.04,.03,-.05,.07],dtype=torch.float64).reshape_as(g)
    d,record=solve_budget_direction(g,r,metric,remaining_budget=slack,physical_rms_step=.03)
    expected,mu,tau=independent_dense_solution(g.flatten(),r.flatten(),k,m,slack,.03)
    assert not record['numerical_failure'] and record['constraint_residual']<=0
    torch.testing.assert_close(d.flatten(),expected,rtol=2e-8,atol=2e-10)
    assert record['tau']==pytest.approx(tau,rel=1e-13)
    assert record['dual_bracket_probes']<=32 and record['dual_bisections']<=32
    if mu==0:
        assert record['multiplier']==0 and record['dual_bisections']==0
        assert float(metric.prolong(d).square().mean().sqrt())==pytest.approx(.03,rel=2e-13)
    else:
        assert record['bracket_lower']<=mu<=record['bracket_upper']*(1+1e-12)
    assert float((g*d).sum())<0


def test_zero_gradient_and_zero_ellipsoid_are_legal_stops():
    metric=DistortionBudgetMetric(5,5);zero=torch.zeros(1,3,3,dtype=torch.float64)
    d,record=solve_budget_direction(zero,zero,metric,remaining_budget=0.,physical_rms_step=.004)
    assert record['stop_reason']=='zero_data_gradient' and not record['numerical_failure'] and not d.any()
    d,record=solve_budget_direction(zero+1,zero,metric,remaining_budget=0.,physical_rms_step=.004)
    assert record['stop_reason']=='zero_ellipsoid' and not record['numerical_failure'] and not d.any()
    assert record['dual_bracket_probes']==0


def test_unbracketed_and_nonfinite_are_numerical_failures():
    metric=DistortionBudgetMetric(5,3)
    g=torch.ones(1,1,1,dtype=torch.float64);r=torch.full_like(g,1e-30)
    _,record=solve_budget_direction(g,r,metric,remaining_budget=0.,physical_rms_step=.004)
    assert record['stop_reason']=='dual_bracket_failure' and record['numerical_failure']
    assert record['dual_bracket_probes']==32
    _,record=solve_budget_direction(g*float('nan'),r,metric,remaining_budget=0.,physical_rms_step=.004)
    assert record['numerical_failure'] and record['stop_reason']=='nonfinite_gradient'


def test_feasible_overshot_dual_endpoint_need_not_descend():
    metric=DistortionBudgetMetric(5,3);g=torch.ones(1,1,1,dtype=torch.float64);r=-g
    # No bisections deliberately leaves an over-large FEASIBLE upper endpoint.
    d,record=solve_budget_direction(g,r,metric,remaining_budget=0.,physical_rms_step=.004,max_bisections=0)
    assert record['constraint_residual']<=0
    assert float((g*d).sum())>0
    assert record['stop_reason']=='finite_nondescent_direction' and not record['numerical_failure']


def legal_nonidentity(side=9):
    base=identity(side);x=base[...,0];y=base[...,1]
    base[...,0]+=.025*torch.sin(math.pi*x)*torch.sin(math.pi*y)
    # Preserve the literal perimeter instead of floating sin(pi) roundoff.
    original=identity(side)
    base[:,0]=original[:,0];base[:,-1]=original[:,-1];base[:,:,0]=original[:,:,0];base[:,:,-1]=original[:,:,-1]
    return base


def test_frozen_rotation_majorizer_uses_unweighted_k_at_nonzero_current_coefficients():
    base=legal_nonidentity();metric=DistortionBudgetMetric(9,5)
    torch.manual_seed(29);c=(torch.randn(1,3,3,dtype=torch.float64)*.0005).requires_grad_()
    e=torch.tensor([1.,0.],dtype=torch.float64);current=base+metric.prolong(c)[...,None]*e
    rvalue=p1_arap_energy(current);r=torch.autograd.grad(rvalue,c)[0]
    d=torch.randn_like(c)*.001;next_y=base+metric.prolong(c.detach()+d)[...,None]*e
    upper=rvalue.detach()+(r*d).sum()+.5*(d*metric.stiffness.apply(d)).sum()
    assert p1_arap_energy(next_y)<=upper+1e-16
    # Independent frozen SO(2) rotations give EXACT equality to that quadratic.
    def jac(y):
        a,b=y[:,:-1,:-1],y[:,:-1,1:];dd,cc=y[:,1:,:-1],y[:,1:,1:]
        return torch.stack((torch.stack((b-a,cc-b),-1),torch.stack((cc-dd,dd-a),-1)),-3)*8
    j=jac(current.detach());u,_,vh=torch.linalg.svd(j);rot=u@vh
    frozen=.5*(jac(next_y)-rot).square().sum((-1,-2)).mean()
    torch.testing.assert_close(upper,frozen,rtol=2e-13,atol=3e-16)


def test_actual_budget_floor_perimeter_original_anchor_and_counts():
    base=legal_nonidentity();target=identity(9)
    callback=lambda y:((y-target).square().mean(),{'strain':p1_arap_energy(y,validate=False)})
    budget=float(p1_arap_energy(base));out=optimize_distortion_budget_fiber(base,callback,
        coefficient_side=5,direction=(1.,0.),distortion_budget=budget,physical_rms_step=.004,maximum_gradients=6)
    assert not out.numerical_failure and out.counts['accepted_steps']>0
    assert out.final_objective<out.initial_objective and out.final_distortion<=budget
    q0=q1_corner_determinants(base)*64;floor=.001+.05*(q0-.001)
    assert (q1_corner_determinants(out.vertices)*64>floor).all()
    metric=DistortionBudgetMetric(9,5)
    torch.testing.assert_close(out.vertices,base+metric.prolong(out.coefficients)[...,None]*base.new_tensor([1.,0.]),rtol=0,atol=0)
    for pair in ((out.vertices[:,0],base[:,0]),(out.vertices[:,-1],base[:,-1]),(out.vertices[:,:,0],base[:,:,0]),(out.vertices[:,:,-1],base[:,:,-1])):
        assert torch.equal(*pair)
    assert out.counts['gradient_steps']==out.counts['data_vjps']==out.counts['distortion_vjps']
    assert out.counts['trial_attempts']==out.counts['accepted_steps']+out.counts['rejected_trials']
    for step in out.trace:
        for trial in step.get('trials',[]):
            if trial['accepted']:
                assert trial['distortion']<=budget and trial['objective']<step['objective']
                assert trial['objective']<=trial['armijo_rhs']
                assert trial['alpha']<=.99


def test_zero_budget_identity_retained_without_fake_completed_budget():
    base=identity(5)
    callback=lambda y:(-y[:,1:-1,1:-1,0].sum(),{'strain':p1_arap_energy(y,validate=False)})
    out=optimize_distortion_budget_fiber(base,callback,coefficient_side=5,direction=(1.,0.),
        distortion_budget=0.,physical_rms_step=.004,maximum_gradients=30)
    assert out.stop_reason=='zero_ellipsoid' and not out.numerical_failure
    assert out.counts['gradient_steps']==1 and out.counts['trial_attempts']==0
    assert torch.equal(out.vertices,base) and out.final_distortion==0


@pytest.mark.parametrize('bad_budget',[float('nan'),float('inf'),-1.])
def test_invalid_budget_rejected(bad_budget):
    with pytest.raises(ValueError):optimize_distortion_budget_fiber(identity(5),lambda y:None,
        coefficient_side=5,direction=(1.,0.),distortion_budget=bad_budget,physical_rms_step=.004)


@pytest.mark.parametrize('budget',[True,0,1.5])
def test_strict_integer_gradient_budget(budget):
    with pytest.raises(ValueError):optimize_distortion_budget_fiber(identity(5),lambda y:None,
        coefficient_side=5,direction=(1.,0.),distortion_budget=0.,physical_rms_step=.004,maximum_gradients=budget)


@pytest.mark.parametrize('backtracks',[0,12])
def test_finite_overshoot_rejection_is_not_numerical_failure(backtracks):
    base=identity(5);target=base.clone();target[:,1:-1,1:-1,0]+=.0001
    callback=lambda y:((y-target).square().sum()/2,{'strain':p1_arap_energy(y,validate=False)})
    out=optimize_distortion_budget_fiber(base,callback,coefficient_side=5,direction=(1.,0.),
        distortion_budget=1.,physical_rms_step=.02,maximum_gradients=1,max_backtracks=backtracks)
    assert not out.numerical_failure and out.counts['rejected_trials']>0
    assert out.counts['numerical_failures']==0
    assert out.counts['trial_attempts']==out.counts['accepted_steps']+out.counts['rejected_trials']
    assert out.counts['objective_evaluations']==out.counts['gradient_steps']+out.counts['trial_evaluations']
    if backtracks==0:
        assert out.stop_reason=='line_search_exhausted' and out.counts['accepted_steps']==0
        assert torch.equal(out.vertices,base)
    else:
        assert out.counts['accepted_steps']==1 and out.counts['backtracks']>0
        assert out.final_objective<out.initial_objective


def test_strict_computed_descent_blocks_rounded_equal_data_values():
    base=identity(5)
    callback=lambda y:(y[:,1:-1,1:-1,0].sum()+1e20,{'strain':p1_arap_energy(y,validate=False)})
    out=optimize_distortion_budget_fiber(base,callback,coefficient_side=5,direction=(1.,0.),
        distortion_budget=1.,physical_rms_step=.004,maximum_gradients=3,max_backtracks=2)
    assert out.stop_reason=='line_search_exhausted' and not out.numerical_failure
    assert out.counts['gradient_steps']==1 and out.counts['trial_attempts']==3
    assert all(t['armijo_valid'] and not t['strict_descent'] for t in out.trace[0]['trials'])
    assert torch.equal(out.vertices,base)


@pytest.mark.parametrize('bad_part',['data','strain'])
def test_nonfinite_trial_is_immediate_numerical_failure_with_last_legal_map(bad_part):
    base=identity(5)
    def callback(y):
        data=-y[:,1:-1,1:-1,0].sum();strain=p1_arap_energy(y,validate=False)
        if not torch.is_grad_enabled():
            if bad_part=='data':data=data*float('nan')
            else:strain=strain+float('inf')
        return data,{'strain':strain}
    out=optimize_distortion_budget_fiber(base,callback,coefficient_side=5,direction=(1.,0.),
        distortion_budget=1.,physical_rms_step=.004,maximum_gradients=4)
    assert out.numerical_failure and out.counts['numerical_failures']==1
    assert out.stop_reason=='nonfinite_trial_objective_or_distortion'
    assert out.counts['trial_attempts']==out.counts['trial_evaluations']==1
    assert out.counts['backtracks']==out.counts['rejected_trials']==0
    assert torch.equal(out.vertices,base)


def test_actual_budget_rejects_an_injected_inaccurate_surrogate_proposal(monkeypatch):
    import qcopt.neural_bijection.dense.coordinated_distortion_budget as module
    original=module.solve_budget_direction
    def wrong_slack(g,r,metric,**kw):
        kw['remaining_budget']=1. # Fault injection: actual R must still reject.
        return original(g,r,metric,**kw)
    monkeypatch.setattr(module,'solve_budget_direction',wrong_slack)
    base=identity(5)
    callback=lambda y:(-y[:,1:-1,1:-1,0].sum(),{'strain':p1_arap_energy(y,validate=False)})
    out=module.optimize_distortion_budget_fiber(base,callback,coefficient_side=5,direction=(1.,0.),
        distortion_budget=0.,physical_rms_step=.004,maximum_gradients=2,max_backtracks=2)
    assert not out.numerical_failure and out.stop_reason=='line_search_exhausted'
    assert out.counts['rejected_trials']==3 and out.counts['accepted_steps']==0
    assert all(t['geometry_valid'] and t['armijo_valid'] and not t['budget_valid'] for t in out.trace[0]['trials'])
    assert torch.equal(out.vertices,base) and out.final_distortion==0


def test_current_over_budget_is_failure_not_a_relaxed_start():
    base=legal_nonidentity()
    callback=lambda y:(y.square().mean(),{'strain':p1_arap_energy(y,validate=False)})
    out=optimize_distortion_budget_fiber(base,callback,coefficient_side=5,direction=(1.,0.),
        distortion_budget=float(p1_arap_energy(base))*.5,physical_rms_step=.004)
    assert out.numerical_failure and out.stop_reason=='current_distortion_outside_budget'
    assert out.counts['gradient_steps']==out.counts['data_vjps']==out.counts['trial_attempts']==0
    assert torch.equal(out.vertices,base)


def test_original_price_pareto_point_may_legally_retain_incumbent():
    base=legal_nonidentity();budget=float(p1_arap_energy(base))
    callback=lambda y:(-3*p1_arap_energy(y,validate=False),{'strain':p1_arap_energy(y,validate=False)})
    out=optimize_distortion_budget_fiber(base,callback,coefficient_side=5,direction=(1.,0.),
        distortion_budget=budget,physical_rms_step=.004,maximum_gradients=2)
    assert not out.numerical_failure
    assert out.stop_reason in ('zero_computed_direction','finite_nondescent_direction','line_search_exhausted')
    assert torch.equal(out.vertices,base) and out.counts['gradient_steps']==1


def test_separate_data_and_distortion_vjps_match_literal_coefficient_derivatives(monkeypatch):
    import qcopt.neural_bijection.dense.coordinated_distortion_budget as module
    base=legal_nonidentity();metric=DistortionBudgetMetric(9,5)
    callback=lambda y:((y[...,0]-.23).square().mean(),{'strain':p1_arap_energy(y,validate=False)})
    c=torch.zeros(1,3,3,dtype=torch.float64,requires_grad=True)
    data,parts=callback(base+metric.prolong(c)[...,None]*base.new_tensor([1.,0.]))
    expected_g=torch.autograd.grad(data,c,retain_graph=True)[0]
    expected_r=torch.autograd.grad(parts['strain'],c)[0]
    original=module.solve_budget_direction;seen=[]
    def check(g,r,*args,**kwargs):
        torch.testing.assert_close(g,expected_g,rtol=0,atol=0)
        torch.testing.assert_close(r,expected_r,rtol=0,atol=0)
        seen.append(True);return original(g,r,*args,**kwargs)
    monkeypatch.setattr(module,'solve_budget_direction',check)
    result=module.optimize_distortion_budget_fiber(base,callback,coefficient_side=5,direction=(1.,0.),
        distortion_budget=float(p1_arap_energy(base)),physical_rms_step=.004,maximum_gradients=1)
    assert seen==[True] and result.counts['data_vjps']==result.counts['distortion_vjps']==1
