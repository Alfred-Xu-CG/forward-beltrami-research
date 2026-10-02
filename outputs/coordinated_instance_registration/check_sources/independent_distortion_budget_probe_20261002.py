"""Independent dense-matrix, literal-triangle, and constrained-solve oracles."""
import json
import math
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from qcopt.neural_bijection.dense.coordinated_stiffness import DirichletGalerkinStiffness
from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
torch.set_num_threads(2)


def matrices(fine, coarse):
    """Full-node prolongation from scalar tensor-product hats; explicit stencil."""
    knots = np.arange(1, coarse-1)/(coarse-1)
    nodes = np.arange(fine)/(fine-1)
    H = np.maximum(1-np.abs(nodes[:, None]-knots)*(coarse-1), 0)
    P = np.kron(H, H)
    Kf = np.zeros(((fine-2)**2, (fine-2)**2))
    for y in range(fine-2):
        for x in range(fine-2):
            a = y*(fine-2)+x
            Kf[a,a] = 4
            for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                if 0 <= x+dx < fine-2 and 0 <= y+dy < fine-2:
                    Kf[a,(y+dy)*(fine-2)+x+dx] = -1
    interior = np.arange(fine*fine).reshape(fine,fine)[1:-1,1:-1].ravel()
    K = P[interior].T@Kf@P[interior]
    M = P.T@P/fine**2
    return P,K,M


def basis_eigenvalues(fine, coarse):
    n=coarse-1; t=(fine-1)//n
    i=np.arange(1,n)
    U=np.sqrt(2/n)*np.sin(np.pi*i[:,None]*i[None,:]/n)
    V=np.kron(U,U)
    a=(2-2*np.cos(np.pi*i/n))/t
    b=((2*t*t+1)+(t*t-1)*np.cos(np.pi*i/n))/(3*t)
    return V,(a[:,None]*b[None,:]+b[:,None]*a[None,:]).ravel(),(b[:,None]*b[None,:]/fine**2).ravel()


def triangle_jacobians(Y):
    """Literal source triangle inverse, not production edge-vector formulas."""
    fine=Y.shape[0]
    X=np.stack(np.meshgrid(np.arange(fine)/(fine-1),np.arange(fine)/(fine-1),indexing='xy'),-1)
    result=[]
    for row in range(fine-1):
        for col in range(fine-1):
            for triple in (((row,col),(row,col+1),(row+1,col+1)),
                           ((row,col),(row+1,col+1),(row+1,col))):
                a,b,c=triple
                inverse=torch.tensor(np.linalg.inv(np.stack((X[b]-X[a],X[c]-X[a]),1)),dtype=Y.dtype)
                result.append(torch.stack((Y[b]-Y[a],Y[c]-Y[a]),1)@inverse)
    return torch.stack(result)


def closest_rotations(J):
    U,_,V=np.linalg.svd(J)
    sign=np.linalg.det(U@V)
    correction=np.tile(np.eye(2),(len(J),1,1))
    correction[:,1,1]=sign
    return U@correction@V


def matrix_majorizer_checks():
    rng=np.random.default_rng(872)
    rows=[]
    for fine,coarse in ((5,3),(7,4),(9,3),(9,5),(5,5)):
        P,K,M=matrices(fine,coarse)
        V,k,m=basis_eigenvalues(fine,coarse)
        ek=float(abs(K-(V*k)@V.T).max());em=float(abs(M-(V*m)@V.T).max())
        assert ek<2e-14 and em<2e-16
        op=DirichletGalerkinStiffness(fine,coarse,weight=1.)
        c=rng.normal(size=len(k))
        a=torch.tensor(c.reshape(1,coarse-2,coarse-2),dtype=torch.float64)
        assert abs(op.prolong(a).numpy().ravel()-P@c).max()<2e-15
        assert abs(op.apply(a).numpy().ravel()-K@c).max()<2e-14
        yy,xx=np.meshgrid(np.arange(fine)/(fine-1),np.arange(fine)/(fine-1),indexing='ij')
        base=np.stack((xx,yy),-1)
        base[1:-1,1:-1]+=rng.normal(scale=.009/(fine-1),size=(fine-2,fine-2,2))
        e=np.array([.6,.8])
        base=torch.tensor(base,dtype=torch.float64)
        Pt=torch.tensor(P,dtype=torch.float64)
        et=torch.tensor(e,dtype=torch.float64)
        current=torch.zeros(len(k),dtype=torch.float64,requires_grad=True)
        mapped=lambda z:base+(Pt@z).reshape(fine,fine,1)*et
        J0=triangle_jacobians(base)
        assert np.min(np.linalg.det(J0.numpy()))>0
        rotations=torch.tensor(closest_rotations(J0.numpy()),dtype=torch.float64)
        frozen=lambda z:.5*(triangle_jacobians(mapped(z))-rotations).square().sum((1,2)).mean()
        R=lambda z:p1_arap_energy(mapped(z)[None],validate=True)
        value=R(current);rf=float(value)
        r=torch.autograd.grad(value,current)[0].numpy()
        rfreeze=torch.autograd.grad(frozen(current),current)[0].numpy()
        hessian=torch.autograd.functional.hessian(frozen,current).numpy()
        eH=float(abs(hessian-K).max());er=float(abs(r-rfreeze).max())
        assert eH<1e-13 and er<2e-15 and abs(float(frozen(current))-rf)<1e-15
        gaps=[];quadratic_error=[]
        for _ in range(8):
            d=rng.normal(scale=.001/(fine-1),size=len(k))
            td=torch.tensor(d,dtype=torch.float64)
            upper=rf+r@d+.5*d@K@d
            actual=float(R(td));quad=float(frozen(td))
            assert actual<=upper+1e-15
            assert abs(quad-upper)<1e-15
            gaps.append(upper-actual);quadratic_error.append(abs(quad-upper))
        rows.append(dict(fine=fine,coarse=coarse,K_error=ek,M_error=em,
                         literal_frozen_hessian_error=eH,gradient_touch_error=er,
                         maximum_quadratic_error=max(quadratic_error),minimum_majorizer_gap=min(gaps)))
    return rows


def dense_dual(g,r,K,M,h,ell):
    tau=ell/math.sqrt(g@np.linalg.solve(M,g))
    A=M/tau
    def evaluate(mu):
        d=np.linalg.solve(A+mu*K,-g-mu*r)
        return d, r@d+.5*d@K@d-h
    d,psi=evaluate(0)
    if psi<=0:return d,0.,tau,psi
    if h==0 and not np.any(r):return np.zeros_like(g),None,tau,0.
    lo=0.;hi=3.
    for _ in range(32):
        d,psi=evaluate(hi)
        if psi<=0:break
        lo=hi;hi*=2
    else:raise ArithmeticError('bounded bracket exhausted')
    for _ in range(32):
        mid=(lo+hi)/2
        if mid==lo or mid==hi:break
        _,psi=evaluate(mid)
        if psi<=0:hi=mid
        else:lo=mid
    d,psi=evaluate(hi)
    return d,hi,tau,psi


def constrained_checks():
    _,K,M=matrices(7,4);rng=np.random.default_rng(637)
    rows=[]
    for name,h,kind in [('inactive',20.,'normal'),('active',0.,'normal'),
                        ('positive_slack',.00001,'normal'),('zero_r',.00001,'zero'),
                        ('near_pareto',0.,'pareto'),('singleton',0.,'zero')]:
        r=rng.normal(scale=.02,size=4);g=rng.normal(scale=.3,size=4)
        if kind=='zero':r[:]=0
        if kind=='pareto':g=-3*r+1e-4*g
        d,mu,tau,psi=dense_dual(g,r,K,M,h,.004)
        assert psi<=0
        if name=='singleton':
            assert np.array_equal(d,np.zeros_like(d));rows.append(dict(name=name,zero=True));continue
        model=lambda z:g@z+.5*z@M@z/tau
        grad=lambda z:g+M@z/tau
        solution=minimize(model,np.zeros(4),jac=grad,method='SLSQP',constraints=[
            dict(type='ineq',fun=lambda z:h-r@z-.5*z@K@z,jac=lambda z:-r-K@z)],
            options=dict(ftol=1e-14,maxiter=1000))
        assert solution.success,solution.message
        error=float(abs(model(solution.x)-model(d)))
        assert error<2e-11
        assert g@d<0
        predicted=r@d+.5*d@K@d
        alpha=.99
        exact_slack=alpha*h-.5*alpha*(1-alpha)*d@K@d
        assert r@(alpha*d)+.5*(alpha*d)@K@(alpha*d)<=exact_slack+1e-17
        rows.append(dict(name=name,mu=mu,psi=psi,directional_derivative=float(g@d),
                         independent_SLSQP_model_error=error,rms=float(np.sqrt(d@M@d)),
                         contracted_majorizer_slack=float(h-(r@(alpha*d)+.5*(alpha*d)@K@(alpha*d)))))
    return rows


def wrapper_checks():
    from tools import coordinated_distortion_budget_application as app
    from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
    source=Path(__file__).parent
    sys.path.insert(0,str(source))
    from independent_data_metric_probe_20261002 import fixture
    Y,evidence=fixture();Y.requires_grad_(True)
    old,parts=evidence(Y);D,budgetparts=app.BudgetEvidence(evidence)(Y)
    literal=parts['image']+parts['oob']+1e-4*parts['shape']+.2*parts['match']
    g1=torch.autograd.grad(D,Y,retain_graph=True)[0]
    g2=torch.autograd.grad(literal,Y,retain_graph=True)[0]
    assert torch.equal(D,literal) and torch.equal(g1,g2)
    assert budgetparts['strain'].requires_grad
    total_error=abs(float(D+3*budgetparts['strain']-old))
    assert total_error<1e-15

    yy,xx=np.meshgrid(np.arange(9)/8,np.arange(9)/8,indexing='ij')
    identity=torch.tensor(np.stack((xx,yy),-1)[None],dtype=torch.float64)
    incoming=identity.clone();incoming[:,1:-1,1:-1,0]+=.008
    midpoint=identity+.5*(incoming-identity)
    class FakeEvidence:
        image_weight=oob_weight=1.;shape_weight=1e-4;match_weight=.2
        mind_frame_metadata={'synthetic_fixture':True}
        def __call__(self,Y):
            data=(Y-identity).square().mean()
            R=p1_arap_energy(Y,validate=False)
            zero=Y.sum()*0
            return data+3*R,dict(image=data,oob=zero,shape=zero,match=zero,strain=R)
    fake=FakeEvidence()
    budget=float(p1_arap_energy(incoming));seen=[]
    def solver(anchor,objective,**kwargs):
        assert kwargs['distortion_budget']==budget
        seen.append(kwargs)
        index=len(seen)-1
        target=midpoint if index<2 else incoming
        d0,p0=objective(anchor);d1,p1=objective(target)
        floor=.001+.05*(q1_corner_determinants(anchor)*64-.001)
        slack=float((q1_corner_determinants(target)*64-floor).min())
        assert slack>0 and float(p1['strain'])<=budget
        return SimpleNamespace(vertices=target,initial_objective=float(d0),final_objective=float(d1),
            initial_distortion=float(p0['strain']),final_distortion=float(p1['strain']),
            counts=dict(gradient_steps=1,data_vjps=1,distortion_vjps=1,objective_evaluations=2,
                        accepted_steps=int(index in (0,2)),trial_evaluations=1),
            trace=[],stop_reason='zero_data_gradient' if index%2 else 'gradient_budget',
            minimum_contracted_slack=slack,numerical_failure=False)
    with tempfile.TemporaryDirectory(prefix='independent_budget_wrapper_') as td:
        td=Path(td);affine=td/'affine.npz';incumbent=td/'incumbent.npz'
        np.savez(affine,post_affine_matrix=np.eye(2,dtype=np.float32),post_affine_offset=np.zeros(2,dtype=np.float32))
        np.savez(incumbent,vertices=incoming.numpy(),boundary_reference=identity.numpy(),
            post_affine_matrix=np.eye(2,dtype=np.float32),post_affine_offset=np.zeros(2,dtype=np.float32),interpolation=np.asarray('p1_ac'))
        args=SimpleNamespace(output=td/'output.npz',image_side=9,image_levels=[7,9],levels=[3,5],
            inner_steps=30,device='cpu',threads=2,affine=affine,grid_side=9,minimum_jacobian=.001,learning_rate=.004)
        build=lambda *args:({7:fake,9:fake},{'synthetic_fixture':True},None)
        with patch.object(app,'_validate_configuration',lambda args:None),patch.object(app,'build_pyramid',build):
            report=app.optimize_distortion_budget(args,initial_map=incumbent,fiber_solver=solver)
        assert len(seen)==4 and report['schedule_complete'] and not report['numerical_failure']
        assert report['gradient_steps']==4 and report['maximum_gradient_steps']==120
        assert [row['physical_rms_step'] for row in report['stages']]==[.004,.004,.002,.002]
        assert report['selected_stage']==0 and report['distortion_budget']==budget
        assert report['wrapper_objective_evaluations']==6 and report['objective_evaluations']==14
        with np.load(args.output) as saved:assert np.array_equal(saved['vertices'],midpoint.numpy())
        # Declared numerical failure halts the schedule and remains failed even
        # though the retained exported artifact is geometrically valid.
        seen.clear();args.output=td/'failed.npz'
        def failing(*args,**kwargs):
            result=solver(*args,**kwargs);result.numerical_failure=True;result.stop_reason='nonfinite_trial';return result
        with patch.object(app,'_validate_configuration',lambda args:None),patch.object(app,'build_pyramid',build):
            failed=app.optimize_distortion_budget(args,initial_map=incumbent,fiber_solver=failing)
        assert failed['numerical_failure'] and not failed['schedule_complete'] and len(failed['stages'])==1
    return dict(actual_Evidence_direct_D_and_VJP_exact=True,D_plus_3R_vs_original_E_error=total_error,
        synthetic_wrapper_fixture=dict(fixed_budget=True,retained_stop_continues=True,
            actual_count_not_maximum=True,own_D_best_endpoint=True,numerical_failure_stops_and_labels=True))


def production_core_checks():
    from qcopt.neural_bijection.dense import coordinated_distortion_budget as core
    from independent_joint_pose_probe_20261002 import corners
    _,K,M=matrices(7,4);rng=np.random.default_rng(637);metric=core.DistortionBudgetMetric(7,4)
    result=[]
    for h in (0.,.00001,20.):
        for kind in ('normal','zero','pareto'):
            r=rng.normal(scale=.02,size=4);g=rng.normal(scale=.3,size=4)
            if kind=='zero':r[:]=0
            if kind=='pareto':g=-3*r+1e-4*g
            td=lambda x:torch.tensor(x.reshape(1,2,2),dtype=torch.float64)
            expected,mu,tau,psi=dense_dual(g,r,K,M,h,.004)
            d,record=core.solve_budget_direction(td(g),td(r),metric,remaining_budget=h,physical_rms_step=.004)
            assert not record['numerical_failure']
            error=float(abs(d.numpy().ravel()-expected).max())
            assert error<2e-10,(h,kind,error,record)
            assert record['dual_evaluations']==(0 if kind=='zero' and h==0 else 1+record['dual_bracket_probes']+record['dual_bisections'])
            result.append(dict(h=h,kind=kind,dense_direction_error=error,stop=record['stop_reason']))
    yy,xx=np.meshgrid(np.arange(9)/8,np.arange(9)/8,indexing='ij')
    identity=torch.tensor(np.stack((xx,yy),-1)[None],dtype=torch.float64)
    base=identity.clone();base[:,1:-1,1:-1,0]+=.006
    B=float(p1_arap_energy(base));floor=.001+.05*(corners(base.numpy()[0])-.001)
    def callback(y):return (y-identity).square().mean(),dict(strain=p1_arap_energy(y,validate=False))
    settings=dict(coefficient_side=5,direction=(1.,0.),distortion_budget=B,physical_rms_step=.004,maximum_gradients=6)
    normal=core.optimize_distortion_budget_fiber(base,callback,**settings)
    assert not normal.numerical_failure and normal.counts['accepted_steps']>0
    assert normal.final_objective<normal.initial_objective and normal.final_distortion<=B
    assert (corners(normal.vertices.numpy()[0])>floor).all()
    P,_,_=matrices(9,5)
    expected=base.numpy().copy();expected[0,:,:,0]+=(P@normal.coefficients.numpy().ravel()).reshape(9,9)
    assert abs(expected-normal.vertices.numpy()).max()<2e-17
    for step in normal.trace:
        for trial in step['trials']:
            if trial['accepted']:
                assert trial['distortion']<=B and trial['objective']<step['objective']
                assert trial['objective']<=trial['armijo_rhs']
    # A deliberate nonsmooth/adversarial gradient model has no actual descent:
    # finite trial rejection must exhaust and retain, without a failure label.
    def no_descent(y):
        model=(y-identity).square().mean()
        return model-model.detach()+1,dict(strain=p1_arap_energy(y,validate=False))
    exhausted=core.optimize_distortion_budget_fiber(base,no_descent,**settings)
    assert exhausted.stop_reason=='line_search_exhausted' and not exhausted.numerical_failure
    assert exhausted.counts['trial_attempts']==13 and exhausted.counts['rejected_trials']==13
    assert exhausted.counts['backtracks']==12 and exhausted.counts['gradient_steps']==1
    assert torch.equal(exhausted.vertices,base)
    # Nonfinite candidate callback is a failure, NOT 13 ordinary rejections.
    def nonfinite(y):
        data,parts=callback(y)
        return (data if torch.is_grad_enabled() else data*float('nan')),parts
    failed=core.optimize_distortion_budget_fiber(base,nonfinite,**settings)
    assert failed.numerical_failure and failed.stop_reason=='nonfinite_trial_objective_or_distortion'
    assert failed.counts['trial_attempts']==1 and failed.counts['rejected_trials']==0
    assert failed.counts['backtracks']==0 and failed.counts['numerical_failures']==1
    assert torch.equal(failed.vertices,base)
    # A rounded/callback cap violation is finite and never inflates B.
    def cap_violation(y):
        data,parts=callback(y)
        if not torch.is_grad_enabled():parts['strain']=parts['strain']*0+np.nextafter(B,math.inf)
        return data,parts
    capped=core.optimize_distortion_budget_fiber(base,cap_violation,**settings)
    assert capped.stop_reason=='line_search_exhausted' and not capped.numerical_failure
    assert capped.counts['rejected_trials']==13 and torch.equal(capped.vertices,base)
    assert all(t['budget_valid'] is False for t in capped.trace[0]['trials'])
    zero=core.optimize_distortion_budget_fiber(identity,lambda y:(-y.sum(),dict(strain=p1_arap_energy(y))),
        **dict(settings,distortion_budget=0.))
    assert zero.stop_reason=='zero_ellipsoid' and zero.counts['gradient_steps']==1 and not zero.numerical_failure
    return dict(dense_spectral_direction_comparisons=result,actual_map_valid=True,
        normal_counts=normal.counts,finite_exhaustion_counts=exhausted.counts,
        nonfinite_trial_counts=failed.counts,one_ulp_budget_excess_rejected=True,
        zero_ellipsoid_no_fake_budget=True)


if __name__=='__main__':
    print(json.dumps(dict(matrix_majorizer=matrix_majorizer_checks(),qcqp=constrained_checks(),wrapper=wrapper_checks(),
                         core=production_core_checks()),indent=2))
