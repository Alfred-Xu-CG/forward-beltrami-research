"""Physical fiber, contracted feasibility, bounded objective-only line search."""
import pytest
import torch
import numpy as np

from qcopt.neural_bijection.dense.coordinated_fiber_lbfgs import (
    solve_coordinated_fiber_lbfgs, _direction, _curvature_pair,
)
from qcopt.neural_bijection.dense.coordinated_update import CoordinatedQ1Update,single_direction_corner_change


def grid(rows,columns=None,dtype=torch.float64):
    columns=rows if columns is None else columns
    yy,xx=torch.meshgrid(torch.arange(rows,dtype=dtype)/(rows-1),
                        torch.arange(columns,dtype=dtype)/(columns-1),indexing="ij")
    return torch.stack((xx,yy),-1)[None]


def independent_corners(vertices):
    v=vertices.detach().double().numpy()
    a,b,c,d=v[:,:-1,:-1],v[:,:-1,1:],v[:,1:,1:],v[:,1:,:-1]
    return np.stack([np.linalg.det(np.stack((q-p,r-p),-1))
                     for p,q,r in ((a,b,d),(a,b,c),(d,b,c),(a,c,d))],-1)


@pytest.mark.parametrize("direction",[(1.,0.),(0.,1.),(.6,.8)])
def test_legal_quadratic_progress_exact_fiber_boundary_and_b2_global_reference(direction):
    reference=grid(5,7)*torch.tensor((3.,2.),dtype=torch.float64)
    anchor=reference.repeat(2,1,1,1)
    anchor[...,0]+=.02*anchor[...,1]
    raw=torch.zeros(anchor.shape[:-1],dtype=anchor.dtype)
    raw[:,1:-1,1:-1]=.015
    e=torch.tensor(direction,dtype=anchor.dtype)
    target=anchor+raw[...,None]*e
    def objective(v):
        loss=.5*(v-target).square().sum()
        return loss,dict(image=loss)
    result=solve_coordinated_fiber_lbfgs(anchor,objective,reference=reference,direction=direction,
                                       initial_displacement_rms=.002,max_gradient_evaluations=8)
    assert result.final_objective < result.initial_objective*1e-12
    assert result.best_objective == result.final_objective
    torch.testing.assert_close(result.vertices,target,rtol=0,atol=1e-12)
    torch.testing.assert_close(result.vertices,anchor+result.displacement[...,None]*e,rtol=0,atol=0)
    for index in (0,-1):
        assert torch.equal(result.vertices[:,index],anchor[:,index])
        assert torch.equal(result.vertices[:,:,index],anchor[:,:,index])
    assert not result.vertices.requires_grad and not result.displacement.requires_grad
    assert result.calibration["first_descent_interior_rms"]==pytest.approx(.002)
    q0,qa,qr=map(independent_corners,(anchor,result.vertices,reference))
    s0=q0/qr-.001
    assert np.min(.95*s0+(qa-q0)/qr)>0
    assert result.counts["gradient_evaluations"]<=8
    accepted=[event for event in result.trace if event["kind"]=="trial" and event["accepted"]]
    assert all(event["directional_derivative"]<0 for event in accepted)


def test_affine_corner_fiber_identity_and_analytic_stage_reachability():
    anchor=grid(5,7)
    anchor[...,0]+=.07*anchor[...,1]
    reference=grid(5,7)
    raw=torch.randn(1,5,7,dtype=anchor.dtype,generator=torch.Generator().manual_seed(14))*.4
    target=CoordinatedQ1Update((.6,.8),mode="analytic")(anchor,raw,reference=reference).vertices
    e=torch.tensor((.6,.8),dtype=anchor.dtype)
    u=((target-anchor)*e).sum(-1)
    q0,qt,qr=map(independent_corners,(anchor,target,reference))
    change=single_direction_corner_change(anchor,u,e).numpy()
    np.testing.assert_allclose(qt,q0+change,rtol=2e-14,atol=2e-17)
    np.testing.assert_allclose((qt-q0)/qr,change/qr,rtol=2e-14,atol=2e-15)
    assert np.min(.95*(q0/qr-.001)+change/qr)>-3e-15  # Closed analytic set.
    # Solver uses the strict interior of that same set, not the larger eta set.
    def objective(v):return (v-target).square().sum(),{}
    result=solve_coordinated_fiber_lbfgs(anchor,objective,reference=reference,direction=(.6,.8),
                                       initial_displacement_rms=.005,max_gradient_evaluations=10)
    for event in result.trace:
        if event["accepted"]:
            assert event["actual_contracted_margin_min"]>0 and event["actual_margin_min"]>0
    assert result.final_objective<result.initial_objective


def test_exact_gradient_budget_and_curvature_skips_on_linear_objective():
    anchor=grid(5)
    calls=[]
    def objective(v):
        calls.append(1)
        return v[:,1:-1,1:-1,0].sum(),{}
    result=solve_coordinated_fiber_lbfgs(anchor,objective,initial_displacement_rms=1e-5)
    assert result.counts["gradient_evaluations"]==30
    assert result.counts["objective_evaluations"]==len(calls)==30
    assert result.counts["accepted_steps"]==29
    assert result.counts["curvature_skips"]==29
    assert result.stop_reason=="gradient_budget"
    assert result.counts["rounded_geometry_checks"]==30
    assert result.counts["affine_direction_evaluations"]==29
    assert result.counts["rejected_objective_evaluations"]==0


def test_nan_trial_objective_all_forwards_counted_no_gradient_on_rejections():
    anchor=grid(5)
    calls=[]
    def objective(v):
        calls.append(1)
        loss=v[...,0].square().sum()
        if not torch.equal(v.detach(),anchor):loss=loss*float("nan")
        return loss,{}
    result=solve_coordinated_fiber_lbfgs(anchor,objective,initial_displacement_rms=.0001,max_backtracks=6)
    assert result.stop_reason=="objective_backtrack_limit"
    assert result.counts["objective_evaluations"]==len(calls)==8
    assert result.counts["gradient_evaluations"]==1
    assert result.counts["rejected_objective_evaluations"]==7
    assert result.counts["armijo_halvings"]==6
    assert result.counts["accepted_steps"]==0
    assert torch.equal(result.vertices,anchor) and torch.equal(result.best_vertices,anchor)
    assert all(event["rejection"]=="nonfinite_objective" for event in result.trace[1:])


def test_finite_armijo_rejection_initial_nan_and_gradient_nan():
    anchor=grid(5)
    def mismatch(v):
        linear=-v[:,1:-1,1:-1,0].sum()
        penalty=0 if torch.equal(v.detach(),anchor) else 1e3
        return linear+penalty,{}
    result=solve_coordinated_fiber_lbfgs(anchor,mismatch,initial_displacement_rms=.0001,max_backtracks=2)
    assert result.counts["rejected_objective_evaluations"]==3
    assert all(event["rejection"]=="armijo" for event in result.trace[1:])
    def nan(v):return v.sum()*float("nan"),{}
    initial=solve_coordinated_fiber_lbfgs(anchor,nan,initial_displacement_rms=.001)
    assert initial.stop_reason=="nonfinite_initial_objective"
    assert initial.counts["objective_evaluations"]==1 and initial.counts["gradient_evaluations"]==0
    def bad_gradient(v):return (v-anchor).square().sqrt().sum(),{}
    gradient=solve_coordinated_fiber_lbfgs(anchor,bad_gradient,initial_displacement_rms=.001)
    assert gradient.stop_reason=="nonfinite_initial_gradient"
    assert gradient.counts["gradient_evaluations"]==1


def test_standard_two_loop_positive_pair_and_explicit_bad_curvature_fallback():
    gradient=torch.tensor([1.,-2.],dtype=torch.float64)
    s=torch.tensor([.1,.2],dtype=torch.float64)
    y=2*s
    pair,gamma=_curvature_pair(s,y,1e-10)
    assert gamma==pytest.approx(.5)
    direction,fallback=_direction(gradient,[pair],gamma)
    torch.testing.assert_close(direction,-.5*gradient)
    assert not fallback
    for bad_y in (-s,torch.zeros_like(s)):
        assert _curvature_pair(s,bad_y,1e-10)==(None,None)
    # Finite positive dot products do NOT justify retaining overflowed rho/gamma.
    for tiny_s,tiny_y in ((s*1e-160,y*1e-160),(s*1e150,y*1e-160)):
        assert _curvature_pair(tiny_s,tiny_y,1e-10)==(None,None)
    direction,fallback=_direction(gradient,[],-1.)
    assert fallback and torch.equal(direction,-gradient)


def test_float32_actual_contracted_floor_reject_stops_without_geometry_halving():
    anchor=grid(3,dtype=torch.float32)
    anchor[0,1,1,0]=torch.nextafter(torch.tensor(.25),torch.tensor(1.))
    calls=[]
    def objective(v):
        calls.append(1)
        return v[0,1,1,0],{}
    result=solve_coordinated_fiber_lbfgs(anchor,objective,initial_displacement_rms=1.,minimum_jacobian=.5)
    assert result.stop_reason=="rounded_geometry_rejected"
    assert result.counts["geometry_rejections"]==1
    assert result.counts["objective_evaluations"]==len(calls)==1
    assert result.counts["gradient_evaluations"]==1
    assert result.counts["armijo_halvings"]==0
    assert torch.equal(result.vertices,anchor)
    assert result.trace[-1]["actual_contracted_margin_min"]<=0


@pytest.mark.parametrize("case",["anchor_grad","reference_grad","direction_grad","dtype","size","reference_shape",
                                "initial_rms_tensor","initial_rms_zero","theta","fraction","budget_bool","history","backtracks",
                                "negative_eta","negative_gradient_tolerance","negative_curvature_tolerance"])
def test_constant_shape_precision_configuration_guards(case):
    anchor=grid(5);kwargs=dict(initial_displacement_rms=.001)
    if case=="anchor_grad":anchor.requires_grad_()
    elif case=="reference_grad":kwargs["reference"]=anchor.clone().requires_grad_()
    elif case=="direction_grad":kwargs["direction"]=torch.tensor((1.,0.),requires_grad=True)
    elif case=="dtype":anchor=anchor.half()
    elif case=="size":anchor=grid(2)
    elif case=="reference_shape":kwargs["reference"]=grid(4)
    elif case=="initial_rms_tensor":kwargs["initial_displacement_rms"]=torch.tensor(.001,requires_grad=True)
    elif case=="initial_rms_zero":kwargs["initial_displacement_rms"]=0
    elif case=="theta":kwargs["theta"]=1.
    elif case=="fraction":kwargs["fraction_to_boundary"]=1.
    elif case=="budget_bool":kwargs["max_gradient_evaluations"]=True
    elif case=="history":kwargs["history_size"]=0
    elif case=="backtracks":kwargs["max_backtracks"]=-1
    elif case=="negative_eta":kwargs["minimum_jacobian"]=-1e-310
    elif case=="negative_gradient_tolerance":kwargs["gradient_tolerance"]=-1e-310
    elif case=="negative_curvature_tolerance":kwargs["curvature_tolerance"]=-1e-310
    with pytest.raises(ValueError):solve_coordinated_fiber_lbfgs(anchor,lambda v:(v.sum(),{}),**kwargs)


def test_bad_callback_contract_invalid_geometry_and_initial_budget_one():
    anchor=grid(5)
    for objective in (lambda v:v.sum(),lambda v:(1.,{}),lambda v:(v.sum(),[]),
                      lambda v:(v.sum(),{"bad":v}),lambda v:(v.sum().detach(),{})):
        with pytest.raises(ValueError):solve_coordinated_fiber_lbfgs(anchor,objective,initial_displacement_rms=.001)
    bad=anchor.clone();bad[0,1,1]=-1
    with pytest.raises(ValueError,match="all-four"):
        solve_coordinated_fiber_lbfgs(bad,lambda v:(v.sum(),{}),initial_displacement_rms=.001)
    result=solve_coordinated_fiber_lbfgs(anchor,lambda v:(v.sum(),{}),initial_displacement_rms=.001,max_gradient_evaluations=1)
    assert result.counts["objective_evaluations"]==result.counts["gradient_evaluations"]==1
    assert result.counts["accepted_steps"]==0


def test_minimum_step_and_exact_zero_gradient_stop_without_extra_forward():
    anchor=grid(5)
    def objective(v):return v.sum(),{}
    result=solve_coordinated_fiber_lbfgs(anchor,objective,initial_displacement_rms=.001,minimum_step=2.)
    assert result.stop_reason=="minimum_step"
    assert result.counts["objective_evaluations"]==result.counts["gradient_evaluations"]==1
    def zero(v):return (v*0).sum(),{}
    stationary=solve_coordinated_fiber_lbfgs(anchor,zero,initial_displacement_rms=.001)
    assert stationary.stop_reason=="gradient_tolerance"
    assert stationary.counts["objective_evaluations"]==stationary.counts["gradient_evaluations"]==1


def test_tiny_nonzero_gradient_calibration_is_not_square_underflow_zero():
    anchor=grid(5)
    def tiny(v):return 1e-200*v[:,1:-1,1:-1,0].sum(),{}
    result=solve_coordinated_fiber_lbfgs(anchor,tiny,initial_displacement_rms=.001,max_gradient_evaluations=2)
    assert result.calibration["initial_gradient_interior_rms"]==pytest.approx(1e-200,abs=0)
    assert result.calibration["initial_calibration_valid"]
    assert result.calibration["first_descent_interior_rms"]==pytest.approx(.001)
    assert result.counts["accepted_steps"]==1
    def extreme(v):return 1e-320*v[:,1:-1,1:-1,0].sum(),{}
    stopped=solve_coordinated_fiber_lbfgs(anchor,extreme,initial_displacement_rms=.001)
    assert stopped.stop_reason=="nonfinite_initial_calibration"
    assert not stopped.calibration["initial_calibration_valid"]
    assert stopped.counts["objective_evaluations"]==stopped.counts["gradient_evaluations"]==1


def test_finite_extreme_direction_normalizes_robustly_not_to_zero():
    anchor=grid(5)
    def objective(v):return v[:,1:-1,1:-1].sum(),{}
    common=dict(initial_displacement_rms=.001,max_gradient_evaluations=2)
    ordinary=solve_coordinated_fiber_lbfgs(anchor,objective,direction=(1.,1.),**common)
    extreme=solve_coordinated_fiber_lbfgs(anchor,objective,direction=(1e308,1e308),**common)
    torch.testing.assert_close(ordinary.vertices,extreme.vertices,rtol=0,atol=0)
    assert extreme.counts["accepted_steps"]==1
    assert extreme.final_objective<extreme.initial_objective
