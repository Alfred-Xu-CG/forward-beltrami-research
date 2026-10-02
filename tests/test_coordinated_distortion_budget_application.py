import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools import coordinated_distortion_budget_application as app
from tools import coordinated_real_case as original
from tests.test_coordinated_shared_affine_evidence import configuration


def test_direct_data_assembly_does_not_subtract_large_original_regularizer():
    class Evidence:
        image_weight=1.;oob_weight=1.;shape_weight=1e-4;match_weight=.2
        def __call__(self,y):
            parts=dict(image=y.square().sum(),oob=(y-1).square().sum(),
                       shape=y.pow(4).sum(),match=(y+2).square().sum(),strain=1e30+y.sum())
            return 3*parts['strain'],parts
    y=torch.tensor([.3,.7],dtype=torch.float64,requires_grad=True)
    value,parts=app.BudgetEvidence(Evidence())(y)
    expected=parts['image']+parts['oob']+1e-4*parts['shape']+.2*parts['match']
    assert torch.equal(value,expected) and value.item()>0
    assert torch.equal(torch.autograd.grad(value,y,retain_graph=True)[0],torch.autograd.grad(expected,y)[0])
    assert float(parts['original_total']-3*parts['strain'])==0  # cancellation path would be wrong


@pytest.mark.parametrize('failure',[False,True])
def test_actual_incoming_cap_legal_early_stops_and_numerical_failure(tmp_path,monkeypatch,failure):
    args=configuration(tmp_path,'analytic');args.mind_frame='shared_affine'
    original.optimize(args)
    incoming=args.output
    with np.load(incoming) as saved:original_vertices=saved['vertices'].copy()
    args.output=tmp_path/('budget_fail.npz' if failure else 'budget.npz')
    monkeypatch.setattr(app,'_validate_configuration',lambda cfg:None)
    observations=[]
    def solver(anchor,objective,**kw):
        value,parts=objective(anchor)
        observations.append((anchor.clone(),kw['distortion_budget'],float(parts['strain'])))
        counts=dict(gradient_steps=1,data_vjps=1,distortion_vjps=1,objective_evaluations=1,
            trial_attempts=0,trial_evaluations=0,accepted_steps=0,backtracks=0,rejected_trials=0,
            dual_evaluations=1,dual_bracket_probes=0,dual_bisections=0,numerical_failures=int(failure))
        return SimpleNamespace(vertices=anchor,initial_objective=float(value),final_objective=float(value),
            initial_distortion=float(parts['strain']),final_distortion=float(parts['strain']),counts=counts,
            trace=[],stop_reason='dual_bracket_failure' if failure else 'finite_non_descent',
            minimum_contracted_slack=.01,numerical_failure=failure)
    result=app.optimize_distortion_budget(args,initial_map=incoming,fiber_solver=solver)
    count=1 if failure else 2*len(args.levels)
    assert len(observations)==count and result['gradient_steps']==count
    assert result['numerical_failure']==failure and result['schedule_complete']!=failure
    assert len({record[1] for record in observations})==1
    assert result['distortion_budget']==result['initial']['strain']==result['final']['strain']
    assert result['selected_stage'] is None and result['saved_binary_certificate']['valid']
    assert result['final']['total']==result['initial']['total']
    assert 'per original area-reduced raw raster scale' in result['image_preprocessing']['moving_descriptor_prewarp_scope']
    assert result['objective_evaluations']==count+1+count+1
    with np.load(args.output) as saved:np.testing.assert_array_equal(saved['vertices'],original_vertices)
