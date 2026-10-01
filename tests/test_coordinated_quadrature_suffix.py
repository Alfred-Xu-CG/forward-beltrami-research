"""Paired objective ablation keeps the original analytic Adam suffix."""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import coordinated_fiber_suffix as fiber
from test_coordinated_fiber_suffix import tiny_report, unchanged_solver


def module():
    return importlib.import_module("tools.coordinated_quadrature_suffix")


def toy_problem():
    reference=fiber.identity_vertices(9,device="cpu").double()
    incoming=reference.clone(); incoming[:,1:-1,1:-1,0]+=.005
    prior=reference.clone(); prior[:,1:-1,1:-1,1]+=.003
    config=SimpleNamespace(grid_side=9,minimum_jacobian=.001,inner_steps=2,
        learning_rate=.004,lr_calibration="edge",levels=[5,9])
    candidates=dict(initial=reference,stored_e1_best_prefix=prior,incoming_prefix=incoming)
    return reference,incoming,config,candidates


def quadratic(target):
    def objective(vertices):
        loss=(vertices-target).square().mean()
        return loss,dict(image=loss,strain=loss*0,oob=loss*0)
    return objective


def test_original_chart_budget_and_own_objective_prefix_selection():
    runner=module()
    reference,incoming,config,candidates=toy_problem()
    outputs=[]
    for target in (reference,candidates["stored_e1_best_prefix"]):
        objective=quadratic(target)
        best,terminal,report=runner.run_arm(incoming,candidates,objective,objective,reference,config)
        torch.testing.assert_close(best,target,rtol=0,atol=0)
        assert report["counts"]["trial_objective_evaluations"]==6
        assert report["counts"]["gradient_steps"]==4
        assert report["counts"]["prefix_selection_objective_evaluations"]==3
        assert report["counts"]["complete_objective_evaluations"]==17
        assert all(row["physical_lr"]==pytest.approx(.002) for row in report["stages"])
        assert all(row["initial_coefficient_max_abs"]==0 for row in report["stages"])
        assert report["incoming_prefix_bitwise_equal"]
        assert fiber.validate_q1_map(terminal,reference)["valid"]
        outputs.append(report["selected"]["candidate"])
    assert outputs==["initial","stored_e1_best_prefix"]


def test_actual_invalid_trial_rejected_even_if_layer_diagnostic_claims_positive():
    runner=module()
    reference,incoming,config,candidates=toy_problem()
    def bad_factory(anchor,**options):
        def layer(proposal,**kwargs):
            bad=anchor.clone();bad[:,4,4]=bad[:,4,3]
            return SimpleNamespace(vertices=bad,scale=torch.ones(1),gauge=torch.zeros(1),
                normalized_margin_min=torch.ones(1))
        return layer
    objective=quadratic(reference)
    best,terminal,report=runner.run_arm(incoming,candidates,objective,objective,reference,config,
        layer_factory=bad_factory)
    assert report["counts"]["failed_trials"]==2
    assert report["counts"]["gradient_steps"]==0
    assert report["solver_status"]=="failed_trial_valid_best_retained"
    assert fiber.validate_q1_map(best,reference)["valid"]
    torch.testing.assert_close(terminal,incoming,rtol=0,atol=0)


def test_invalid_incoming_prefix_fails_before_optimization():
    runner=module()
    reference,incoming,config,candidates=toy_problem()
    incoming[:,4,4]=incoming[:,4,3]
    with pytest.raises(RuntimeError,match="incoming"):
        runner.run_arm(incoming,candidates,quadratic(reference),quadratic(reference),reference,config)


def test_positive_margin_candidate_with_changed_boundary_is_not_accepted():
    runner=module()
    reference,incoming,config,candidates=toy_problem()
    shift=torch.tensor([.01,0.],dtype=incoming.dtype)
    def bad_factory(anchor,**options):
        def layer(proposal,**kwargs):
            return SimpleNamespace(vertices=anchor+shift+0*proposal[...,None],
                scale=torch.ones(1),gauge=torch.zeros(1),normalized_margin_min=torch.ones(1))
        return layer
    objective=quadratic(incoming+shift)
    with pytest.raises(RuntimeError,match="accepted x/y stage"):
        runner.run_arm(incoming,candidates,objective,objective,reference,config,layer_factory=bad_factory)


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_tiny_paired_replay_literal_shared_prefix_and_cross_objectives(tmp_path,monkeypatch,diagonal):
    runner=module()
    original_args,_=tiny_report(tmp_path,diagonal)
    original=fiber.run(original_args,production=False,solver=unchanged_solver([]))
    def forbidden(*args,**kwargs):pytest.fail("quadrature replay must not rerun prefixes")
    monkeypatch.setattr(fiber,"optimize",forbidden)
    args=argparse.Namespace(reuse_prefix_report=original_args.output,output=tmp_path/"quadrature.json")
    report=runner.run(args,production=False)
    assert report["baseline_executed_this_run"] is False
    assert report["shared_incoming_prefix_bitwise_equal"]
    assert report["candidate_prefix_family"]==["initial","stored_e1_best_prefix","incoming_prefix"]
    assert report["quarter_cache_resident_bytes"]>0
    assert report["quarter_cache_metadata"]["trial"]["moving_descriptor_interpolations_per_call"]==4
    assert report["quarter_cache_metadata"]["selection"]["center_map_queries"]==1
    assert set(report["cross_objectives"])=={"centers","quarters"}
    for name,multiplier in (("centers",1),("quarters",4)):
        arm=report["arms"][name]
        assert arm["counts"]["gradient_steps"]==4
        assert arm["counts"]["trial_objective_evaluations"]==6
        assert arm["saved_binary_certificate"]["valid"]
        assert arm["moving_descriptor_interpolations_per_objective"]==multiplier
        assert arm["cross_objective_evaluations"]==2
        assert arm["moving_descriptor_interpolation_equivalents"]==multiplier*(17+2)
        with np.load(arm["output"],allow_pickle=False) as saved:
            assert saved["vertices"].shape==(1,9,9,2)
        assert set(report["cross_objectives"][name])=={"centers","quarters"}
        assert report["cross_objectives"][name][name]["total"]<=min(
            row["total"] for row in arm["prefix_candidates"])+1e-7
    assert not (tmp_path/"quadrature_baseline.npz").exists()
    # E1 is the original final analytic Adam suffix, not a second optimizer.
    with np.load(original["baseline_output"],allow_pickle=False) as baseline:
        with np.load(report["arms"]["centers"]["output"],allow_pickle=False) as replay:
            np.testing.assert_array_equal(baseline["vertices"],replay["vertices"])
    with np.load(original["prefix_output"],allow_pickle=False) as old:
        with np.load(report["shared_prefix_output"],allow_pickle=False) as new:
            np.testing.assert_array_equal(old["coarse_vertices"],new["coarse_vertices"])


@pytest.mark.parametrize("occupied",["quadrature.json","quadrature_prefix.npz",
    "quadrature_centers.npz","quadrature_centers.json","quadrature_quarters.npz","quadrature_quarters.json"])
def test_new_output_collisions_fail_before_loading_any_inputs(tmp_path,occupied):
    (tmp_path/occupied).write_text("preserve")
    args=argparse.Namespace(reuse_prefix_report=tmp_path/"missing.json",output=tmp_path/"quadrature.json")
    with pytest.raises(FileExistsError):module().run(args)
    assert (tmp_path/occupied).read_text()=="preserve"
