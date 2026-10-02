"""Application capture-only CPU fixtures, not actual Inductor speed evidence."""
import argparse
import json

import pytest

from tools import coordinated_compiled_application_compare as comparison
from tools import coordinated_compiled_priors as priors
from test_coordinated_application_profile import fixture_report


def args_for(tmp_path):
    return argparse.Namespace(predictions=fixture_report(tmp_path),output=tmp_path/"comparison.json")


def capture_factory(monkeypatch):
    original=priors.make_joint_priors
    def capture(backend=None):
        assert backend=="inductor"
        return original("eager")
    monkeypatch.setattr(priors,"make_joint_priors",capture)
    return capture


def test_actual_14_tiny_applications_cold_then_abba_capture_only(tmp_path,monkeypatch):
    args=args_for(tmp_path)
    factory=capture_factory(monkeypatch)
    report=comparison.run(args,production=False,test_backend_label="TEST MOCK: eager-backend graph capture only, NOT Inductor performance evidence")
    assert report["status"]=="complete"
    assert report['pair_name']=='he_to_cc10'
    assert report["speed_evidence_eligible"] is False
    assert "TEST MOCK" in report["evidence_scope"]
    assert priors.make_joint_priors is factory
    runs=report["runs"]
    assert len(runs)==14 and len(report["comparisons"])==91
    assert [row["backend"] for row in runs]==["eager","inductor"]+["eager","inductor","inductor","eager"]*3
    assert len({row["output"] for row in runs})==14
    assert all(row["gradient_steps"]==4 and row["evaluations"]==8 and row["objective_evaluations"]==18 for row in runs)
    assert all(row["saved_binary_certificate"]["valid"] for row in runs)
    assert all(row["map_bitwise_equal"] and row["counters_equal"] for row in report["comparisons"])
    assert all(len(row["factory_calls"])==(row["backend"]=="inductor") for row in runs)
    assert all(row["compiler_factory_seconds"]<=row["complete_call_seconds"] for row in runs)
    assert len(report["warm_summary"]["paired_ABBA_groups"])==3
    assert report["numerical_summary"]["warm_same_backend_repeat_variation"]["pair_count"]==30
    assert report["numerical_summary"]["warm_cross_backend_differences"]["pair_count"]==36
    assert report["numerical_equivalence_claimed"] is False
    assert json.loads(args.output.read_text())["annotations_read"] is False


def test_actual_14_point_sampling_applications_hold_compiled_priors_fixed(tmp_path,monkeypatch):
    from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
    args=args_for(tmp_path);args.comparison_kind="frozen_points";args.pair_name='he_to_ki67'
    factory=capture_factory(monkeypatch)
    original=ImageCorrespondences.prepare_fixed_p1_sampling
    preparations=[]
    def counted(self,rows,columns,diagonal="ac"):
        preparations.append((rows,columns,diagonal))
        return original(self,rows,columns,diagonal)
    monkeypatch.setattr(ImageCorrespondences,"prepare_fixed_p1_sampling",counted)
    report=comparison.run(args,production=False,test_backend_label="TEST MOCK: eager-backend capture, NOT Inductor performance evidence")
    assert report["status"]=="complete" and report["comparison_kind"]=="frozen_points"
    assert report['pair_name']=='he_to_ki67'
    assert report['source_moving'].endswith('ki67_moving512.png')
    assert report['source_matches'].endswith('he_to_ki67_raw_matches.json')
    assert report["speed_evidence_eligible"] is False
    assert "No hidden prior warmup" in report["cold_scope"]
    assert "NOT a point-compiler comparison" in report["cold_scope"]
    assert priors.make_joint_priors is factory
    assert ImageCorrespondences.prepare_fixed_p1_sampling is counted
    assert preparations==[(9,9,"ac")]*7
    runs=report["runs"]
    assert [row["backend"] for row in runs]==["existing","frozen"]+["existing","frozen","frozen","existing"]*3
    assert all(row["joint_prior_backend"]=="inductor" and len(row["factory_calls"])==1 for row in runs)
    assert all(row["match_p1_sampling"]==row["backend"] for row in runs)
    assert all(len(row["point_preparation_calls"])==(row["backend"]=="frozen") for row in runs)
    assert all(row["gradient_steps"]==4 and row["evaluations"]==8 and row["objective_evaluations"]==18 for row in runs)
    assert all(row["saved_binary_certificate"]["valid"] and row["failed_trials"]==0 for row in runs)
    assert all(row["map_max_abs_delta"]<1e-10 and abs(row["final_total_delta"])<1e-10 and row["counters_equal"] for row in report["comparisons"])
    assert set(report["warm_summary"]["complete_call_medians"])=={"existing","frozen"}
    assert "existing_over_frozen_median" in report["warm_summary"]
    assert "eager_over_compiled_median" not in report["warm_summary"]
    assert report["dispatch_arms"]==dict(existing=dict(joint_prior_backend="inductor",match_p1_sampling="existing"),
                                         frozen=dict(joint_prior_backend="inductor",match_p1_sampling="frozen"))


def test_actual_14_packed_diagnostic_applications_hold_priors_and_points_fixed(tmp_path,monkeypatch):
    args=args_for(tmp_path);args.comparison_kind="trial_diagnostics"
    capture_factory(monkeypatch)
    report=comparison.run(args,production=False,test_backend_label="TEST MOCK: CPU eager capture, NOT GPU performance evidence")
    assert report["status"]=="complete" and report["speed_evidence_eligible"] is False
    assert [row["backend"] for row in report["runs"]]==["existing","packed"]+["existing","packed","packed","existing"]*3
    assert all(row["joint_prior_backend"]=="inductor" and row["match_p1_sampling"]=="frozen" for row in report["runs"])
    assert all(row["trial_diagnostics"]==row["backend"] for row in report["runs"])
    assert all(row["map_bitwise_equal"] and row["counters_equal"] and row["trial_structure_equal"]
               and row["selected_stage_equal"] and row["failures_equal"]
               and row["max_abs_trial_total_delta"]==0 for row in report["comparisons"])
    assert "existing_over_packed_median" in report["warm_summary"]
    assert "existing_over_frozen_median" not in report["warm_summary"]
    assert "packing" in report["warm_summary"]["diagnostic_cold_scope"].lower()


def test_wrong_packed_reported_dispatch_stops_comparison(tmp_path,monkeypatch):
    args=args_for(tmp_path);args.comparison_kind="trial_diagnostics"
    capture_factory(monkeypatch)
    original=comparison.application.optimize
    def wrong(config):
        result=original(config);result["trial_diagnostics"]="wrong";return result
    monkeypatch.setattr(comparison.application,"optimize",wrong)
    report=comparison.run(args,production=False,test_backend_label="TEST MOCK incorrect diagnostic dispatch")
    assert report["status"]=="incomplete_comparison_no_speed_claim" and len(report["runs"])==1
    assert any("trial_diagnostics" in reason for reason in report["runs"][0]["invalid_reasons"])


@pytest.mark.parametrize("field",["joint_prior_backend","match_p1_sampling"])
def test_points_wrong_reported_dispatch_stops_comparison(tmp_path,monkeypatch,field):
    args=args_for(tmp_path);args.comparison_kind="frozen_points"
    capture_factory(monkeypatch)
    original=comparison.application.optimize
    def wrong(config):
        result=original(config)
        result[field]="wrong"
        return result
    monkeypatch.setattr(comparison.application,"optimize",wrong)
    report=comparison.run(args,production=False,test_backend_label="TEST MOCK incorrect dispatch")
    assert report["status"]=="incomplete_comparison_no_speed_claim"
    assert len(report["runs"])==1 and "warm_summary" not in report
    assert any(field in reason for reason in report["runs"][0]["invalid_reasons"])


def test_compile_failure_preserves_first_completed_attempt_and_does_not_fallback(tmp_path,monkeypatch):
    args=args_for(tmp_path)
    def fail(backend=None):
        raise RuntimeError("deliberate compiler failure")
    monkeypatch.setattr(priors,"make_joint_priors",fail)
    report=comparison.run(args,production=False,test_backend_label="TEST MOCK compiler failure")
    assert report["status"]=="incomplete_comparison_no_speed_claim"
    assert len(report["runs"])==2 and len(report["unattempted_runs"])==12
    assert report["runs"][0]["status"]=="complete"
    assert report["runs"][1]["status"]=="failed_no_fallback"
    assert report["runs"][1]["factory_calls"][0]["status"]=="failed"
    assert priors.make_joint_priors is fail
    assert "warm_summary" not in report and report["speed_evidence_eligible"] is False
    assert json.loads(args.output.read_text())["runs"][0]["gradient_steps"]==4


@pytest.mark.parametrize("defect",["gradient_steps","saved_binary_certificate","failed_trials"])
def test_invalid_budget_or_certificate_suppresses_speed_claim(tmp_path,monkeypatch,defect):
    args=args_for(tmp_path)
    original=comparison.application.optimize
    def changed(config):
        report=original(config)
        report[defect]={"valid":False} if defect=="saved_binary_certificate" else 1
        return report
    monkeypatch.setattr(comparison.application,"optimize",changed)
    report=comparison.run(args,production=False,test_backend_label="TEST MOCK invalid result")
    assert report["status"]=="incomplete_comparison_no_speed_claim" and len(report["runs"])==1
    assert report["runs"][0]["invalid_reasons"] and "warm_summary" not in report


def test_collision_recipe_and_test_label_guards(tmp_path):
    args=args_for(tmp_path)
    with pytest.raises(ValueError,match="test backend label"):
        comparison.run(args,test_backend_label="mock")
    args.output.with_name("comparison_warm_g2_3_eager.npz").touch()
    with pytest.raises(FileExistsError):
        comparison.run(args,production=False)


def test_cold_costs_excluded_from_warm_medians():
    runs=comparison.schedule()
    for row in runs:
        row["complete_call_seconds"]=(100 if row["backend"]=="eager" else 200) if row["phase"]=="observed_cold" else (4 if row["backend"]=="eager" else 2)
    summary=comparison.warm_summary(runs)
    assert summary["complete_call_medians"]==dict(eager=4,inductor=2)
    assert summary["eager_over_compiled_median"]==2
    assert summary["cold_complete_call_seconds"]==dict(eager=100,inductor=200)
    assert summary["observed_cold_compiled_minus_warm_compiled_median_seconds"]==198


@pytest.mark.parametrize("device,index",[("cuda",0),("cuda:2",2)])
def test_cuda_initialization_resolves_unspecified_recipe_index(tmp_path,monkeypatch,device,index):
    args=args_for(tmp_path)
    original=comparison.profile.load_configuration
    def configuration(*values,**kwargs):
        config=original(*values,**kwargs)
        config.device=device
        return config
    selected=[]
    monkeypatch.setattr(comparison.profile,"load_configuration",configuration)
    monkeypatch.setattr(comparison.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(comparison.torch.cuda,"set_device",selected.append)
    def stop():
        raise RuntimeError("stop before actual CUDA initialization")
    monkeypatch.setattr(comparison.torch.cuda,"init",stop)
    with pytest.raises(RuntimeError,match="stop before actual CUDA"):
        comparison.run(args,production=False)
    assert selected==[index]
