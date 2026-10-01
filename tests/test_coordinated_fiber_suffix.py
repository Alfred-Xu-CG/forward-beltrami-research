"""Same-prefix and evidence/configuration tests for the suffix diagnostic."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from tools import coordinated_fiber_suffix as suffix


def production_configuration():
    return dict(method="analytic",coordinate_mode="alternating",fine_patch_cells=0,
        proposal_filter_steps=0,cycles=1,interpolation="p1_ac",p1_sampling="frozen",
        control_hierarchy="nested_p1",nested_evaluation="coarse_exact",output_selection="best_full",
        levels=[17,33,65,129,1025],grid_side=1025,image_side=1024,
        image_levels=[64,128,256,512,1024],inner_steps=30,learning_rate=.004,
        lr_calibration="edge",fixed="fixed.png",moving="moving.png",affine="a.npz",
        matches="matches.json",strain_model="p1_arap",strain_weight=3.,shape_weight=.0001,
        loss="mind",mind_order="transport",match_weight=.01,geometry_backend="stage_cache")


@pytest.mark.parametrize("grid",[257,1025])
def test_configuration_preserves_every_setting_except_explicit_final_size_and_output(grid,tmp_path):
    original=production_configuration()
    report={"configuration":copy.deepcopy(original)}
    config=suffix.prepare_configuration(report,tmp_path/"baseline.npz",grid)
    expected=copy.deepcopy(original)
    expected.update(grid_side=grid,levels=[17,33,65,129,grid],output=tmp_path/"baseline.npz")
    for key in ("fixed","moving","affine","matches"):expected[key]=Path(expected[key])
    assert vars(config)==expected
    assert report["configuration"]==original
    assert suffix._physical_initial_rms(config)==pytest.approx(.004*16/(grid-1))


@pytest.mark.parametrize("changes",[
    {"method":"radial"},{"coordinate_mode":"joint"},{"fine_patch_cells":16},
    {"proposal_filter_steps":1},{"cycles":2},{"interpolation":"q1"},
    {"image_side":512},{"image_levels":[32,64,128,256,1024]},
    {"inner_steps":29},{"control_hierarchy":"fixed"},{"output_selection":"last"},
    {"p1_sampling":"existing"},{"learning_rate":float("nan")},
])
def test_reject_unsupported_production_config(changes,tmp_path):
    config=production_configuration();config.update(changes)
    with pytest.raises(ValueError):suffix.prepare_configuration({"configuration":config},tmp_path/"x",257)


def tiny_report(tmp_path,diagonal="ac"):
    raster=np.random.default_rng(13).integers(20,230,(16,16),dtype=np.uint8)
    Image.fromarray(raster).save(tmp_path/"fixed.png")
    Image.fromarray(np.roll(raster,1,axis=1)).save(tmp_path/"moving.png")
    # Deliberately float64 storage: production first rounds to the SAME float32
    # affine as the baseline and then promotes that value to geometry precision.
    matrix=np.array([[.99,.015],[-.005,1.01]],dtype=np.float64)
    offset=np.array([.003,-.002],dtype=np.float64)
    np.savez(tmp_path/"affine.npz",post_affine_matrix=matrix,post_affine_offset=offset)
    config=production_configuration()
    config.update(fixed=str(tmp_path/"fixed.png"),moving=str(tmp_path/"moving.png"),
        affine=str(tmp_path/"affine.npz"),matches=None,match_weight=0.,grid_side=9,
        levels=[5,9],image_side=16,image_levels=[8,16],inner_steps=2,precision="float64",
        image_precision="float32",device="cpu",threads=1,minimum_jacobian=.001,
        oob_weight=1.,interpolation="p1_"+diagonal)
    path=tmp_path/"source.json"
    path.write_text(json.dumps({"configuration":config}))
    return argparse.Namespace(report=path,output=tmp_path/"comparison.json",grid_side=9),config


def unchanged_solver(calls,reason="gradient_budget"):
    def solve(anchor,objective,**options):
        calls.append((anchor.detach().clone(),options))
        with torch.no_grad():value=float(objective(anchor)[0])
        return SimpleNamespace(best_vertices=anchor.detach().clone(),
            initial_objective=value,final_objective=value,best_objective=value,
            counts=dict(objective_evaluations=1,gradient_evaluations=0,
                geometry_rejections=int(reason=="rounded_geometry_rejected"),armijo_halvings=0),
            stop_reason=reason,calibration=dict(initial_displacement_rms=options["initial_displacement_rms"]),
            trace=[dict(kind="initial",accepted=True,objective=value)])
    return solve


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_real_tiny_baseline_and_mock_suffix_share_exact_prefix(tmp_path,diagonal):
    args,config=tiny_report(tmp_path,diagonal)
    calls=[]
    comparison=suffix.run(args,production=False,solver=unchanged_solver(calls))
    assert len(calls)==2
    with np.load(comparison["prefix_output"],allow_pickle=False) as archive:
        np.testing.assert_array_equal(calls[0][0].numpy(),archive["vertices"])
        np.testing.assert_array_equal(archive["coarse_vertices"],archive["vertices"][:,::2,::2])
        expected_prefix_objective=float(archive["best_prefix_objective"])
    assert calls[0][1]["direction"]==(1.,0.) and calls[1][1]["direction"]==(0.,1.)
    assert calls[0][1]["initial_displacement_rms"]==pytest.approx(.004*4/8)
    assert calls[0][1]["max_gradient_evaluations"]==30
    assert calls[0][1]["history_size"]==5 and calls[0][1]["max_backtracks"]==6
    assert "tangent_rescue" not in calls[0][1]
    assert comparison["fiber_final"]["total"]<=expected_prefix_objective+1e-7
    assert comparison["shared_transition_from_exact_oldnodes"]
    assert comparison["snapshot_bitwise_equal"]
    assert comparison["source_configuration"]==config
    assert comparison["baseline_suffix_counts"]["successful_gradient_steps"]==4
    assert comparison["baseline_suffix_counts"]["trial_objective_evaluations"]==6
    assert comparison["baseline_suffix_callback_elapsed_seconds"]>=0
    assert comparison["baseline_certificate"]["valid"] and comparison["fiber_certificate"]["valid"]
    assert comparison["fiber_solver_status"]=="budget_or_tolerance_reached"
    assert comparison["fiber_objective_evaluation_counts"]==dict(
        solver_objective_evaluations=2,stage_selection_objective_evaluations=2,
        final_report_objective_evaluations=1,complete_objective_evaluations=5)
    with np.load(comparison["fiber_output"],allow_pickle=False) as archive:
        assert archive["post_affine_matrix"].dtype==np.float32
        np.testing.assert_array_equal(archive["post_affine_matrix"],np.array([[.99,.015],[-.005,1.01]],dtype=np.float32))


def test_rounded_failure_retains_valid_best_and_stops_suffix(tmp_path):
    args,_=tiny_report(tmp_path)
    calls=[]
    result=suffix.run(args,production=False,solver=unchanged_solver(calls,"rounded_geometry_rejected"))
    assert len(calls)==1 and result["fiber_certificate"]["valid"]
    record=json.loads(Path(result["fiber_output"]).with_suffix(".json").read_text())
    assert record["stages"][0]["stop_reason"]=="rounded_geometry_rejected"
    assert result["fiber_suffix_counts"]["geometry_rejections"]==1
    assert result["fiber_solver_status"]=="solver_failure_valid_best_retained"


@pytest.mark.parametrize("reason,status",[
    ("nonfinite_initial_calibration","solver_failure_valid_best_retained"),
    ("minimum_step","early_numerical_stop_valid_best_retained"),
    ("objective_backtrack_limit","early_numerical_stop_valid_best_retained"),
    ("gradient_tolerance","budget_or_tolerance_reached"),
])
def test_numerical_stops_distinct_from_validity_and_normal_budget(reason,status):
    assert suffix._stop_status(reason)==status


@pytest.mark.parametrize("occupied",["comparison.json","comparison_prefix.npz",
    "comparison_baseline.npz","comparison_baseline.json","comparison_fiber.npz","comparison_fiber.json"])
def test_refuses_every_existing_output_before_baseline(tmp_path,monkeypatch,occupied):
    (tmp_path/occupied).write_text("existing")
    def forbidden(*args,**kwargs):pytest.fail("must not begin baseline")
    monkeypatch.setattr(suffix,"optimize",forbidden)
    args=argparse.Namespace(report=tmp_path/"missing.json",output=tmp_path/"comparison.json",grid_side=257)
    with pytest.raises(FileExistsError):suffix.run(args)
    assert (tmp_path/occupied).read_text()=="existing"


def test_baseline_suffix_counts_dont_count_failed_backward_as_success():
    rows=[dict(level=9,step=i) for i in (0,1)]
    record=dict(trace=rows,failures=[dict(level=9,step=1,reason="missing/nonfinite gradient")])
    counts=suffix._baseline_suffix_counts(record,9,2)
    assert counts["successful_gradient_steps"]==1
    assert counts["backward_attempts"]==2 and counts["trial_objective_evaluations"]==2


def test_tiny_actual_core_integration_keeps_objective_selected_valid_map(tmp_path):
    args,_=tiny_report(tmp_path)
    comparison=suffix.run(args,production=False)
    assert comparison["shared_transition_from_exact_oldnodes"] and comparison["fiber_certificate"]["valid"]
    with np.load(comparison["prefix_output"],allow_pickle=False) as archive:
        prefix_best=float(archive["best_prefix_objective"])
    assert comparison["fiber_final"]["total"]<=prefix_best+1e-7
    counts=comparison["fiber_suffix_counts"]
    assert 1<=counts["gradient_evaluations"]<=60
    assert counts["objective_evaluations"]>=counts["gradient_evaluations"]
    direct=json.loads(Path(comparison["fiber_output"]).with_suffix(".json").read_text())
    assert direct["stages"][0]["calibration"]["initial_displacement_rms"]==pytest.approx(.002)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_nonold_snapshot_roundoff_does_not_change_actual_transition(dtype,diagonal):
    coarse=suffix.identity_vertices(5,device="cpu").to(dtype)
    coarse[:,1:-1,1:-1,0]+=.003
    expected=suffix.refine_p1_vertices(coarse,2,diagonal)
    snapshot=expected.clone()
    snapshot[0,1,1,0]=torch.nextafter(snapshot[0,1,1,0],torch.tensor(float("inf"),dtype=dtype))
    assert not torch.equal(snapshot,expected)
    actual,recovered,metadata=suffix.reconstruct_transition(snapshot,5,
        device="cpu",dtype=dtype,diagonal=diagonal)
    torch.testing.assert_close(recovered,coarse,rtol=0,atol=0)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert metadata["shared_transition_from_exact_oldnodes"]
    assert not metadata["snapshot_bitwise_equal"]
    assert metadata["frozen_materialization_max_difference"]>0
    reference=suffix.identity_vertices(9,device="cpu").to(dtype)
    assert suffix.validate_q1_map(actual,reference)["valid"]
    suffix.require_nested_fine_margin(actual,suffix.q1_corner_determinants(reference.double()),.001,"test")


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_reuse_saved_prefix_never_reruns_baseline_or_uses_terminal_fiber(tmp_path,monkeypatch,diagonal):
    args,_=tiny_report(tmp_path,diagonal)
    original=suffix.run(args,production=False,solver=unchanged_solver([]))
    prefix_path=Path(original["prefix_output"])
    prefix_bytes=prefix_path.read_bytes()
    # Terminal fiber artifacts are not inputs to replay.
    Path(original["fiber_output"]).unlink()
    Path(original["fiber_output"]).with_suffix(".json").unlink()
    def forbidden(*args,**kwargs):pytest.fail("replay must not rerun baseline")
    monkeypatch.setattr(suffix,"optimize",forbidden)
    replay=argparse.Namespace(report=None,reuse_prefix_report=args.output,
        output=tmp_path/"replay.json",grid_side=9,tangent_rescue=True)
    calls=[]
    result=suffix.run(replay,production=False,solver=unchanged_solver(calls))
    with np.load(prefix_path,allow_pickle=False) as saved:
        np.testing.assert_array_equal(calls[0][0].numpy(),saved["vertices"])
        assert result["fiber_final"]["total"]<=float(saved["best_prefix_objective"])+1e-7
    assert prefix_bytes==prefix_path.read_bytes()
    assert all(options["tangent_rescue"] for _,options in calls)
    assert result["baseline_executed_this_run"] is False
    assert result["baseline_complete_call_seconds"]==0.
    assert result["reused_prefix_load_seconds"]>=0.
    assert result["historical_baseline_complete_call_seconds"]==original["baseline_complete_call_seconds"]
    assert result["baseline_suffix_counts"]==original["baseline_suffix_counts"]
    assert result["setup_complete_objective_evaluations"]==2
    assert np.isfinite(result["reevaluated_best_prefix_objective"])
    assert result["best_prefix_objective_difference"]==pytest.approx(
        result["reevaluated_best_prefix_objective"]-result["stored_best_prefix_objective"])
    assert result["source_configuration"]==original["actual_configuration"]
    assert result["baseline_output"]==original["baseline_output"]
    assert not (tmp_path/"replay_baseline.npz").exists()
    assert not (tmp_path/"replay_baseline.json").exists()
    assert result["fiber_certificate"]["valid"]


def test_reuse_rejects_a_different_final_control_size_before_evidence(tmp_path,monkeypatch):
    args,_=tiny_report(tmp_path)
    suffix.run(args,production=False,solver=unchanged_solver([]))
    def forbidden(*args,**kwargs):pytest.fail("must reject before evidence or baseline")
    monkeypatch.setattr(suffix,"optimize",forbidden)
    monkeypatch.setattr(suffix,"_make_evidence",forbidden)
    replay=argparse.Namespace(report=None,reuse_prefix_report=args.output,
        output=tmp_path/"replay.json",grid_side=17)
    with pytest.raises(ValueError,match="same final control"):
        suffix.run(replay,production=False,solver=unchanged_solver([]))


@pytest.mark.parametrize("field",["coarse_vertices","boundary_reference","post_affine_matrix"])
def test_reuse_rejects_changed_old_nodes_reference_or_affine(tmp_path,monkeypatch,field):
    args,_=tiny_report(tmp_path)
    original=suffix.run(args,production=False,solver=unchanged_solver([]))
    path=Path(original["prefix_output"])
    with np.load(path,allow_pickle=False) as saved:
        payload={key:saved[key].copy() for key in saved.files}
    payload[field].flat[0]+=.001
    np.savez(path,**payload)
    def forbidden(*args,**kwargs):pytest.fail("invalid replay must never reach optimization")
    monkeypatch.setattr(suffix,"optimize",forbidden)
    replay=argparse.Namespace(report=None,reuse_prefix_report=args.output,
        output=tmp_path/"replay.json",grid_side=9)
    with pytest.raises(ValueError,match="saved prefix"):
        suffix.run(replay,production=False,solver=forbidden)


@pytest.mark.parametrize("rescue",[False,True])
def test_tiny_actual_core_replays_saved_prefix_with_optional_rescue(tmp_path,monkeypatch,rescue):
    args,_=tiny_report(tmp_path)
    suffix.run(args,production=False,solver=unchanged_solver([]))
    def forbidden(*args,**kwargs):pytest.fail("replay must not rerun baseline")
    monkeypatch.setattr(suffix,"optimize",forbidden)
    replay=argparse.Namespace(report=None,reuse_prefix_report=args.output,
        output=tmp_path/"replay.json",grid_side=9,tangent_rescue=rescue)
    torch.set_num_threads(2)
    result=suffix.run(replay,production=False)
    assert torch.get_num_threads()==1
    assert result["tangent_rescue"] is rescue
    assert result["fiber_certificate"]["valid"]
    assert result["fiber_final"]["total"]<=min(result["reevaluated_best_prefix_objective"],
        result["reevaluated_initial_objective"])+1e-7
    assert 1<=result["fiber_suffix_counts"]["gradient_evaluations"]<=60
    assert result["setup_complete_objective_evaluations"]==2


@pytest.mark.parametrize("threads",[0,True,1.5])
def test_replay_rejects_nonpositive_or_noninteger_threads(tmp_path,monkeypatch,threads):
    args,_=tiny_report(tmp_path)
    original=suffix.run(args,production=False,solver=unchanged_solver([]))
    baseline_path=Path(original["baseline_output"]).with_suffix(".json")
    baseline=json.loads(baseline_path.read_text())
    baseline["configuration"]["threads"]=threads
    baseline_path.write_text(json.dumps(baseline))
    original["actual_configuration"]["threads"]=threads
    args.output.write_text(json.dumps(original))
    def forbidden(*args,**kwargs):pytest.fail("invalid threads must be rejected before evidence")
    monkeypatch.setattr(suffix,"_make_evidence",forbidden)
    replay=argparse.Namespace(report=None,reuse_prefix_report=args.output,
        output=tmp_path/"replay.json",grid_side=9)
    with pytest.raises(ValueError,match="threads.*positive integer"):
        suffix.run(replay,production=False,solver=unchanged_solver([]))
