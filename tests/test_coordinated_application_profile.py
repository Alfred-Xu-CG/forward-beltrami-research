"""Instrumentation observes original calls and does not alter their arithmetic."""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import coordinated_lung_all20 as cohort
from tools import coordinated_real_case as application
from test_coordinated_lung_all20 import assets,fake_match,fake_optimize,fake_dhr


def module():return importlib.import_module("tools.coordinated_application_profile")


def test_phase_wrappers_restore_even_when_work_raises():
    profile=module()
    original=(application.Evidence.__call__,torch.nn.functional.grid_sample,
        torch.autograd.backward,torch.optim.Adam.step)
    with pytest.raises(RuntimeError,match="deliberate"):
        with profile.instrumentation():
            assert application.Evidence.__call__ is not original[0]
            raise RuntimeError("deliberate")
    assert (application.Evidence.__call__,torch.nn.functional.grid_sample,
        torch.autograd.backward,torch.optim.Adam.step)==original


def event(name,device,cpu,device_self,*,annotation=False,cpu_total=None,device_total=None):
    return SimpleNamespace(key=name,device_type=device,count=1,is_user_annotation=annotation,is_legacy=False,
        self_cpu_time_total=cpu,cpu_time_total=cpu if cpu_total is None else cpu_total,
        self_device_time_total=device_self,device_time_total=device_self if device_total is None else device_total)


def test_cuda_kernel_totals_are_not_double_counted_with_cpu_linked_events_or_nested_ranges():
    summary=module().summarize_events([
        event("application.complete", "CPU",40,0,annotation=True,cpu_total=150,device_total=100),
        event("aten::add", "CPU",60,20),event("kernel_add", "CUDA",0,25),
        event("cuda_annotation", "CUDA",0,200,annotation=True)],cuda_requested=True)
    assert summary["self_cpu_total_ms"]==pytest.approx(.1)
    assert summary["self_cpu_operator_total_ms"]==pytest.approx(.06)
    assert summary["self_cuda_kernel_total_ms"]==pytest.approx(.025)
    assert summary["cpu_operator_linked_self_cuda_total_ms"]==pytest.approx(.02)
    assert summary["ranges"][0]["inclusive_cuda_ms"]==pytest.approx(.1)
    assert "non-additive" in summary["range_scope"]
    assert "memory copies (memcpy)" in summary["cuda_total_scope"]
    assert "memset" in summary["cuda_total_scope"]
    assert "not compute-only time or FLOPs" in summary["cuda_total_scope"]


def fixture_report(tmp_path):
    args=assets(tmp_path)
    cohort.run(args,production=False,match_extractor=fake_match,optimizer=fake_optimize,dhr_runner=fake_dhr)
    return args.output/"predictions.json"


def test_three_real_tiny_runs_preserve_recipe_maps_values_counts_and_restore_wrappers(tmp_path):
    profile=module();source=fixture_report(tmp_path)
    # PyTorch itself installs a class-level profile hook on first Adam creation.
    # Establish that normal state before testing our reversible instrumentation.
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    step=torch.optim.Adam.step
    report=profile.run(argparse.Namespace(predictions=source,output=tmp_path/"profile.json",trace=None),production=False)
    assert torch.optim.Adam.step is step
    assert report["status"]=="complete" and report["matcher_executed"] is False
    assert report["activities"]==["CPU"]
    assert [row["kind"] for row in report["runs"]]==["unprofiled_before","profiled","unprofiled_after"]
    assert all(row["gradient_steps"]==4 and row["evaluations"]==8 for row in report["runs"])
    assert all(row["saved_binary_certificate"]["valid"] for row in report["runs"])
    assert all(row["map_bitwise_equal"] and row["counters_equal"] for row in report["agreement"].values())
    assert all(row["final_total_delta"]==0 for row in report["agreement"].values())
    assert report["profile"]["self_cpu_total_ms"]>0
    assert report["profile"]["self_cuda_kernel_total_ms"] is None
    names={row["name"] for row in report["profile"]["ranges"]}
    assert {"application.complete","application.evidence","application.decoder","application.backward",
        "application.adam_step","application.prior_arap","application.raster_sampling"}<=names
    assert report["profile_overhead_vs_bracket_mean"]>0
    saved=json.loads((tmp_path/"profile.json").read_text())
    assert saved["source_matches"]==str(source.parent/"he_to_cc10_raw_matches.json")


def test_changed_recipe_is_rejected_before_profile_or_execution(tmp_path):
    profile=module();source=fixture_report(tmp_path)
    report=json.loads(source.read_text())
    report["rows"][0]["methods"]["analytic"]["configuration"]["learning_rate"]*=2
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="recipe"):
        profile.load_configuration(source,tmp_path/"new.npz",production=False)


def test_existing_output_is_not_overwritten(tmp_path):
    path=tmp_path/"profile.json";path.write_text("preserve")
    with pytest.raises(FileExistsError):
        module().run(argparse.Namespace(predictions=tmp_path/"missing.json",output=path,trace=None))
    assert path.read_text()=="preserve"


def test_requested_pair_uses_its_exact_row_recipe_and_raw_matches(tmp_path):
    profile=module();source=fixture_report(tmp_path)
    default=profile.load_configuration(source,tmp_path/'default.npz',production=False)
    explicit=profile.load_configuration(source,tmp_path/'explicit.npz',production=False,pair_name='he_to_cc10')
    assert {key:value for key,value in vars(default).items() if key!='output'}=={
        key:value for key,value in vars(explicit).items() if key!='output'}
    selected=profile.load_configuration(source,tmp_path/'selected.npz',production=False,pair_name='he_to_ki67')
    recorded=next(row for row in json.loads(source.read_text())['rows'] if row['name']=='he_to_ki67')
    for key in ('fixed','moving','affine'):
        assert getattr(selected,key)==Path(recorded[key])
    assert selected.matches==source.parent/'he_to_ki67_raw_matches.json'
    assert selected.moving.name=='ki67_moving512.png'
    assert selected.learning_rate==recorded['methods']['analytic']['configuration']['learning_rate']


@pytest.mark.parametrize('defect',['unknown','duplicate','row_direction','configuration','raw_direction','raw_affine'])
def test_requested_pair_refuses_unknown_ambiguous_or_mismatched_sources(tmp_path,defect):
    profile=module();source=fixture_report(tmp_path);value=json.loads(source.read_text())
    row=next(row for row in value['rows'] if row['name']=='he_to_ki67')
    requested='not_a_pair' if defect=='unknown' else 'he_to_ki67'
    if defect=='duplicate':value['rows'].append(row)
    elif defect=='row_direction':row['moving_stain']='cc10'
    elif defect=='configuration':row['methods']['analytic']['configuration']['moving']=str(tmp_path/'wrong.png')
    elif defect in ('raw_direction','raw_affine'):
        path=source.parent/row['raw_matches']['path'];raw=json.loads(path.read_text())
        if defect=='raw_direction':raw['moving']=str(tmp_path/'cc10_moving512.png')
        else:raw['post_affine_offset']=[.01,0.]
        path.write_text(json.dumps(raw))
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError):profile.load_configuration(source,tmp_path/'new.npz',production=False,pair_name=requested)


def test_profile_run_forwards_requested_pair_and_reports_identity(tmp_path):
    source=fixture_report(tmp_path)
    report=module().run(argparse.Namespace(predictions=source,output=tmp_path/'selected_profile.json',
        trace=None,pair_name='he_to_ki67'),production=False)
    assert report['pair_name']=='he_to_ki67' and report['source_moving'].endswith('ki67_moving512.png')
    assert report['source_matches'].endswith('he_to_ki67_raw_matches.json')
