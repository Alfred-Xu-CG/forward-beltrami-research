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
