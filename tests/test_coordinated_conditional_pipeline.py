"""CPU injected lifecycle, literal tables, failure denominator and peak accounting."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import coordinated_conditional_pipeline as pipeline


class FakeCuda:
    def __init__(self):
        self.allocated = self.reserved = 0
        self.resets = 0

    def synchronize(self, device):
        pass

    def max_memory_allocated(self, device):
        return self.allocated

    def max_memory_reserved(self, device):
        return self.reserved

    def reset_peak_memory_stats(self, device=None):
        self.resets += 1
        self.allocated = self.reserved = 0


@pytest.mark.parametrize("raises", [False, True])
def test_setup_nested_reset_final_segment_maxima_and_restore(raises):
    cuda = FakeCuda()
    original = cuda.reset_peak_memory_stats
    collector = pipeline.ResetAwarePeaks("cuda", cuda_api=cuda)
    try:
        with collector:
            cuda.allocated, cuda.reserved = 900, 1000  # Setup transient, erased internally.
            cuda.reset_peak_memory_stats("cuda")
            cuda.allocated, cuda.reserved = 400, 1500
            cuda.reset_peak_memory_stats(device="cuda")
            cuda.allocated, cuda.reserved = 700, 800
            if raises:
                raise ValueError("injected")
    except ValueError:
        assert raises
    assert cuda.reset_peak_memory_stats == original
    assert not pipeline.ResetAwarePeaks._active
    assert collector.report()["peak_allocated_bytes"] == 900
    assert collector.report()["peak_reserved_bytes"] == 1500
    assert collector.report()["captured_segments"] == 3
    assert cuda.resets == 3


def test_cpu_peak_scope_and_nested_rejection():
    with pipeline.ResetAwarePeaks("cpu") as cpu:
        assert cpu.report()["peak_allocated_bytes"] is None
    cuda = FakeCuda()
    with pipeline.ResetAwarePeaks("cuda", cuda_api=cuda):
        with pytest.raises(RuntimeError, match="non-nested"):
            with pipeline.ResetAwarePeaks("cuda", cuda_api=cuda):
                pass


def table():
    points = np.column_stack((np.linspace(.1,.7,10), np.linspace(.2,.8,10))).tolist()
    return dict(status="ok", fixed="fixed.png", moving="moving.png", image_side=512,
                post_affine_matrix=np.eye(2,dtype=np.float32).tolist(), post_affine_offset=[0.,0.],
                source_points_unit=points, target_points_unit=copy.deepcopy(points), confidence=[.8]*10,
                coordinate_convention="fixed unit q -> affine-aligned moving unit p; full moving point is A p+b",
                targets_manual_landmarks_or_dense_teacher_loaded=False)


def test_table_comparison_shape_values_and_no_mutation():
    original = table()
    fresh = copy.deepcopy(original)
    assert pipeline.compare_tables(fresh, original)["all_arrays_exact"]
    fresh["confidence"][0] += .01
    report = pipeline.compare_tables(fresh, original)
    assert not report["all_arrays_exact"]
    assert report["arrays"]["confidence"]["maximum_absolute_difference"] == pytest.approx(.01)
    fresh["source_points_unit"].pop()
    report = pipeline.compare_tables(fresh, original)
    assert report["arrays"]["source_points_unit"]["maximum_absolute_difference"] is None
    assert original == table()


def fixture(tmp_path):
    sg = table()
    np.savez(tmp_path/"affine.npz", post_affine_matrix=np.eye(2,dtype=np.float32),
             post_affine_offset=np.zeros(2,dtype=np.float32))
    (tmp_path/"sg.json").write_text(json.dumps(sg),encoding="utf-8")
    (tmp_path/"ma.json").write_text(json.dumps(sg),encoding="utf-8")
    (tmp_path/"fused.json").write_text(json.dumps(pipeline.fuse_tables(sg,sg)),encoding="utf-8")
    cfg = dict(method="analytic",loss="mind",output_selection="best_full",grid_side=257,image_side=512,
        image_levels=[32,64,128,256,512],levels=[17,33,65,129,257],inner_steps=30,cycles=1,
        strain_weight=3.,strain_model="p1_arap",shape_weight=1e-4,match_weight=.1,oob_weight=1.,
        minimum_jacobian=.001,precision="float64",image_precision="float32",interpolation="p1_ac",
        mind_frame="shared_affine",fixed="fixed.png",moving="moving.png",affine=str(tmp_path/"affine.npz"),
        matches=str(tmp_path/"sg.json"),output="old.npz",device="cpu",learning_rate=.004)
    cases = [dict(name=f"case{i}",cohort="miit" if i<3 else "existing",original_configuration=copy.deepcopy(cfg)) for i in range(25)]
    for ma in (True,False):
        rows = [dict(name=r["name"],status="ok",match_table="ma.json",configuration=dict(matches="fused.json")) for r in cases]
        (tmp_path/("ma_manifest.json" if ma else "fusion_manifest.json")).write_text(json.dumps(
            dict(prediction_complete=True,annotations_read=False,rows=rows)),encoding="utf-8")
    return cases


@pytest.mark.parametrize("failure_mode", ["none", "several", "setup", "close"])
def test_actual_fresh_inputs_order_failures_costs_and_no_fallback(tmp_path,monkeypatch,failure_mode):
    cases = fixture(tmp_path)
    monkeypatch.setattr(pipeline,"source_cases",lambda *_:copy.deepcopy(cases))
    out = tmp_path/"run"
    events = []
    def sg(args):
        saved = json.loads((out/"predictions.json").read_text())
        assert len(saved["rows"]) == 25 and not saved["prediction_complete"]
        index = int(args.output.stem.removeprefix("case"))
        events.append(("sg",index))
        if failure_mode == "several" and index == 1:
            raise ValueError("SG failure")
        record = table()
        record["confidence"][0] = .75  # Genuine changed fresh evidence, MUST remain used.
        args.output.write_text(json.dumps(record),encoding="utf-8")
        return record
    class Model:
        def __init__(self,*args,**kwargs):
            assert len(events) == 25
            events.append(("setup",0))
            if failure_mode == "setup":
                raise ValueError("setup failure")
            self.setup_report = {"setup_seconds":.1}
        def extract(self,fixed,moving,affine,*,output):
            index = int(output.stem.removeprefix("case"))
            events.append(("ma",index))
            if failure_mode == "several" and index == 2:
                raise ValueError("MA failure")
            record = table()
            output.write_text(json.dumps(record),encoding="utf-8")
            return record
        def close(self):
            events.append(("close",0))
            if failure_mode == "close":
                raise ValueError("release failure")
    def optimize(cfg):
        assert ("close",0) in events and len([e for e in events if e[0]=="ma"]) == 25
        assert json.loads((out/"predictions.json").read_text())["extraction_complete"]
        assert cfg.match_weight == .2 and cfg.inner_steps*len(cfg.levels)*2 == 300
        new = json.loads(cfg.matches.read_text())
        assert new["confidence"][0] == pytest.approx(.75/(9*.8+.75))
        index = int(cfg.output.stem.split("_")[0].removeprefix("case"))
        events.append(("opt",index))
        if failure_mode == "several" and index == 24:
            raise ValueError("optimizer failure")
        return dict(gradient_steps=300,failed_trials=0)
    def scoring(manifest,output):
        assert manifest["prediction_complete"] and len(manifest["rows"]) == 25
        assert all(r["status"] in ("ok","failed") for r in manifest["rows"])
        events.append(("scoring",0))
    result = pipeline.run("unused","unused",tmp_path/"ma_manifest.json",tmp_path/"fusion_manifest.json",
        out,"source","checkpoint",sg_extractor=sg,model_factory=Model,optimizer=optimize,
        export_validator=lambda *_:.5,scoring_adapter=scoring)
    failures = {"none":0,"several":3,"setup":25,"close":25}[failure_mode]
    assert result["failed"] == failures and result["successful"] == 25-failures
    assert result["attempt_denominator"] == 25 and result["prediction_complete"]
    assert not result["all_tables_exact"]
    assert result["whole_batch_seconds"] > 0 and result["whole_batch_memory"]["peak_reserved_bytes"] is None
    assert all(r["complete_call_seconds"] == sum(r["costs"].values()) for r in result["rows"])
    assert result["first_case"] == "case0" and len(result["later_cases"]) == 24
    assert result["sg_setup_attempts"] == 25 and result["ma_setup_attempts"] == 1
    assert result["batch_seconds_per_attempt"] == result["whole_batch_seconds"]/25
    saved_times = [r["completed_map_elapsed_from_batch_start"] for r in result["rows"] if r["status"] == "ok"]
    assert saved_times == sorted(saved_times)
    assert result["time_to_first_saved_map"] == (saved_times[0] if saved_times else None)
    assert all(t < result["whole_batch_seconds"] for t in saved_times)
    assert json.loads((tmp_path/"sg.json").read_text()) == table()
    if failure_mode == "several":
        assert result["rows"][1]["optimization_status"] == "skipped"
        assert result["rows"][2]["optimization_status"] == "skipped"
    if failure_mode in ("setup", "close"):
        assert not any(e[0] == "opt" for e in events)
