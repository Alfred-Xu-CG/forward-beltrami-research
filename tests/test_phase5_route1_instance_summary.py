"""Receipt-only tests: no benchmark numbers are fabricated by the summarizer."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/phase5/route1_instance_summary.py"


def module():
    spec = importlib.util.spec_from_file_location("instance_summary_test", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def settings(backend="direct", dtype="float64"):
    if backend == "direct":
        return dict(variant="SuperLU factorization; two-coordinate RHS; reused transpose factor",
                    relative_tolerance=None, absolute_tolerance=None, max_iterations=None,
                    internal_arithmetic="float64")
    return dict(variant=("unpreconditioned BiCGStab; explicit true residual each iteration"
                         if backend == "directed_iterative" else
                         "unpreconditioned CG; candidate/periodic true-residual replacement"),
                relative_tolerance=(1e-5 if dtype == "float32" else
                                    1e-10 if backend == "directed_iterative" else 1e-11),
                absolute_tolerance=0., max_iterations=500 if backend == "directed_iterative" else 1000,
                internal_arithmetic=dtype)


def receipt():
    metrics = dict(flip_count=0, minimum_signed_area=.05, minimum_area_ratio=.4,
                   boundary_order_min_gap=.2, global_injectivity_certificate=True,
                   map_rmse=.1, maximum_map_error=.2, mu_rmse=.1, maximum_mu_error=.2)
    target_metrics = dict(metrics, map_rmse=None, maximum_map_error=None,
                          mu_rmse=None, maximum_mu_error=None)
    trace = [dict(iteration=i, outer_step=i, objective=loss, map_rmse=.1,
                  forward_seconds=.1, backward_seconds=.1 if i == 0 else 0.,
                  audit_seconds=.02, optimizer_step_seconds=.03 if i == 0 else 0.,
                  observation_wall_seconds=wall, primal_solves=i+1, adjoint_solves=i,
                  global_solves=2*i+1, flip_count=0, minimum_area_ratio=.4,
                  topology_certified=True)
             for i, loss, wall in [(0, .2, .15), (1, .1, .4)]]
    run = dict(task="supervised_map", backend="DirectTutteLayer", backend_requested="direct",
               status="success", optimizer_name="adam", control_vertices_per_side=3,
               control_vertex_count=9, image_resolution=8, initial_objective=.2,
               final_objective=.1, best_objective=.1, primal_solves=2, adjoint_solves=1,
               global_solves=3, target_setup_solves=0, target_setup_seconds=0.,
               wall_seconds=.5, wrapper_wall_seconds=.8, objective_threshold=.15,
               evaluations_to_threshold=2, global_solves_to_threshold=3,
               wall_seconds_to_threshold=.4, all_iterates_certified=True,
               final_metrics=metrics, trace=trace, target_object_identity=123,
               rss_before_bytes=100, rss_after_bytes=200, process_hwm_bytes=300,
               peak_gpu_allocated_bytes=None, peak_gpu_reserved_bytes=None,
               gpu_baseline_allocated_bytes=None, gpu_baseline_reserved_bytes=None,
               threshold_semantics="first independently audited evaluation; LBFGS trials included",
               solver_settings=settings())
    return dict(schema="phase5_route1_instance_v1", research_question="I1 supervised fitting",
                configuration=dict(task="supervised_map", control_vertices_per_side=3,
                    control_vertex_count=9, face_count=8, image_resolution=8,
                    optimizer="adam", steps=1, learning_rate=.1, objective_threshold=.15,
                    target_strength=.25, seed=20260922, image_name=None,
                    dtype="float64", device="cpu", backends=["direct"],
                    warp_convention="fixed-to-moving backward coordinates; no inverse is computed"),
                environment=dict(hostname="cpu-host", device="cpu", commit="a"*40,
                    platform="test", python="3", torch="2", numpy="1", scipy="1",
                    torch_threads=1, cuda_visible_devices=None),
                shared_target=dict(generated_once=True, setup_primal_solves=1,
                    setup_seconds=.3, metrics=target_metrics, object_identity=123,
                    control_shape=[9, 2], dense_shape=[8, 8, 2],
                    control_sum=8.5, dense_sum=60., control_l2=3., dense_l2=8.), runs=[run])


def save(tmp_path, data, name="input.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_flatten_complete_row_and_trace_sums(tmp_path):
    rows = module().summarize_receipts([save(tmp_path, receipt())])
    assert len(rows) == 1
    row = rows[0]
    assert (row["N"], row["resolution"], row["backend"], row["commit"]) == (3, 8, "direct", "a"*40)
    assert row["forward_seconds_total"] == pytest.approx(.2)
    assert row["backward_seconds_total"] == pytest.approx(.1)
    assert row["optimizer_seconds_total"] == pytest.approx(.03)
    assert row["audit_seconds_total"] == pytest.approx(.04)
    assert row["timed_components_seconds_total"] == pytest.approx(.37)
    assert row["wall_seconds"] == .5 and row["wrapper_wall_seconds"] == .8
    assert row["final_minimum_area_ratio"] == .4
    assert row["trace_max_flip_count"] == 0 and row["all_iterates_certified"] is True
    assert row["global_solves_to_threshold"] == 3
    assert row["inverse_consistency"] is None
    assert row["target_identity_check"] == "matched"


def test_cross_host_target_summary_ignores_identity_and_setup_time(tmp_path):
    a = receipt()
    b = copy.deepcopy(a)
    b["environment"].update(hostname="gpu-host", device="cuda:0")
    b["configuration"].update(device="cuda:0", backends=["symmetric"])
    b["shared_target"].update(object_identity=456, setup_seconds=9.)
    b["runs"][0].update(target_object_identity=456, backend_requested="symmetric",
                         backend="MatrixFreeSymmetricTutteLayer", peak_gpu_allocated_bytes=100,
                         peak_gpu_reserved_bytes=200, gpu_baseline_allocated_bytes=0,
                         gpu_baseline_reserved_bytes=0, solver_settings=settings("symmetric"))
    b["shared_target"]["dense_sum"] += 1e-12
    rows = module().summarize_receipts([save(tmp_path, a), save(tmp_path, b, "b.json")])
    assert len(rows) == 2
    assert all(row["target_summary_group_size"] == 2 for row in rows)


@pytest.mark.parametrize("field", ["control_sum", "dense_sum", "control_l2", "dense_l2"])
def test_cross_host_target_summary_mismatch_rejected(tmp_path, field):
    a = receipt()
    b = copy.deepcopy(a)
    b["environment"]["hostname"] = "other"
    b["shared_target"][field] += .01
    with pytest.raises(ValueError, match="target summary"):
        module().summarize_receipts([save(tmp_path, a), save(tmp_path, b, "b.json")])


@pytest.mark.parametrize("path,value", [
    (("schema",), "other"), (("runs", 0, "status"), "ok"),
    (("configuration", "steps"), True), (("configuration", "learning_rate"), 0),
    (("configuration", "control_vertex_count"), 10), (("configuration", "face_count"), 9),
    (("configuration", "dtype"), "float16"), (("configuration", "device"), "tpu"),
    (("configuration", "backends"), ["direct", "direct"]),
    (("environment", "device"), "cuda"), (("environment", "commit"), "unknown"),
    (("shared_target", "generated_once"), False), (("shared_target", "dense_shape"), [8, 9, 2]),
    (("runs", 0, "target_object_identity"), 999), (("runs", 0, "task"), "image_registration"),
    (("runs", 0, "backend"), "PretendDirect"), (("runs", 0, "global_solves"), 4),
    (("runs", 0, "best_objective"), .3), (("runs", 0, "final_objective"), None),
    (("runs", 0, "final_metrics", "map_rmse"), None),
    (("runs", 0, "final_metrics", "minimum_signed_area"), .4),
    (("runs", 0, "trace", 1, "iteration"), 4),
    (("runs", 0, "trace", 1, "global_solves"), 5),
    (("runs", 0, "trace", 1, "objective"), .9),
    (("runs", 0, "trace", 1, "topology_certified"), False),
    (("runs", 0, "trace", 0, "forward_seconds"), -1),
    (("runs", 0, "wall_seconds_to_threshold"), .1),
])
def test_invalid_receipt_rejected(tmp_path, path, value):
    data = receipt()
    part = data
    for key in path[:-1]:
        part = part[key]
    part[path[-1]] = value
    with pytest.raises(ValueError):
        module().summarize_receipts([save(tmp_path, data)])


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_anywhere_rejected(tmp_path, token):
    text = json.dumps(receipt()).replace('"control_sum": 8.5', '"control_sum": ' + token)
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        module().summarize_receipts([path])


def test_duplicate_json_keys_and_duplicate_experiment_rejected(tmp_path):
    path = save(tmp_path, receipt())
    with pytest.raises(ValueError, match="duplicate experiment"):
        module().summarize_receipts([path, path])
    path.write_text(path.read_text().replace('"steps": 1', '"steps": 1, "steps": 2'), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate.*key"):
        module().summarize_receipts([path])


def test_missing_unknown_fields_rejected(tmp_path):
    data = receipt()
    del data["runs"][0]["target_object_identity"]
    with pytest.raises(ValueError):
        module().summarize_receipts([save(tmp_path, data)])
    data = receipt()
    data["configuration"]["stepz"] = 10
    with pytest.raises(ValueError):
        module().summarize_receipts([save(tmp_path, data)])


def test_absent_trace_means_unknown_component_times_not_zero(tmp_path):
    data = receipt()
    del data["runs"][0]["trace"]
    row = module().summarize_receipts([save(tmp_path, data)])[0]
    assert row["trace_available"] is False
    assert row["forward_seconds_total"] is None
    assert row["trace_max_flip_count"] is None


@pytest.mark.parametrize("status", ["failure", "not_applicable"])
def test_failure_rows_keep_only_recorded_information(tmp_path, status):
    data = receipt()
    run = dict(backend_requested="direct", status=status)
    if status == "failure":
        run.update(failure_type="RuntimeError", failure_message="did not converge", wrapper_wall_seconds=.7,
                   solver_settings=data["runs"][0]["solver_settings"])
    else:
        data["configuration"]["device"] = data["environment"]["device"] = "cuda"
        run["reason"] = "CPU-only"
    data["runs"] = [run]
    row = module().summarize_receipts([save(tmp_path, data)])[0]
    assert row["status"] == status
    assert row["final_objective"] is None and row["global_solves"] is None
    assert row["inverse_consistency"] is None and row["final_map_rmse"] is None
    assert row["target_identity_check"] == "not_recorded"
    run["target_object_identity"] = 999
    with pytest.raises(ValueError):
        module().summarize_receipts([save(tmp_path, data)])


def test_cli_csv_and_invalid_input_does_not_replace_output(tmp_path):
    path = save(tmp_path, receipt())
    output = tmp_path / "summary.csv"
    process = subprocess.run([sys.executable, str(SCRIPT), str(path), "--output", str(output)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    original = output.read_text(encoding="utf-8")
    rows = list(csv.DictReader(original.splitlines()))
    assert len(rows) == 1 and rows[0]["inverse_consistency"] == ""
    process = subprocess.run([sys.executable, str(SCRIPT), str(path), str(path), "--output", str(output)], capture_output=True, text=True)
    assert process.returncode != 0
    assert output.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("task,optimizer,device", [("supervised_map", "adam", "cpu"),
    ("image_registration", "lbfgs", "cpu"), ("supervised_map", "adam", "cpu:0")])
def test_real_tiny_runner_receipt(tmp_path, task, optimizer, device):
    output = tmp_path / "real.json"
    runner = SCRIPT.with_name("route1_instance_benchmark.py")
    process = subprocess.run([sys.executable, str(runner), "--task", task,
        "--backends", "direct,symmetric", "--optimizer", optimizer, "--control-side", "3",
        "--image-resolution", "8", "--steps", "1", "--learning-rate", ".05",
        "--dtype", "float64", "--device", device, "--include-trace", "--output", str(output)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    rows = module().summarize_receipts([output])
    assert len(rows) == 2
    assert all(row["status"] == "success" and row["trace_available"] for row in rows)
    assert all(row["global_solves"] == 2*row["primal_solves"]-1 for row in rows)
    if optimizer == "adam":
        assert all(row["global_solves"] == 3 for row in rows)


@pytest.mark.parametrize("change", ["hwm", "cpu_gpu_peak", "threshold_count", "threshold_initial", "time_sum", "missing_backend", "extra_backend"])
def test_additional_inconsistent_reports_rejected(tmp_path, change):
    data = receipt()
    run = data["runs"][0]
    if change == "hwm":
        run["process_hwm_bytes"] = 50
    elif change == "cpu_gpu_peak":
        run["peak_gpu_allocated_bytes"] = 1
    elif change == "threshold_count":
        del run["trace"]
        run["global_solves_to_threshold"] = 2
    elif change == "threshold_initial":
        del run["trace"]
        run["initial_objective"] = .12
    elif change == "time_sum":
        run["trace"][0]["forward_seconds"] = 1.
    elif change == "missing_backend":
        data["configuration"]["backends"].append("symmetric")
    else:
        data["runs"].append(copy.deepcopy(run))
    with pytest.raises(ValueError):
        module().summarize_receipts([save(tmp_path, data)])


def test_distinct_target_strengths_not_confused_and_empty_inputs_rejected(tmp_path):
    a = receipt()
    b = copy.deepcopy(a)
    b["configuration"]["target_strength"] = .5
    b["shared_target"]["dense_sum"] += .1
    assert len(module().summarize_receipts([save(tmp_path, a), save(tmp_path, b, "b.json")])) == 2
    with pytest.raises(ValueError):
        module().summarize_receipts([])


def test_cli_cannot_overwrite_input_receipt(tmp_path):
    path = save(tmp_path, receipt())
    original = path.read_bytes()
    process = subprocess.run([sys.executable, str(SCRIPT), str(path), "--output", str(path)], capture_output=True, text=True)
    assert process.returncode != 0
    assert path.read_bytes() == original


@pytest.mark.parametrize("key,value", [("peak_gpu_allocated_bytes", None),
    ("peak_gpu_allocated_bytes", 201), ("gpu_baseline_allocated_bytes", 101),
    ("gpu_baseline_reserved_bytes", 201)])
def test_gpu_peak_baseline_consistency(tmp_path, key, value):
    data = receipt()
    data["configuration"].update(device="cuda", backends=["symmetric"])
    data["environment"]["device"] = "cuda"
    data["runs"][0].update(backend_requested="symmetric", backend="MatrixFreeSymmetricTutteLayer",
        peak_gpu_allocated_bytes=100, peak_gpu_reserved_bytes=200,
        gpu_baseline_allocated_bytes=0, gpu_baseline_reserved_bytes=0, solver_settings=settings("symmetric"))
    data["runs"][0][key] = value
    with pytest.raises(ValueError):
        module().summarize_receipts([save(tmp_path, data)])


@pytest.mark.parametrize("device", ["cpu:0", "cpu:1"])
def test_indexed_cpu_device_uses_cpu_semantics(tmp_path, device):
    data = receipt()
    data["configuration"]["device"] = data["environment"]["device"] = device
    assert module().summarize_receipts([save(tmp_path, data)])[0]["device"] == device


@pytest.mark.parametrize("failure_point", ["write", "fsync", "replace"])
def test_atomic_output_preserves_existing_on_failure(tmp_path, monkeypatch, failure_point):
    import os
    m = module()
    source = save(tmp_path, receipt())
    output = tmp_path / "summary.csv"
    output.write_bytes(b"PREEXISTING SUMMARY\r\n")
    original = output.read_bytes()
    existing = set(tmp_path.iterdir())
    if failure_point == "write":
        class BrokenWriter:
            def __init__(self, stream, fieldnames):
                self.stream = stream
            def writeheader(self):
                self.stream.write("partial header\n")
            def writerows(self, rows):
                raise OSError("simulated write failure")
        monkeypatch.setattr(m.csv, "DictWriter", BrokenWriter)
    else:
        def fail(*args, **kwargs):
            raise OSError("simulated " + failure_point + " failure")
        monkeypatch.setattr(os, failure_point, fail)
    with pytest.raises(OSError, match="simulated"):
        m.main([str(source), "--output", str(output)])
    assert output.read_bytes() == original
    assert set(tmp_path.iterdir()) == existing


def test_atomic_output_syncs_then_replaces_in_same_directory(tmp_path, monkeypatch):
    import os
    m = module()
    source = save(tmp_path, receipt())
    output = tmp_path / "summary.csv"
    events = []
    fsync, replace = os.fsync, os.replace
    def spy_sync(fd):
        events.append("fsync")
        fsync(fd)
    def spy_replace(src, dst):
        assert Path(src).parent == output.parent
        assert Path(src) != output
        assert Path(src).read_text(encoding="utf-8").startswith("source_receipt,")
        events.append("replace")
        replace(src, dst)
    monkeypatch.setattr(os, "fsync", spy_sync)
    monkeypatch.setattr(os, "replace", spy_replace)
    m.main([str(source), "--output", str(output)])
    assert events == ["fsync", "replace"]
    assert set(tmp_path.iterdir()) == {source, output}


def test_zero_target_strength_rejected(tmp_path):
    data = receipt()
    data["configuration"]["target_strength"] = 0.
    with pytest.raises(ValueError, match="target_strength"):
        module().summarize_receipts([save(tmp_path, data)])


@pytest.mark.parametrize("backend,key,value", [
    ("direct", "internal_arithmetic", "float32"), ("direct", "relative_tolerance", 1e-5),
    ("direct", "absolute_tolerance", 0.), ("direct", "max_iterations", 500),
    ("direct", "variant", "unpreconditioned CG; candidate/periodic true-residual replacement"),
    ("symmetric", "variant", "SuperLU factorization; two-coordinate RHS; reused transpose factor"),
    ("symmetric", "relative_tolerance", None), ("symmetric", "internal_arithmetic", "float32"),
    ("directed_iterative", "variant", "unpreconditioned CG; candidate/periodic true-residual replacement"),
    ("directed_iterative", "internal_arithmetic", "float32"),
    ("directed_iterative", "max_iterations", None), ("directed_iterative", "relative_tolerance", 0.),
])
def test_solver_settings_match_backend_and_dtype(tmp_path, backend, key, value):
    data = receipt()
    data["configuration"]["backends"] = [backend]
    data["runs"][0].update(backend_requested=backend, backend={"direct": "DirectTutteLayer",
        "symmetric": "MatrixFreeSymmetricTutteLayer", "directed_iterative": "MatrixFreeDirectedTutteLayer"}[backend],
        solver_settings=settings(backend))
    data["runs"][0]["solver_settings"][key] = value
    with pytest.raises(ValueError, match="solver"):
        module().summarize_receipts([save(tmp_path, data)])


def test_failure_row_still_checks_solver_semantics(tmp_path):
    data = receipt()
    data["runs"] = [dict(backend_requested="direct", status="failure", failure_type="RuntimeError",
                         failure_message="failed", wrapper_wall_seconds=.1, solver_settings=settings("symmetric"))]
    with pytest.raises(ValueError, match="solver"):
        module().summarize_receipts([save(tmp_path, data)])


@pytest.mark.parametrize("case", ["semantics", "zero_observation", "late_observation", "final_backward", "final_optimizer"])
def test_first_hit_timing_semantics_rejected(tmp_path, case):
    data = receipt()
    run = data["runs"][0]
    if case == "semantics":
        run["threshold_semantics"] = "last unaudited evaluation"
    elif case == "zero_observation":
        run["trace"][0]["observation_wall_seconds"] = 0.
    elif case == "late_observation":
        run["trace"][1]["observation_wall_seconds"] = run["wall_seconds_to_threshold"] = .2
    elif case == "final_backward":
        run["trace"][-1]["backward_seconds"] = .001
    else:
        run["trace"][-1]["optimizer_step_seconds"] = .001
    with pytest.raises(ValueError):
        module().summarize_receipts([save(tmp_path, data)])


@pytest.mark.parametrize("prefix,sum_value,l2", [("control", 1e100, 1e-100),
    ("dense", 1e100, 1e-100), ("control", 1e-200, 1e-300)])
def test_summary_cauchy_bound_rejected_without_absolute_tolerance_loophole(tmp_path, prefix, sum_value, l2):
    data = receipt()
    data["shared_target"][prefix + "_sum"] = sum_value
    data["shared_target"][prefix + "_l2"] = l2
    with pytest.raises(ValueError, match="Cauchy"):
        module().summarize_receipts([save(tmp_path, data)])


def test_summary_cauchy_equality_and_roundoff_accepted(tmp_path):
    data = receipt()
    data["shared_target"].update(control_sum=18., control_l2=18**.5,
                                 dense_sum=128., dense_l2=128**.5*(1-1e-15))
    assert len(module().summarize_receipts([save(tmp_path, data)])) == 1


@pytest.mark.parametrize("before,after,hwm", [(700149760, 695119872, 699985920),
                                           (706695168, 701005824, 705417216)])
def test_linux_cross_snapshot_rss_can_exceed_later_approximate_hwm(tmp_path, before, after, hwm):
    # Real formal receipts, not synthetic relaxation of an exact invariant:
    # Linux /proc RSS accounting is asynchronous. after/HWM share one read;
    # before and HWM do not. Preserve the observations, do not alter either.
    data = receipt()
    data["runs"][0].update(rss_before_bytes=before, rss_after_bytes=after, process_hwm_bytes=hwm)
    row = module().summarize_receipts([save(tmp_path, data)])[0]
    assert (row["rss_before_bytes"], row["rss_after_bytes"], row["process_hwm_bytes"]) == (before, after, hwm)
