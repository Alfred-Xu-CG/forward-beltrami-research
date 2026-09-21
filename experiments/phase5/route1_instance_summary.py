"""Validate Route-I instance receipts and export one row per requested backend.

Usage: python experiments/phase5/route1_instance_summary.py receipt*.json --output summary.csv

No solver imports or execution. Torch is used only to parse device names with
the same semantics as the runner, without initializing a device. All inputs are
validated before writing a same-directory temporary CSV, fsync, and atomic
replacement of the destination. Missing measurements remain blank, never zero. A
failure/not_applicable row need not have an object id because the v1 runner does
not record one there; the CSV says not_recorded. Successful rows must match the
receipt's shared target id. Ids are never compared across processes.

Cross-receipt target keys are (task, seed, N, resolution, target_strength), not
optimizer/backend/device/dtype. Shapes match exactly and CPU-float64 target sum
and L2 summaries use rtol=atol=1e-12. This checks summary consistency, NOT full
array equality or a collision-resistant identity. Different target strengths
are distinct experiments. Target setup time may differ across hosts.

Trace component totals sum recorded per-evaluation timings. Wall/wrapper times
are separate enclosing intervals, not additional components. LBFGS traces include
line-search trials; best/threshold refer to audited evaluations, not necessarily
accepted optimizer states. Without trace, component totals and trace topology
extrema are unknown. HWM is process-lifetime RSS peak, not a delta or isolated
backend attribution. Linux RSS counters are approximate/asynchronous, so an
earlier RSS snapshot need not be <= a later reported HWM:
https://www.kernel.org/doc/html/v6.9/filesystems/proc.html
GPU peaks/baselines retain the runner's allocator semantics.
No inverse was computed by this runner: inverse_consistency is always blank.
This validates receipt consistency, not scientific truth or provenance integrity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile


BACKENDS = {"direct": "DirectTutteLayer", "directed_iterative": "MatrixFreeDirectedTutteLayer",
            "symmetric": "MatrixFreeSymmetricTutteLayer"}
CONFIG = "task control_vertices_per_side control_vertex_count face_count image_resolution optimizer steps learning_rate objective_threshold target_strength seed image_name dtype device backends warp_convention".split()
METRICS = "flip_count minimum_signed_area minimum_area_ratio boundary_order_min_gap global_injectivity_certificate map_rmse maximum_map_error mu_rmse maximum_mu_error".split()
ACCURACY = "map_rmse maximum_map_error mu_rmse maximum_mu_error".split()
MEMORY = "rss_before_bytes rss_after_bytes process_hwm_bytes peak_gpu_allocated_bytes peak_gpu_reserved_bytes gpu_baseline_allocated_bytes gpu_baseline_reserved_bytes".split()
COUNTS = "primal_solves adjoint_solves global_solves".split()
THRESHOLD = "evaluations_to_threshold global_solves_to_threshold wall_seconds_to_threshold".split()
THRESHOLD_SEMANTICS = "first independently audited evaluation; LBFGS trials included"
COMPONENTS = {"forward_seconds": "forward_seconds_total", "backward_seconds": "backward_seconds_total",
              "optimizer_step_seconds": "optimizer_seconds_total", "audit_seconds": "audit_seconds_total"}
SUCCESS = ("task backend backend_requested status optimizer_name control_vertices_per_side control_vertex_count image_resolution initial_objective final_objective best_objective target_setup_solves target_setup_seconds wall_seconds wrapper_wall_seconds objective_threshold all_iterates_certified final_metrics target_object_identity threshold_semantics solver_settings".split()
           + MEMORY + COUNTS + THRESHOLD)
TRACE = ("iteration outer_step objective map_rmse observation_wall_seconds flip_count minimum_area_ratio topology_certified".split()
         + list(COMPONENTS) + COUNTS)
SUMMARY = "control_shape dense_shape control_sum dense_sum control_l2 dense_l2".split()
FIELDS = ("source_receipt research_question task N control_vertex_count face_count resolution image optimizer steps lr threshold seed target_strength host device dtype commit backend backend_class status failure_type failure_message reason target_identity_check target_summary_group_size target_setup_seconds target_setup_primal_solves run_target_setup_seconds run_target_setup_solves initial_objective final_objective best_objective".split()
          + COUNTS + THRESHOLD + "wall_seconds wrapper_wall_seconds trace_available trace_evaluations".split()
          + list(COMPONENTS.values()) + ["timed_components_seconds_total"] + MEMORY
          + "all_iterates_certified trace_all_certified trace_max_flip_count trace_minimum_area_ratio inverse_consistency threshold_semantics solver_settings warp_convention memory_semantics".split()
          + ["final_" + key for key in METRICS] + ["target_" + key for key in METRICS]
          + ["target_" + key for key in SUMMARY])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, required, optional=()):
    _require(isinstance(value, dict), "expected an object")
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    _require(not missing and not extra, f"invalid fields: missing={sorted(missing)}, unexpected={sorted(extra)}")


def _number(value, name, *, minimum=None, integer=False, nullable=False):
    if value is None and nullable:
        return
    _require(type(value) is int if integer else type(value) in (int, float), f"{name}: expected {'integer' if integer else 'number'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, f"{name}: must be finite")
    _require(minimum is None or value >= minimum, f"{name}: below minimum {minimum}")


def _string(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name}: nonempty string required")


def _boolean(value, name):
    _require(type(value) is bool, f"{name}: boolean required")


def _finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)
    elif type(value) is float:
        _require(math.isfinite(value), "all JSON numbers must be finite")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _device_type(value):
    """Parse like the runner, not by equality to the literal string 'cpu'."""
    import torch

    _string(value, "device")
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid device: {value}") from exc
    _require(str(device) == value, "device must use the canonical string emitted by torch.device")
    return device.type


def _metrics(value, face_count, *, target=False):
    _keys(value, METRICS)
    _number(value["flip_count"], "flip_count", minimum=0, integer=True)
    _require(value["flip_count"] <= face_count, "flip count exceeds face count")
    _boolean(value["global_injectivity_certificate"], "global_injectivity_certificate")
    for key in ("minimum_signed_area", "minimum_area_ratio", "boundary_order_min_gap"):
        _number(value[key], key)
    _require(value["boundary_order_min_gap"] >= 0, "negative boundary gap")
    _require(_close(value["minimum_signed_area"] * face_count, value["minimum_area_ratio"]),
             "signed area / determinant inconsistent on uniform unit-square mesh")
    _require((value["flip_count"] == 0) == (value["minimum_area_ratio"] > 0), "flip count / minimum determinant inconsistent")
    if value["global_injectivity_certificate"]:
        _require(value["flip_count"] == 0 and value["boundary_order_min_gap"] > 0, "inconsistent topology certificate")
    for key in ACCURACY:
        if target:
            _require(value[key] is None, "shared target has no accuracy reference; expected null")
        else:
            _number(value[key], key, minimum=0)
    if not target:
        _require(value["map_rmse"] <= value["maximum_map_error"] + 1e-12, "map RMSE exceeds maximum")
        _require(value["mu_rmse"] <= value["maximum_mu_error"] + 1e-12, "mu RMSE exceeds maximum")


def _configuration(c):
    _keys(c, CONFIG)
    _require(c["task"] in ("supervised_map", "image_registration"), "invalid task")
    _require(c["optimizer"] in ("adam", "lbfgs"), "invalid optimizer")
    _require(c["dtype"] in ("float32", "float64"), "invalid dtype")
    _device_type(c["device"])
    for key, minimum in (("control_vertices_per_side", 3), ("image_resolution", 2), ("steps", 1),
                         ("control_vertex_count", 1), ("face_count", 1), ("seed", 0)):
        _number(c[key], key, integer=True, minimum=minimum)
    n = c["control_vertices_per_side"]
    _require(c["control_vertex_count"] == n*n and c["face_count"] == 2*(n-1)**2, "invalid mesh sizes")
    _number(c["learning_rate"], "learning_rate", minimum=0)
    _require(c["learning_rate"] > 0, "learning rate must be positive")
    _number(c["target_strength"], "target_strength", minimum=0)
    _require(c["target_strength"] > 0, "target_strength must be positive")
    _number(c["objective_threshold"], "objective_threshold", minimum=0, nullable=True)
    if c["task"] == "supervised_map":
        _require(c["image_name"] is None, "supervised task image must be null")
    else:
        _string(c["image_name"], "image_name")
    _require(c["warp_convention"] == "fixed-to-moving backward coordinates; no inverse is computed", "unexpected warp convention")
    b = c["backends"]
    _require(isinstance(b, list) and b and all(isinstance(x, str) and x in BACKENDS for x in b), "invalid backends")
    _require(len(set(b)) == len(b), "duplicate backend")


def _solve_counts(value):
    for key in COUNTS:
        _number(value[key], key, integer=True, minimum=0)
    _require(value["global_solves"] == value["primal_solves"] + value["adjoint_solves"], "inconsistent solve counts")


def _solver_settings(value, backend, dtype):
    _keys(value, "variant relative_tolerance absolute_tolerance max_iterations internal_arithmetic".split())
    _string(value["variant"], "solver variant")
    _require(value["internal_arithmetic"] in ("float32", "float64"), "invalid solver arithmetic")
    for key in ("relative_tolerance", "absolute_tolerance"):
        _number(value[key], key, minimum=0, nullable=True)
    _number(value["max_iterations"], "max_iterations", integer=True, minimum=1, nullable=True)
    if backend == "direct":
        expected = dict(variant="SuperLU factorization; two-coordinate RHS; reused transpose factor",
                        relative_tolerance=None, absolute_tolerance=None, max_iterations=None,
                        internal_arithmetic="float64")
    else:
        directed = backend == "directed_iterative"
        expected = dict(
            variant=("unpreconditioned BiCGStab; explicit true residual each iteration" if directed
                     else "unpreconditioned CG; candidate/periodic true-residual replacement"),
            relative_tolerance=1e-5 if dtype == "float32" else 1e-10 if directed else 1e-11,
            absolute_tolerance=0., max_iterations=500 if directed else 1000,
            internal_arithmetic=dtype)
    _require(value == expected, "solver settings disagree with backend/dtype and v1 runner semantics")


def _trace(run, c):
    trace = run["trace"]
    _require(isinstance(trace, list) and len(trace) >= 2, "trace must contain initial and final evaluations")
    previous_time = -1.
    previous_outer = -1
    completed_components = 0.
    for index, item in enumerate(trace):
        _keys(item, TRACE)
        for key in ("iteration", "outer_step"):
            _number(item[key], key, integer=True, minimum=0)
        _require(item["iteration"] == index, "trace iteration must be consecutive")
        _require(previous_outer <= item["outer_step"] <= c["steps"], "invalid outer step")
        previous_outer = item["outer_step"]
        for key in list(COMPONENTS) + ["objective", "map_rmse", "observation_wall_seconds"]:
            _number(item[key], key, minimum=0)
        _require(previous_time <= item["observation_wall_seconds"] <= run["wall_seconds"], "nonmonotonic/out-of-range observation time")
        # Observation follows this evaluation's forward+audit but precedes its
        # backward. LBFGS overhead for an entire outer step is attributed to its
        # final closure, so it is only included in subsequent observations.
        lower_bound = math.fsum((completed_components, item["forward_seconds"], item["audit_seconds"]))
        observation = item["observation_wall_seconds"]
        _require(lower_bound == 0 or observation > 0, "positive completed work with zero observation wall time")
        _require(lower_bound <= observation + max(1e-9, observation*1e-9), "completed component times exceed observation wall time")
        completed_components = math.fsum((completed_components, *(item[key] for key in COMPONENTS)))
        previous_time = item["observation_wall_seconds"]
        _solve_counts(item)
        _require(item["primal_solves"] == index+1 and item["adjoint_solves"] == index, "trace solve counts inconsistent with one evaluation per solve")
        _number(item["flip_count"], "flip_count", integer=True, minimum=0)
        _number(item["minimum_area_ratio"], "minimum_area_ratio")
        _boolean(item["topology_certified"], "topology_certified")
        _require(item["flip_count"] <= c["face_count"], "too many trace flips")
        _require((item["flip_count"] == 0) == (item["minimum_area_ratio"] > 0), "trace flips/area inconsistent")
        _require(not item["topology_certified"] or item["flip_count"] == 0, "invalid trace certificate")
    if c["optimizer"] == "adam":
        _require(len(trace) == c["steps"]+1, "Adam trace length mismatch")
        _require(all(item["outer_step"] == i for i, item in enumerate(trace)), "Adam outer step mismatch")
    _require(trace[-1]["outer_step"] == c["steps"], "missing final evaluation")
    _require(trace[-1]["backward_seconds"] == 0 and trace[-1]["optimizer_step_seconds"] == 0,
             "final evaluation has no backward or optimizer step")
    for key in COUNTS:
        _require(trace[-1][key] == run[key], "final trace solve counts mismatch")
    _require(run["initial_objective"] == trace[0]["objective"] and run["final_objective"] == trace[-1]["objective"], "objective endpoints disagree with trace")
    _require(run["best_objective"] == min(x["objective"] for x in trace), "best objective disagrees with trace")
    _require(run["all_iterates_certified"] == all(x["topology_certified"] for x in trace), "all-iterates certificate disagrees with trace")
    for a, b in (("flip_count", "flip_count"), ("minimum_area_ratio", "minimum_area_ratio"),
                 ("topology_certified", "global_injectivity_certificate"), ("map_rmse", "map_rmse")):
        _require(trace[-1][a] == run["final_metrics"][b], "final metrics disagree with final trace")
    hit = next((x for x in trace if c["objective_threshold"] is not None and x["objective"] <= c["objective_threshold"]), None)
    expected = (None, None, None) if hit is None else (hit["iteration"]+1, hit["global_solves"], hit["observation_wall_seconds"])
    _require(tuple(run[key] for key in THRESHOLD) == expected, "threshold does not match first audited hit")
    total = math.fsum(x[key] for x in trace for key in COMPONENTS)
    _require(total <= run["wall_seconds"] + max(1e-9, run["wall_seconds"]*1e-9), "component times exceed enclosing wall time")


def _run(run, c, target):
    _require(isinstance(run, dict), "run must be object")
    status = run.get("status")
    _require(status in ("success", "failure", "not_applicable"), "invalid run status")
    backend = run.get("backend_requested")
    _require(isinstance(backend, str) and backend in c["backends"], "run backend not configured")
    device_type = _device_type(c["device"])
    if status != "success":
        required = (["backend_requested", "status", "failure_type", "failure_message", "wrapper_wall_seconds", "solver_settings"]
                    if status == "failure" else ["backend_requested", "status", "reason"])
        _keys(run, required, ["target_object_identity"])
        if status == "failure":
            _string(run["failure_type"], "failure_type")
            _require(isinstance(run["failure_message"], str), "failure message must be string")
            _number(run["wrapper_wall_seconds"], "wrapper_wall_seconds", minimum=0)
            _solver_settings(run["solver_settings"], backend, c["dtype"])
        else:
            _string(run["reason"], "reason")
            _require(backend == "direct" and device_type != "cpu", "unexpected not_applicable run")
    else:
        _keys(run, SUCCESS, ["trace"])
        _require(not (backend == "direct" and device_type != "cpu"), "direct cannot succeed on non-CPU device")
        _require(run["backend"] == BACKENDS[backend], "backend class mismatch")
        for key in ("task", "control_vertices_per_side", "control_vertex_count", "image_resolution", "objective_threshold"):
            _require(type(run[key]) is type(c[key]) and run[key] == c[key], f"run/config mismatch: {key}")
        _require(run["optimizer_name"] == c["optimizer"], "optimizer mismatch")
        for key in ("initial_objective", "final_objective", "best_objective", "wall_seconds", "wrapper_wall_seconds", "target_setup_seconds"):
            _number(run[key], key, minimum=0)
        _require(run["best_objective"] <= min(run["initial_objective"], run["final_objective"]), "best objective exceeds endpoints")
        _require(run["wrapper_wall_seconds"] >= run["wall_seconds"], "wrapper time below training time")
        _number(run["target_setup_solves"], "target_setup_solves", minimum=0, integer=True)
        _require(run["target_setup_solves"] == 0 and run["target_setup_seconds"] == 0, "shared target must not be regenerated per backend")
        _solve_counts(run)
        _require(run["primal_solves"] == run["adjoint_solves"]+1 and run["adjoint_solves"] >= c["steps"], "invalid training solve counts")
        if c["optimizer"] == "adam":
            _require(run["adjoint_solves"] == c["steps"], "Adam step/solve count mismatch")
        _boolean(run["all_iterates_certified"], "all_iterates_certified")
        _metrics(run["final_metrics"], c["face_count"])
        _require(not run["all_iterates_certified"] or run["final_metrics"]["global_injectivity_certificate"], "final certificate contradicts all-iterates claim")
        for key in MEMORY:
            _number(run[key], key, integer=True, minimum=0, nullable=True)
        if run["process_hwm_bytes"] is not None:
            # Only the after/HWM values come from the same /proc status read.
            # Cross-snapshot monotonicity is not an exact Linux RSS invariant.
            _require(run["rss_after_bytes"] is None or run["rss_after_bytes"] <= run["process_hwm_bytes"],
                     "same-snapshot RSS exceeds process HWM")
        gpu_fields = MEMORY[3:]
        if device_type != "cuda":
            _require(all(run[key] is None for key in gpu_fields), "non-CUDA run must not claim CUDA measurements")
        else:
            _require(all(run[key] is not None for key in gpu_fields), "GPU success missing allocator measurements")
            _require(run["peak_gpu_allocated_bytes"] <= run["peak_gpu_reserved_bytes"] and
                     run["gpu_baseline_allocated_bytes"] <= run["gpu_baseline_reserved_bytes"], "allocated exceeds reserved")
            _require(run["gpu_baseline_allocated_bytes"] <= run["peak_gpu_allocated_bytes"] and
                     run["gpu_baseline_reserved_bytes"] <= run["peak_gpu_reserved_bytes"], "GPU baseline exceeds peak")
        for key in THRESHOLD:
            _number(run[key], key, integer=key != "wall_seconds_to_threshold", minimum=0 if key == "wall_seconds_to_threshold" else 1, nullable=True)
        present = [run[key] is not None for key in THRESHOLD]
        _require(all(present) or not any(present), "partial threshold fields")
        if all(present):
            _require(c["objective_threshold"] is not None and run["best_objective"] <= c["objective_threshold"], "threshold claimed but not reached")
            _require(run[THRESHOLD[0]] <= run["primal_solves"] and run[THRESHOLD[1]] <= run["global_solves"] and run[THRESHOLD[2]] <= run["wall_seconds"], "threshold counts/time out of range")
            _require(run["global_solves_to_threshold"] == 2*run["evaluations_to_threshold"]-1,
                     "threshold solve count inconsistent with audited evaluation")
            _require((run["evaluations_to_threshold"] == 1) == (run["initial_objective"] <= c["objective_threshold"]),
                     "threshold first evaluation disagrees with initial objective")
        else:
            _require(c["objective_threshold"] is None or run["best_objective"] > c["objective_threshold"], "reached threshold missing")
        _require(run["threshold_semantics"] == THRESHOLD_SEMANTICS, "incorrect first-hit threshold_semantics")
        _solver_settings(run["solver_settings"], backend, c["dtype"])
        if "trace" in run:
            _trace(run, c)
    if "target_object_identity" in run:
        _number(run["target_object_identity"], "target_object_identity", integer=True, minimum=1)
        _require(run["target_object_identity"] == target["object_identity"], "run target_object_identity mismatch")


def _validate(data):
    _finite_tree(data)
    _keys(data, "schema research_question configuration environment shared_target runs".split())
    _require(data["schema"] == "phase5_route1_instance_v1", "unsupported schema")
    _string(data["research_question"], "research_question")
    c, env, target = data["configuration"], data["environment"], data["shared_target"]
    _configuration(c)
    _keys(env, "hostname platform python torch numpy scipy commit device cuda_visible_devices torch_threads".split(),
          "cuda_runtime gpu_name gpu_total_bytes".split())
    for key in ("hostname", "platform", "python", "torch", "numpy", "scipy"):
        _string(env[key], key)
    _require(env["device"] == c["device"], "environment/config device mismatch")
    _require(env["commit"] is None or (isinstance(env["commit"], str) and re.fullmatch(r"[a-f0-9]{40}", env["commit"])), "invalid commit")
    _number(env["torch_threads"], "torch_threads", integer=True, minimum=1)
    _require(env["cuda_visible_devices"] is None or isinstance(env["cuda_visible_devices"], str), "invalid cuda_visible_devices")
    for key in ("cuda_runtime", "gpu_name"):
        if key in env:
            _string(env[key], key)
    if "gpu_total_bytes" in env:
        _number(env["gpu_total_bytes"], "gpu_total_bytes", integer=True, minimum=1)
    _keys(target, "generated_once setup_primal_solves setup_seconds metrics object_identity".split() + SUMMARY)
    _require(target["generated_once"] is True, "target must be generated_once")
    _number(target["setup_primal_solves"], "setup_primal_solves", integer=True, minimum=1)
    _require(target["setup_primal_solves"] == 1, "target requires one setup solve")
    _number(target["setup_seconds"], "setup_seconds", minimum=0)
    _number(target["object_identity"], "object_identity", integer=True, minimum=1)
    for key, expected in (("control_shape", [c["control_vertex_count"], 2]),
                          ("dense_shape", [c["image_resolution"], c["image_resolution"], 2])):
        _require(isinstance(target[key], list) and all(type(v) is int for v in target[key]) and target[key] == expected, "invalid target summary shape")
    for key in SUMMARY[2:]:
        _number(target[key], key, minimum=0)
        _require(target[key] > 0, "degenerate target summary")
    for prefix in ("control", "dense"):
        # Divide the finite sum rather than squaring or multiplying the norm;
        # this avoids overflow and does not forgive tiny-scale violations via
        # an absolute tolerance. n counts scalar coordinates, not vertices.
        lower_norm = abs(target[prefix + "_sum"]) / math.sqrt(math.prod(target[prefix + "_shape"]))
        norm = target[prefix + "_l2"]
        _require(lower_norm <= norm or math.isclose(lower_norm, norm, rel_tol=1e-12, abs_tol=0.),
                 f"{prefix} target summary violates Cauchy bound")
    _metrics(target["metrics"], c["face_count"], target=True)
    _require(target["metrics"]["global_injectivity_certificate"], "target is not certified")
    _require(isinstance(data["runs"], list), "runs must be list")
    _require(len(data["runs"]) == len(c["backends"]), "missing/extra backend runs")
    seen = set()
    for run in data["runs"]:
        _run(run, c, target)
        _require(run["backend_requested"] not in seen, "duplicate backend run")
        seen.add(run["backend_requested"])


def _flatten(data, run, path):
    c, env, target = data["configuration"], data["environment"], data["shared_target"]
    row = dict.fromkeys(FIELDS)
    row.update(source_receipt=str(path), research_question=data["research_question"], task=c["task"],
               N=c["control_vertices_per_side"], resolution=c["image_resolution"], image=c["image_name"],
               optimizer=c["optimizer"], steps=c["steps"], lr=c["learning_rate"], threshold=c["objective_threshold"],
               control_vertex_count=c["control_vertex_count"], face_count=c["face_count"],
               seed=c["seed"], target_strength=c["target_strength"], host=env["hostname"], device=c["device"],
               dtype=c["dtype"], commit=env["commit"], backend=run["backend_requested"], backend_class=run.get("backend"),
               status=run["status"], target_identity_check="matched" if "target_object_identity" in run else "not_recorded",
               target_setup_seconds=target["setup_seconds"], target_setup_primal_solves=target["setup_primal_solves"],
               run_target_setup_seconds=run.get("target_setup_seconds"), run_target_setup_solves=run.get("target_setup_solves"),
               trace_available="trace" in run, warp_convention=c["warp_convention"],
               memory_semantics="RSS approximate asynchronous snapshots; HWM process lifetime, not cross-snapshot bound; GPU allocator peaks include per-run setup; shared target setup is excluded")
    for key in ("failure_type failure_message reason initial_objective final_objective best_objective wall_seconds wrapper_wall_seconds all_iterates_certified threshold_semantics".split() + COUNTS + THRESHOLD + MEMORY):
        row[key] = run.get(key)
    if "solver_settings" in run:
        row["solver_settings"] = json.dumps(run["solver_settings"], sort_keys=True, separators=(",", ":"))
    for key in METRICS:
        row["final_"+key] = run.get("final_metrics", {}).get(key)
        row["target_"+key] = target["metrics"][key]
    for key in SUMMARY:
        row["target_"+key] = json.dumps(target[key]) if isinstance(target[key], list) else target[key]
    if "trace" in run:
        trace = run["trace"]
        row.update(trace_evaluations=len(trace), trace_all_certified=all(x["topology_certified"] for x in trace),
                   trace_max_flip_count=max(x["flip_count"] for x in trace),
                   trace_minimum_area_ratio=min(x["minimum_area_ratio"] for x in trace))
        for source, destination in COMPONENTS.items():
            row[destination] = math.fsum(x[source] for x in trace)
        row["timed_components_seconds_total"] = math.fsum(row[key] for key in COMPONENTS.values())
    return row


def summarize_receipts(paths):
    """Return fully validated rows, rejecting duplicate experiment identities.

    Duplicate identity includes all configuration except the backend-list plus
    requested backend, host and commit. Status/timings/source path/object id are
    not identity fields: copying or renaming a receipt cannot create a replicate.
    There is no replicate id in v1; same-config reruns must be summarized apart.
    """
    rows, seen, targets, groups = [], set(), {}, {}
    for name in paths:
        path = Path(name).resolve()
        try:
            data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
            _validate(data)
            c, env, target = data["configuration"], data["environment"], data["shared_target"]
            key = tuple(c[x] for x in ("task", "seed", "control_vertices_per_side", "image_resolution", "target_strength"))
            if key in targets:
                previous = targets[key]
                _require(all(target[x] == previous[x] for x in SUMMARY[:2]) and
                         all(_close(target[x], previous[x]) for x in SUMMARY[2:]), "cross-receipt target summary mismatch")
            else:
                targets[key] = target
            groups.setdefault(key, []).append(path)
            for run in data["runs"]:
                identity = tuple(c[x] for x in CONFIG if x != "backends") + (env["hostname"], env["commit"], run["backend_requested"])
                _require(identity not in seen, "duplicate experiment row")
                seen.add(identity)
                row = _flatten(data, run, path)
                rows.append((key, row))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    _require(bool(rows), "at least one receipt is required")
    for key, row in rows:
        row["target_summary_group_size"] = len(groups[key])
    return [row for _, row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _require(args.output.resolve() not in [p.resolve() for p in args.receipts], "output cannot overwrite an input receipt")
        rows = summarize_receipts(args.receipts)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                dir=args.output.parent, prefix=f".{args.output.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"{len(rows)} rows: {args.output}")


if __name__ == "__main__":
    main()
