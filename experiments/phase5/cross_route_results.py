"""Build the Phase-V common-input table from authoritative receipts.

Route-II rows are deliberately anchored to the clean ``74b452c`` A6000
receipts and to ``protocol_slices.primary_exact_global_solves`` with budget 83.
The extractor never substitutes the step-40 slice or a later final state.  The
later final state and first-threshold observation are retained in separate CSV
columns.  Equal numeric learning rates are labelled as a robustness cohort,
not as parameterization-fair tuning.

The output is an evidence table, not a ranking.  T1/O1 and T2/O3 are selected
for the eventual common-input comparison.  O4 is preserved as a non-selected
T2 candidate because its map run succeeded but its image run failed before the
83-solve slice.  Missing measurements are written as CSV blanks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


ROUTE2_SCHEMA = "phase5_route2_mvc_instance_benchmark_v1"
ROUTE2_AUTHORITY_COMMIT = "74b452c39c39601bf5d1204288307e9a02a2aaa5"
P1_SCHEMA = "phase5_route3_positive_hodge_instance_v1"
P1_AUTHORITY_COMMIT = "35a1878195d55c3f681fc7ea28b8d3d2b251cb6f"
PREF_SCHEMA = "phase5_route3_pref_common_benchmark_v1"
PREF_AUTHORITY_COMMIT = "0da8ffa7b0816c7dc6fc541fbabf48c37fe65247"
TARGET_RTOL = 1.0e-12
TARGET_ATOL = 1.0e-12
EXACT_SOLVE_BUDGET = 83
METHOD_ROUTES = {
    "O1_sigmoid_positive": "T1",
    "O3_mvc_adam": "T2",
    "O4_covariance_retraction": "T2_candidate",
}
EXPECTED_KEYS = {(task, method) for task in ("map", "image") for method in METHOD_ROUTES}

METRIC_KEYS = (
    "flip_count",
    "minimum_signed_area",
    "minimum_area_ratio",
    "boundary_order_min_gap",
    "map_rmse",
    "maximum_map_error",
    "mu_rmse",
    "maximum_mu_error",
)

FIELDS = (
    "source_receipt route_label method task selected_for_common_input protocol_kind "
    "primary_slice status comparable failure_reason notes primary_timing_scope memory_scope "
    "hard_topology_scope initialization_scope control_side resolution seed "
    "target_strength image_name learning_rate hardware commit target_control_sum "
    "target_control_l2 target_dense_sum target_dense_l2 primary_objective "
    "primary_global_solves primary_wall_seconds primary_forward_seconds "
    "primary_backward_seconds primary_dense_seconds flip_count minimum_signed_area "
    "minimum_area_ratio boundary_order_min_gap topology_certified map_rmse "
    "maximum_map_error mu_rmse maximum_mu_error image_mse threshold_steps "
    "threshold_global_solves threshold_wall_seconds final_objective final_global_solves "
    "final_topology_certified algorithm_wall_seconds wrapper_wall_seconds "
    "cuda_peak_memory_GB_decimal cuda_peak_memory_GiB_binary latent_dimension"
    " process_hwm_GB_decimal explicit_state_MB_decimal gradient_check_relative_error"
).split()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)
    elif isinstance(value, float):
        _require(math.isfinite(value), "all JSON numbers must be finite")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path}: top level must be an object")
    _finite_tree(value)
    return value


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    _require(type(value) in (int, float), f"{name}: finite number required")
    _require(math.isfinite(value), f"{name}: finite number required")
    _require(minimum is None or value >= minimum, f"{name}: must be >= {minimum}")
    return float(value)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name}: integer >= {minimum} required")
    return value


def _close(left: Any, right: Any) -> bool:
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isclose(float(left), float(right), rel_tol=TARGET_RTOL, abs_tol=TARGET_ATOL)
    return left == right


def _identity_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_identity_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_identity_equal(a, b) for a, b in zip(left, right))
    return _close(left, right)


def _shape_text(identity: dict[str, Any], name: str) -> str:
    shape = identity.get("shape")
    _require(isinstance(shape, list) and shape and all(type(v) is int and v > 0 for v in shape),
             f"{name}.shape: positive integer list required")
    return "x".join(str(v) for v in shape)


def _validate_metric_record(metrics: dict[str, Any], *, prefix: str) -> None:
    _require(isinstance(metrics, dict), f"{prefix}: metric object required")
    for key in METRIC_KEYS:
        _require(key in metrics, f"{prefix}: missing {key}")
    flips = _integer(metrics["flip_count"], f"{prefix}.flip_count")
    for key in METRIC_KEYS[1:]:
        _number(metrics[key], f"{prefix}.{key}")
    _require(metrics["map_rmse"] <= metrics["maximum_map_error"] + TARGET_ATOL,
             f"{prefix}: map RMSE exceeds maximum")
    _require(metrics["mu_rmse"] <= metrics["maximum_mu_error"] + TARGET_ATOL,
             f"{prefix}: mu RMSE exceeds maximum")
    if "global_injectivity_certificate" in metrics:
        certificate = metrics["global_injectivity_certificate"]
    else:
        certificate = metrics.get("topology_certified")
    _require(type(certificate) is bool, f"{prefix}: topology certificate required")
    if certificate:
        _require(flips == 0 and metrics["minimum_signed_area"] > 0
                 and metrics["minimum_area_ratio"] > 0
                 and metrics["boundary_order_min_gap"] > 0,
                 f"{prefix}: inconsistent positive topology certificate")


def _validate_configuration(receipt: dict[str, Any], path: Path) -> tuple[str, str]:
    _require(receipt.get("schema") == ROUTE2_SCHEMA, f"{path}: unexpected Route-II schema")
    environment = receipt.get("environment")
    _require(isinstance(environment, dict), f"{path}: missing environment")
    _require(environment.get("dirty") is False, f"{path}: Route-II authority must be clean")
    _require(environment.get("commit") == ROUTE2_AUTHORITY_COMMIT,
             f"{path}: Route-II authority must be commit 74b452c")
    config = receipt.get("configuration")
    _require(isinstance(config, dict), f"{path}: missing configuration")
    expected = {
        "control_side": 25,
        "resolution": 256,
        "seed": 20260922,
        "target_strength": 0.25,
        "image_name": "medical_phantom",
        "learning_rate": 0.001,
        "slice_global_solves": EXACT_SOLVE_BUDGET,
        "steps": 81,
        "device": "cuda",
        "dtype": "float64",
        "backend": "directed_iterative",
        "warp_convention": "backward_map_fixed_to_moving",
    }
    for key, value in expected.items():
        _require(config.get(key) == value, f"{path}: unexpected configuration {key}")
    task = config.get("task")
    _require(task in ("map", "image"), f"{path}: task must be map or image")
    methods = receipt.get("methods")
    _require(isinstance(methods, dict) and len(methods) == 1, f"{path}: one method required")
    method = next(iter(methods))
    _require(method in METHOD_ROUTES, f"{path}: unsupported Route-II method {method}")
    _require(config.get("methods") == [method], f"{path}: method/config mismatch")
    return task, method


def _validate_threshold(method: dict[str, Any]) -> None:
    threshold = method.get("objective_threshold")
    trace = method.get("trace")
    _require(isinstance(trace, list) and trace, "method trace is required")
    first = None if threshold is None else next(
        (row for row in trace if row.get("objective") <= threshold), None
    )
    expected = (None, None, None) if first is None else (
        first["iteration"], first["global_solves"], first["observation_wall_seconds"]
    )
    reported = (
        method.get("accepted_steps_to_threshold"),
        method.get("global_solves_to_threshold"),
        method.get("wall_seconds_to_threshold"),
    )
    _require(all(_close(a, b) for a, b in zip(reported, expected)),
             "reported threshold fields disagree with first qualifying trace row")


def _primary_metric_columns(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {
            key: None
            for key in (
                "flip_count", "minimum_signed_area", "minimum_area_ratio",
                "boundary_order_min_gap", "topology_certified", "map_rmse",
                "maximum_map_error", "mu_rmse", "maximum_mu_error",
            )
        }
    return {
        "flip_count": metrics["flip_count"],
        "minimum_signed_area": metrics["minimum_signed_area"],
        "minimum_area_ratio": metrics["minimum_area_ratio"],
        "boundary_order_min_gap": metrics["boundary_order_min_gap"],
        "topology_certified": metrics.get(
            "global_injectivity_certificate", metrics.get("topology_certified")
        ),
        "map_rmse": metrics["map_rmse"],
        "maximum_map_error": metrics["maximum_map_error"],
        "mu_rmse": metrics["mu_rmse"],
        "maximum_mu_error": metrics["maximum_mu_error"],
    }


def _one_route2_row(path: Path, receipt: dict[str, Any]) -> tuple[tuple[str, str], dict[str, Any], dict[str, Any]]:
    task, method_name = _validate_configuration(receipt, path)
    config = receipt["configuration"]
    environment = receipt["environment"]
    problem = receipt.get("problem_size")
    _require(isinstance(problem, dict), f"{path}: missing problem_size")
    _require(problem.get("control_vertex_count") == 625 and problem.get("face_count") == 1152,
             f"{path}: unexpected mesh size")
    target = receipt.get("shared_target", {}).get("numeric_identity")
    _require(isinstance(target, dict) and isinstance(target.get("control"), dict)
             and isinstance(target.get("dense"), dict), f"{path}: missing target numeric identity")
    for name in ("control", "dense"):
        identity = target[name]
        _shape_text(identity, f"target.{name}")
        for field in ("sum", "l2", "minimum", "maximum"):
            _number(identity.get(field), f"target.{name}.{field}")

    method = receipt["methods"][method_name]
    _require(isinstance(method, dict), f"{path}: invalid method result")
    _validate_threshold(method)
    trace = method["trace"]
    last = trace[-1]
    _require(method.get("completed_steps") == last.get("iteration"), "final iteration disagrees with trace")
    _require(method.get("global_solves") == last.get("global_solves"), "final solve count disagrees with trace")
    _require(_close(method.get("final_objective"), last.get("objective")), "final objective disagrees with trace")
    final_metrics = method.get("final_metrics")
    _validate_metric_record(final_metrics, prefix="final_metrics")
    for key in METRIC_KEYS:
        _require(_close(final_metrics[key], last[key]), f"final metric {key} disagrees with trace")
    _require(final_metrics["global_injectivity_certificate"] == last["topology_certified"],
             "final topology certificate disagrees with trace")

    primary = receipt.get("protocol_slices", {}).get("primary_exact_global_solves")
    _require(isinstance(primary, dict), f"{path}: missing primary exact-solve slice")
    _require(primary.get("budget") == EXACT_SOLVE_BUDGET
             and primary.get("protocol") == "exact_global_solve_primary",
             f"{path}: primary slice is not exactly 83 completed global solves")
    rows = primary.get("rows")
    _require(isinstance(rows, dict), f"{path}: invalid exact-solve rows")
    exact = rows.get(method_name)
    missing_reason = primary.get("missing", {}).get(method_name)
    if exact is not None:
        _require(primary.get("complete") is True, f"{path}: exact slice marked incomplete")
        _require(exact.get("global_solves") == EXACT_SOLVE_BUDGET,
                 f"{path}: primary slice must contain exactly 83 completed global solves")
        matches = [row for row in trace if row.get("global_solves") == EXACT_SOLVE_BUDGET]
        _require(len(matches) == 1 and matches[0] == exact,
                 f"{path}: exact-83 row does not exactly match the unique trace row")
        _validate_metric_record(exact, prefix="exact83")
    else:
        _require(task == "image" and method_name == "O4_covariance_retraction",
                 f"{path}: exact-83 row may be absent only for failed O4 image")
        _require(primary.get("complete") is False and isinstance(missing_reason, str)
                 and "no accepted state at 83 completed global solves" in missing_reason,
                 f"{path}: missing exact-83 row lacks the recorded failure reason")
        exact = None

    failure = method.get("failure")
    if failure is None:
        _require(method.get("status") == "success", f"{path}: success/failure mismatch")
        final_semantics = "completed_final_state"
    else:
        _require(task == "image" and method_name == "O4_covariance_retraction",
                 f"{path}: unexpected method failure")
        _require(method.get("status") == "optimizer_failure", f"{path}: failure status mismatch")
        _require(failure.get("phase") == "retraction_decode" and failure.get("attempted_step") == 18,
                 f"{path}: unexpected O4 image failure")
        final_semantics = "last_certified_accepted_state_before_failure"

    cuda = receipt.get("memory", {}).get("cuda")
    _require(isinstance(cuda, dict) and cuda.get("measurement_status") == "ok",
             f"{path}: valid CUDA memory measurement required")
    peak = _integer(cuda.get("peak_allocated_bytes"), "CUDA peak allocated bytes")
    process_hwm = _integer(
        receipt.get("memory", {}).get("cpu", {}).get("after", {}).get("process_hwm_bytes"),
        "process HWM bytes",
    )
    selected = method_name in ("O1_sigmoid_positive", "O3_mvc_adam")
    route = METHOD_ROUTES[method_name]
    failure_reason = None if failure is None else (
        f"step {failure.get('attempted_step')} {failure.get('phase')}: "
        f"{failure.get('exception_type')}: {failure.get('message')}"
    )
    notes = (
        "selected Route-I baseline; exact83 is primary; final/threshold are supplementary; "
        "shared numeric LR is a robustness cohort, not parameterization-fair tuning; no ranking before P1"
        if method_name == "O1_sigmoid_positive"
        else "selected task-complete MVC method; exact83 is primary; final/threshold are supplementary; "
        "shared numeric LR is a robustness cohort, not parameterization-fair tuning; no ranking before P1"
        if method_name == "O3_mvc_adam"
        else "not selected: O4 image fails before exact83 despite successful O4 map; "
        f"reported final state is {final_semantics}; no ranking before P1"
    )
    row: dict[str, Any] = {
        "source_receipt": path.name,
        "route_label": route,
        "method": method_name,
        "task": task,
        "selected_for_common_input": selected,
        "protocol_kind": "optimized_instance",
        "primary_slice": "exact83" if exact is not None else "exact83_missing",
        "status": method["status"],
        "comparable": bool(exact is not None and selected),
        "failure_reason": failure_reason,
        "notes": notes,
        "primary_timing_scope": (
            "cumulative trace components through exact83; forward is primal-solve time, "
            "backward is adjoint plus local backward, dense is dense loss evaluation"
        ),
        "memory_scope": (
            "fresh-process process-lifetime HWM and CUDA allocator peak including "
            "comparison plus independent final re-decode"
        ),
        "hard_topology_scope": "positive_Tutte_decoder_with_independent_iterate_audits",
        "initialization_scope": "uniform_zero_logits_with_common_fixed_target_boundary",
        "control_side": config["control_side"],
        "resolution": config["resolution"],
        "seed": config["seed"],
        "target_strength": config["target_strength"],
        "image_name": config["image_name"],
        "learning_rate": config["learning_rate"],
        "hardware": (
            f"{environment.get('hostname')} | {environment.get('gpu_name')} | "
            f"{environment.get('device')} | {config['dtype']}"
        ),
        "commit": environment["commit"],
        "target_control_sum": target["control"]["sum"],
        "target_control_l2": target["control"]["l2"],
        "target_dense_sum": target["dense"]["sum"],
        "target_dense_l2": target["dense"]["l2"],
        "threshold_steps": method.get("accepted_steps_to_threshold"),
        "threshold_global_solves": method.get("global_solves_to_threshold"),
        "threshold_wall_seconds": method.get("wall_seconds_to_threshold"),
        "final_global_solves": method["global_solves"],
        "final_objective": method["final_objective"],
        "final_topology_certified": final_metrics["global_injectivity_certificate"],
        # The Route-II receipt reports image MSE as objective, not a separately
        # defined similarity.  Leave the column blank rather than relabel it.
        "image_mse": None,
        "latent_dimension": None,
        "algorithm_wall_seconds": method["algorithm_wall_seconds"],
        "wrapper_wall_seconds": receipt["wrapper_wall_seconds"],
        "cuda_peak_memory_GB_decimal": peak / 1.0e9,
        "cuda_peak_memory_GiB_binary": peak / float(2**30),
        "process_hwm_GB_decimal": process_hwm / 1.0e9,
        "explicit_state_MB_decimal": None,
        "gradient_check_relative_error": None,
    }
    row.update(_primary_metric_columns(exact))
    if exact is None:
        row.update({
            "primary_objective": None,
            "primary_global_solves": None,
            "primary_wall_seconds": None,
            "primary_forward_seconds": None,
            "primary_backward_seconds": None,
            "primary_dense_seconds": None,
        })
    else:
        exact_index = next(
            index
            for index, trace_row in enumerate(trace)
            if trace_row.get("global_solves") == EXACT_SOLVE_BUDGET
        )
        cumulative_rows = trace[: exact_index + 1]
        row.update({
            "primary_objective": exact["objective"],
            "primary_global_solves": exact["global_solves"],
            "primary_wall_seconds": exact["observation_wall_seconds"],
            "primary_forward_seconds": sum(item["primal_seconds"] for item in cumulative_rows),
            "primary_backward_seconds": sum(
                item["adjoint_seconds"] + item["local_backward_seconds"]
                for item in cumulative_rows
            ),
            "primary_dense_seconds": sum(item["dense_loss_seconds"] for item in cumulative_rows),
            "image_mse": exact["objective"] if task == "image" else None,
        })
    _require(set(row) == set(FIELDS),
             f"internal row schema mismatch: missing={sorted(set(FIELDS)-set(row))}, extra={sorted(set(row)-set(FIELDS))}")
    return (task, method_name), target, row


def extract_route2_authority(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Validate and flatten the six clean O1/O3/O4 map+image receipts."""
    records: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        key, target, row = _one_route2_row(path, _load(path))
        _require(key not in records, f"duplicate Route-II task/method receipt: {key}")
        records[key] = (target, row)
    if set(records) != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - set(records))
        extra = sorted(set(records) - EXPECTED_KEYS)
        raise ValueError(f"Route-II authority requires exactly six receipts: missing={missing}, extra={extra}")
    for task in ("map", "image"):
        keys = [(task, method) for method in METHOD_ROUTES]
        authority = records[keys[0]][0]
        for key in keys[1:]:
            _require(_identity_equal(authority, records[key][0]),
                     f"{task} target numeric identity mismatch across Route-II receipts")
    return [records[key][1] for key in sorted(records)]


def _one_p1_row(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    _require(receipt.get("schema") == P1_SCHEMA, f"{path}: unexpected P1 schema")
    _require(receipt.get("status") == "ok", f"{path}: P1 receipt is not successful")
    environment = receipt.get("environment", {})
    _require(environment.get("dirty") is False, f"{path}: P1 authority must be clean")
    _require(environment.get("commit") == P1_AUTHORITY_COMMIT,
             f"{path}: P1 authority must be commit 35a1878")
    config = receipt.get("configuration", {})
    expected = {
        "control_side": 25,
        "resolution": 256,
        "seed": 20260922,
        "strength": 0.25,
        "learning_rate": 0.03,
        "steps": 81,
        "formal_primary_global_solve_slice": 83,
        "formal_final_global_solve_censor": 163,
        "device": "cuda",
        "dtype": "float64",
        "methods": ["P1_positive_uniform"],
        "fixed_target_boundary": True,
    }
    for key, value in expected.items():
        _require(config.get(key) == value, f"{path}: unexpected P1 configuration {key}")
    task = config.get("task")
    _require(task in ("map", "image"), f"{path}: invalid P1 task")
    methods = receipt.get("methods")
    _require(isinstance(methods, list) and len(methods) == 1, f"{path}: one P1 method required")
    method = methods[0]
    _require(method.get("name") == "P1_positive_uniform", f"{path}: wrong P1 method")
    _require(method.get("status") == "success" and method.get("equal_budget_rank_eligible") is True,
             f"{path}: P1 uniform method must be successful and rank eligible")
    _require(method.get("all_iterates_topology_certified") is True,
             f"{path}: every P1 iterate must be certified")
    slices = method.get("protocol_slices", {})
    primary = slices.get("primary_exact_global_solves", {})
    exact = primary.get("row")
    _require(primary.get("missing") is False and isinstance(exact, dict),
             f"{path}: P1 exact83 slice is missing")
    _require(exact.get("completed_global_solves") == EXACT_SOLVE_BUDGET,
             f"{path}: P1 primary slice is not exact83")
    final = slices.get("final_observation", {})
    _require(final.get("completed_global_solves") == 163,
             f"{path}: P1 final slice is not exact163")
    _require(exact.get("topology_certified") is True and final.get("topology_certified") is True,
             f"{path}: P1 protocol slices must be certified")
    target = receipt.get("shared_target", {})
    target_metrics = target.get("metrics", {})
    for name in ("control_sum", "control_l2", "dense_sum", "dense_l2"):
        _number(target.get(name), f"{path}: shared_target.{name}")
    authority = receipt.get("target_authority_audit", {})
    _require(authority.get("matched") is True and authority.get("required") is True,
             f"{path}: P1 target must match Route-II authority")
    final_metrics = method.get("final_metrics")
    _validate_metric_record(final_metrics, prefix=f"{path}: P1 final_metrics")
    independent = method.get("independent_redecode", {})
    _require(independent.get("agreement_passed") is True
             and independent.get("topology_certified") is True,
             f"{path}: P1 independent final re-decode failed")
    cumulative = exact.get("cumulative_timing_at_observation", {})
    for key in (
        "solver_seconds", "backward_seconds", "dense_interpolation_seconds",
        "task_objective_seconds", "audit_seconds",
    ):
        _number(cumulative.get(key), f"{path}: P1 cumulative timing {key}", minimum=0.0)
    method_memory = method.get("memory", {})
    peak = _integer(
        receipt.get("whole_benchmark_memory", {}).get("whole_benchmark_cuda_peak_allocated_bytes"),
        f"{path}: P1 whole-benchmark CUDA peak",
    )
    process_hwm = _integer(method_memory.get("process_hwm_bytes"), f"{path}: P1 process HWM")
    latent = method.get("final_state", {}).get("latent_w")
    _require(isinstance(latent, list) and len(latent) == config.get("faces")
             and all(isinstance(item, list) and len(item) == 2 for item in latent),
             f"{path}: invalid P1 latent state")
    image_name = config.get("image_name")
    _require((task == "map" and image_name is None)
             or (task == "image" and image_name == "medical_phantom"),
             f"{path}: unexpected P1 image name")
    boundary_gap = _number(
        target_metrics.get("boundary_order_min_gap"),
        f"{path}: P1 target boundary gap",
    )
    time_to_useful = method.get("time_to_useful", {})
    row: dict[str, Any] = {
        "source_receipt": path.name,
        "route_label": "P1",
        "method": "P1_positive_uniform",
        "task": task,
        "selected_for_common_input": True,
        "protocol_kind": "optimized_instance",
        "primary_slice": "exact83",
        "status": "success",
        "comparable": True,
        "failure_reason": None,
        "notes": (
            "route-specific LR selected on separate seed; exact83 is primary; frozen "
            "target-independent projector setup is charged in wrapper wall; exact83 does "
            "not serialize maximum map error or minimum signed area"
        ),
        "primary_timing_scope": (
            "cumulative through exact83 observation; solver excludes dense/objective, "
            "backward excludes the post-observation update"
        ),
        "memory_scope": (
            "whole-benchmark CUDA peak includes target/query/projector setup and optimization; "
            "process HWM also includes runtime and independent dense final audit"
        ),
        "hard_topology_scope": "positive_symmetric_Tutte_fixed_graph_and_convex_boundary",
        "initialization_scope": (
            "zero_face_w_hence_mu_zero_then_frozen_projector_target_independent"
        ),
        "control_side": config["control_side"],
        "resolution": config["resolution"],
        "seed": config["seed"],
        "target_strength": config["strength"],
        "image_name": image_name,
        "learning_rate": config["learning_rate"],
        "hardware": (
            f"{environment.get('hostname')} | {environment.get('gpu_name')} | "
            f"{environment.get('device')} | {config['dtype']}"
        ),
        "commit": environment["commit"],
        "target_control_sum": target["control_sum"],
        "target_control_l2": target["control_l2"],
        "target_dense_sum": target["dense_sum"],
        "target_dense_l2": target["dense_l2"],
        "primary_objective": exact["objective"],
        "primary_global_solves": exact["completed_global_solves"],
        "primary_wall_seconds": exact["observation_wall_seconds"],
        "primary_forward_seconds": cumulative["solver_seconds"],
        "primary_backward_seconds": cumulative["backward_seconds"],
        "primary_dense_seconds": cumulative["dense_interpolation_seconds"],
        # The exact83 certificate implies zero flips, but the receipt does not
        # serialize exact83 minimum signed area or maximum map error.
        "flip_count": 0,
        "minimum_signed_area": None,
        "minimum_area_ratio": exact["minimum_area_ratio"],
        "boundary_order_min_gap": boundary_gap,
        "topology_certified": exact["topology_certified"],
        "map_rmse": exact["map_rmse"],
        "maximum_map_error": None,
        "mu_rmse": exact["mu_rmse"],
        "maximum_mu_error": exact["maximum_mu_error"],
        "image_mse": exact["objective"] if task == "image" else None,
        "threshold_steps": method.get("iterations_to_threshold"),
        "threshold_global_solves": method.get("global_solves_to_threshold"),
        "threshold_wall_seconds": method.get("wall_seconds_to_threshold"),
        "final_objective": method["final_objective"],
        "final_global_solves": final["completed_global_solves"],
        "final_topology_certified": final["topology_certified"],
        "algorithm_wall_seconds": method["wall_seconds"],
        "wrapper_wall_seconds": time_to_useful.get("projector_setup_plus_method_wall_seconds"),
        "cuda_peak_memory_GB_decimal": peak / 1.0e9,
        "cuda_peak_memory_GiB_binary": peak / float(2**30),
        "latent_dimension": len(latent) * 2,
        "process_hwm_GB_decimal": process_hwm / 1.0e9,
        "explicit_state_MB_decimal": None,
        "gradient_check_relative_error": None,
    }
    _require(set(row) == set(FIELDS), f"{path}: internal P1 row schema mismatch")
    return row


def _one_pref_row(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    _require(receipt.get("schema") == PREF_SCHEMA, f"{path}: unexpected P-ref schema")
    _require(receipt.get("status") == "ok", f"{path}: P-ref receipt is not successful")
    environment = receipt.get("environment", {})
    _require(environment.get("dirty") is False, f"{path}: P-ref authority must be clean")
    _require(environment.get("commit") == PREF_AUTHORITY_COMMIT,
             f"{path}: P-ref authority must be commit 0da8ffa")
    config = receipt.get("configuration", {})
    expected = {
        "control_side": 25,
        "resolution": 256,
        "seed": 20260922,
        "target_strength": 0.25,
        "device": "cpu",
        "dtype": "float64",
        "threads": 1,
    }
    for key, value in expected.items():
        _require(config.get(key) == value, f"{path}: unexpected P-ref configuration {key}")
    task = receipt.get("task")
    _require(task in ("map", "image") and config.get("task") == task,
             f"{path}: invalid P-ref task")
    _require(receipt.get("route_label") == "P-ref"
             and receipt.get("protocol_kind")
             == "one_shot_fixed_boundary_structure_preserving_reference",
             f"{path}: wrong P-ref protocol labels")
    target = receipt.get("shared_target", {})
    identity = target.get("numeric_identity", {})
    authority = target.get("route2_authority_match", {})
    _require(authority.get("matched") is True and authority.get("required") is True,
             f"{path}: P-ref target must match Route-II authority")
    for name in ("control", "dense"):
        _require(isinstance(identity.get(name), dict), f"{path}: missing P-ref target {name}")
        for field in ("sum", "l2", "minimum", "maximum"):
            _number(identity[name].get(field), f"{path}: P-ref target {name}.{field}")
    metrics = receipt.get("metrics")
    _validate_metric_record(metrics, prefix=f"{path}: P-ref metrics")
    _require(receipt.get("topology_scope")
             == "observed_P1_topology_for_this_target_only_no_general_guarantee",
             f"{path}: unexpected P-ref topology scope")
    counts = receipt.get("solve_counts", {})
    _require(counts.get("primary_total_global_solve_calls_including_adjoint") == 2
             and counts.get("primary_forward_global_solve_calls") == 1
             and counts.get("primary_backward_global_adjoint_calls") == 1,
             f"{path}: unexpected P-ref primary solve counts")
    timings = receipt.get("timings_seconds", {})
    memory = receipt.get("memory", {})
    process_hwm = _integer(
        memory.get("after_primary", {}).get("process_hwm_bytes"),
        f"{path}: P-ref process HWM",
    )
    explicit_state = _integer(
        memory.get("explicit_tensor_and_query_state_bytes"),
        f"{path}: P-ref explicit state bytes",
    )
    vjp = receipt.get("vjp_check", {})
    _require(vjp.get("excluded_from_primary_solve_counts") is True,
             f"{path}: P-ref VJP diagnostic count must be separate")
    problem = receipt.get("problem_size", {})
    _require(problem.get("face_count") == 1152 and problem.get("control_vertex_count") == 625,
             f"{path}: unexpected P-ref problem size")
    objective = receipt.get("primary_objective", {})
    _require(objective.get("optimized") is False, f"{path}: P-ref must not claim optimization")
    row: dict[str, Any] = {
        "source_receipt": path.name,
        "route_label": "P-ref",
        "method": "P-ref_full_Whitney_teacher",
        "task": task,
        "selected_for_common_input": True,
        "protocol_kind": "target_derived_one_shot_reference",
        "primary_slice": "one_forward_one_adjoint",
        "status": "success",
        "comparable": False,
        "failure_reason": None,
        "notes": (
            "same target but target-derived CPU teacher; no optimization or general topology "
            "guarantee; timing, solve budget, threshold, and memory are not rank-comparable"
        ),
        "primary_timing_scope": (
            "one Whitney forward, dense interpolation/task construction, and one diagnostic "
            "adjoint; target setup, validation, and finite differences excluded"
        ),
        "memory_scope": (
            "process-lifetime CPU HWM after primary plus explicitly listed tensor/query state; "
            "no CUDA allocation"
        ),
        "hard_topology_scope": "observed_this_target_only_no_general_guarantee",
        "initialization_scope": "target_derived_exact_facewise_Beltrami_teacher_no_optimization",
        "control_side": config["control_side"],
        "resolution": config["resolution"],
        "seed": config["seed"],
        "target_strength": config["target_strength"],
        "image_name": config.get("image_name"),
        "learning_rate": None,
        "hardware": f"{environment.get('hostname')} | CPU | cpu | {config['dtype']}",
        "commit": environment["commit"],
        "target_control_sum": identity["control"]["sum"],
        "target_control_l2": identity["control"]["l2"],
        "target_dense_sum": identity["dense"]["sum"],
        "target_dense_l2": identity["dense"]["l2"],
        "primary_objective": objective["value"],
        "primary_global_solves": counts["primary_total_global_solve_calls_including_adjoint"],
        "primary_wall_seconds": timings["differentiable_reference_path"],
        "primary_forward_seconds": timings["forward_solve"],
        "primary_backward_seconds": timings["backward"],
        "primary_dense_seconds": (
            timings["dense_interpolation"] + timings["dense_warp_and_image_objective"]
        ),
        "flip_count": metrics["flip_count"],
        "minimum_signed_area": metrics["minimum_signed_area"],
        "minimum_area_ratio": metrics["minimum_area_ratio"],
        "boundary_order_min_gap": metrics["boundary_order_min_gap"],
        "topology_certified": metrics["global_injectivity_certificate"],
        "map_rmse": metrics["map_rmse"],
        "maximum_map_error": metrics["maximum_map_error"],
        "mu_rmse": metrics["mu_rmse"],
        "maximum_mu_error": metrics["maximum_mu_error"],
        "image_mse": metrics["image_mse"] if task == "image" else None,
        "threshold_steps": None,
        "threshold_global_solves": None,
        "threshold_wall_seconds": None,
        "final_objective": objective["value"],
        "final_global_solves": counts["primary_total_global_solve_calls_including_adjoint"],
        "final_topology_certified": metrics["global_injectivity_certificate"],
        "algorithm_wall_seconds": timings["differentiable_reference_path"],
        "wrapper_wall_seconds": timings["total_with_diagnostics"],
        "cuda_peak_memory_GB_decimal": None,
        "cuda_peak_memory_GiB_binary": None,
        "latent_dimension": problem["face_count"] * 2,
        "process_hwm_GB_decimal": process_hwm / 1.0e9,
        "explicit_state_MB_decimal": explicit_state / 1.0e6,
        "gradient_check_relative_error": vjp["relative_error"],
    }
    _require(set(row) == set(FIELDS), f"{path}: internal P-ref row schema mismatch")
    return row


def extract_common_results(
    route2_paths: Iterable[Path],
    p1_paths: Iterable[Path],
    pref_paths: Iterable[Path],
) -> list[dict[str, Any]]:
    """Build the ten-row same-input evidence table without false rankings."""
    route2_rows = extract_route2_authority(route2_paths)
    p1_rows = [_one_p1_row(Path(path), _load(Path(path))) for path in p1_paths]
    pref_rows = [_one_pref_row(Path(path), _load(Path(path))) for path in pref_paths]
    _require({row["task"] for row in p1_rows} == {"map", "image"} and len(p1_rows) == 2,
             "P1 authority requires exactly one map and one image receipt")
    _require({row["task"] for row in pref_rows} == {"map", "image"} and len(pref_rows) == 2,
             "P-ref authority requires exactly one map and one image receipt")
    for task in ("map", "image"):
        authority = next(
            row for row in route2_rows
            if row["task"] == task and row["method"] == "O1_sigmoid_positive"
        )
        for candidate in (
            next(row for row in p1_rows if row["task"] == task),
            next(row for row in pref_rows if row["task"] == task),
        ):
            for key in (
                "target_control_sum", "target_control_l2",
                "target_dense_sum", "target_dense_l2",
            ):
                _require(_close(authority[key], candidate[key]),
                         f"{task}: cross-route target mismatch in {key}")
    return [*route2_rows, *sorted(p1_rows, key=lambda row: row["task"]),
            *sorted(pref_rows, key=lambda row: row["task"])]


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    """Write one compact rebuildable CSV, using blanks for missing values."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route2_receipts", nargs="+", type=Path)
    parser.add_argument("--p1-receipts", nargs=2, type=Path, required=True)
    parser.add_argument("--pref-receipts", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_csv(
        extract_common_results(
            args.route2_receipts,
            args.p1_receipts,
            args.pref_receipts,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
