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
    "primary_slice status comparable failure_reason notes control_side resolution seed "
    "target_strength image_name learning_rate hardware commit target_control_sum "
    "target_control_l2 target_dense_sum target_dense_l2 primary_objective "
    "primary_global_solves primary_wall_seconds primary_forward_seconds "
    "primary_backward_seconds primary_dense_seconds flip_count minimum_signed_area "
    "minimum_area_ratio boundary_order_min_gap topology_certified map_rmse "
    "maximum_map_error mu_rmse maximum_mu_error image_mse threshold_steps "
    "threshold_global_solves threshold_wall_seconds final_objective final_global_solves "
    "final_topology_certified algorithm_wall_seconds wrapper_wall_seconds "
    "cuda_peak_memory_GB_decimal cuda_peak_memory_GiB_binary latent_dimension"
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
        row.update({
            "primary_objective": exact["objective"],
            "primary_global_solves": exact["global_solves"],
            "primary_wall_seconds": exact["observation_wall_seconds"],
            "primary_forward_seconds": exact["primal_seconds"],
            "primary_backward_seconds": exact["adjoint_seconds"] + exact["local_backward_seconds"],
            "primary_dense_seconds": exact["dense_loss_seconds"],
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_csv(extract_route2_authority(args.route2_receipts), args.output)


if __name__ == "__main__":
    main()
