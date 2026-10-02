"""Recompute ONLY F2(.95 floor reserve) for every original all20 direction.

Frozen raw-match JSON, analytic and native DHR exports remain archive references.
No matcher, DHR, analytic optimizer or annotation reader is called here. Copied
historical timings are explicitly excluded from the new runner's elapsed time.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import time

import torch
import numpy as np

from tools.coordinated_lung_all20 import (
    STAINS, METHODS, pairs, make_configuration as original_configuration,
    _check_input, _check_safe_export, _plain,
)
from tools.coordinated_real_case import optimize
from tools.coordinated_f2_floor_compare import _actual_ratio


def make_configuration(pair, args, matches, *, production=True):
    configuration = original_configuration(pair, "f2", args, production=production)
    configuration.matches = matches
    configuration.f2_floor_safety_fraction = .95
    return configuration


def _source(args):
    manifest = args.predictions/"predictions.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if (value.get("prediction_complete") is not True or value.get("annotations_read") is not False
            or value.get("cohort_size") != 20 or not isinstance(value.get("rows"), list)
            or len(value["rows"]) != 20):
        raise ValueError("completed label-free original all20 manifest required")
    for row, (fixed, moving) in zip(value["rows"], itertools.permutations(STAINS, 2), strict=True):
        if (row.get("name"), row.get("fixed_stain"), row.get("moving_stain")) != (
                fixed+"_to_"+moving, fixed, moving):
            raise ValueError("original ordered unique20 cohort required")
        if row.get("input_status") not in ("ok", "failed", "skipped"):
            raise ValueError("original input attempts must be terminal")
        if row.get("raw_matches", {}).get("status") != "ok":
            raise ValueError("successful frozen raw matches required for all20; no evidence replacement")
        methods = row.get("methods", {})
        if set(methods) != set(METHODS) or any(
                methods[name].get("status") not in ("ok", "failed", "skipped") for name in METHODS):
            raise ValueError("all three original method statuses must be terminal")
    return value, manifest


def _artifact(value, source_directory):
    if not isinstance(value, str) or not value:
        raise ValueError("source artifact path required")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (source_directory/path).resolve()


def _relative(path, output):
    return Path(os.path.relpath(path, output)).as_posix()


def _archive_method(method, source_directory, output, origin):
    result = copy.deepcopy(method)
    for key in ("output", "report", "output_directory", "field", "postprocessing_params", "configuration"):
        if isinstance(result.get(key), str):
            result[key] = _relative(_artifact(result[key], source_directory), output)
    result.update(recomputed=False, timing_reused_from_original_run=True,
                  archive_origin_prediction_manifest=origin)
    return result


def run(args, *, production=True, optimizer=None):
    if (isinstance(args.floor_safety_fraction, bool)
            or not isinstance(args.floor_safety_fraction, (int, float)) or args.floor_safety_fraction != .95):
        raise ValueError("predeclared floor_safety_fraction=.95 only; no parameter sweep")
    if args.device not in ("cpu", "cuda") or isinstance(args.threads, bool) or not isinstance(args.threads, int) or args.threads < 1:
        raise ValueError("cpu/cuda and positive integer thread count required")
    args = argparse.Namespace(**vars(args))
    for key in ("canvas", "affines_from", "predictions", "output"):
        setattr(args, key, Path(getattr(args, key)).resolve())
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError("new/empty corrected all20 output directory required")
    started = time.perf_counter()
    source, source_manifest = _source(args)  # No scorer/annotation import.
    origin = _relative(source_manifest, args.output)
    optimizer = optimize if optimizer is None else optimizer
    cohort = pairs(args)
    prepared = []
    rows = []
    for pair, old in zip(cohort, source["rows"], strict=True):
        matches = _artifact(old["raw_matches"]["path"], args.predictions)
        configuration = make_configuration(pair, args, matches, production=production)
        raw = copy.deepcopy(old["raw_matches"])
        raw.update(path=_relative(matches, args.output), recomputed=False,
                   timing_reused_from_original_run=True, archive_origin_prediction_manifest=origin)
        row = dict(**_plain(pair), input_status="pending", status="pending", raw_matches=raw,
                   methods={name: _archive_method(old["methods"][name], args.predictions, args.output, origin)
                            for name in ("analytic", "dhr")})
        row["methods"]["f2"] = dict(status="pending", recomputed=True,
            floor_safety_fraction=.95, output=configuration.output.name,
            report=configuration.output.with_suffix(".json").name, configuration=_plain(vars(configuration)))
        rows.append(row)
        prepared.append((pair, old, matches, configuration, row))
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(protocol="all20 original ordered directions; ONLY F2 recomputed with explicit .95 floor reserve",
        cohort_size=20, stains=list(STAINS), prediction_complete=False, annotations_read=False,
        recomputed_methods=["f2"], archived_methods=["analytic", "dhr"], raw_matches_recomputed=False,
        source_prediction_manifest=origin, floor_safety_fraction=.95,
        old_timing_included_in_new_total=False, canvas=str(args.canvas), affines_from=str(args.affines_from),
        device=args.device, threads=args.threads, started_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
        timing_scope="new elapsed includes current source/input validation, twenty F2 calls and manifest writing; imports excluded. Copied matcher/analytic/DHR timing fields are historical and excluded. No claim that those methods were rerun.",
        memory_scope="new F2 executable peaks only; archived matcher/analytic/DHR peaks remain historical, not a new all-method peak",
        cohort_scope="ONE previously viewed lung-lesion3 specimen; twenty directions not independent patients, labels never loaded here",
        initialization="identity residual, unchanged original positive image-only affine and frozen raw-match records; no field or landmark teacher")
    def persist():
        report["elapsed_seconds"] = time.perf_counter()-started
        (args.output/"predictions.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    persist()
    side = 512 if production else 16
    for pair, old, matches, configuration, row in prepared:
        record = row["methods"]["f2"]
        tick = time.perf_counter()
        expected = configuration.cycles*len(configuration.levels)*configuration.inner_steps
        record["expected_gradient_steps"] = expected
        try:
            # Validate the same image/positive-affine inputs, not old F2 success.
            matrix, offset = _check_input(pair, {**old, "status": old["input_status"]}, side)
            if old["methods"]["analytic"]["status"] == "ok":
                # Actual archived initializer equality, not just a relocated pathname.
                with np.load(_artifact(old["methods"]["analytic"]["output"], args.predictions), allow_pickle=False) as saved:
                    if (not np.array_equal(saved["post_affine_matrix"], matrix)
                            or not np.array_equal(saved["post_affine_offset"], offset)):
                        raise ValueError("supplied affine differs from archived common initializer")
            if not matches.is_file():
                raise FileNotFoundError(matches)
            row["input_status"] = "ok"
            result = optimizer(configuration)
            record.update(**{key: result.get(key) for key in (
                "initial", "final", "gradient_steps", "failed_trials", "evaluations", "objective_evaluations",
                "saved_binary_certificate", "peak_allocated_bytes", "optimize_seconds", "loading_seconds",
                "feature_seconds", "serialization_seconds", "certification_seconds", "end_to_end_seconds")})
            if (result.get("f2_floor_safety_fraction") != .95
                    or result.get("configuration", {}).get("f2_floor_safety_fraction") != .95):
                raise ValueError("new optimizer report must explicitly declare floor reserve.95")
            ratio = _actual_ratio(configuration.output)
            record.update(actual_minimum_corner_ratio=ratio,
                          actual_strict_floor_valid=ratio > configuration.minimum_jacobian)
            _check_safe_export(configuration.output, result, matrix, offset, configuration.grid_side)
            if not record["actual_strict_floor_valid"]:
                raise ValueError("new export must meet the unchanged strict actual eta floor")
            record.update(status="ok", budget_complete=result["gradient_steps"] == expected and result["failed_trials"] == 0)
        except Exception as error:
            if row["input_status"] == "pending":
                row["input_status"] = "failed"
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        record["complete_call_seconds"] = time.perf_counter()-tick
        row["status"] = "ok" if all(m["status"] == "ok" for m in row["methods"].values()) else "partial_failure"
        print(json.dumps(dict(name=row["name"], f2=record["status"],
                             gradients=record.get("gradient_steps"), failed_trials=record.get("failed_trials"))), flush=True)
        persist()
    report.update(prediction_complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
        method_success_counts={name: sum(row["methods"][name]["status"] == "ok" for row in rows) for name in METHODS},
        safe_method_incomplete_budget_counts={"f2": sum(row["methods"]["f2"].get("budget_complete") is False for row in rows)},
        recomputed_f2_complete_call_seconds_total=sum(row["methods"]["f2"]["complete_call_seconds"] for row in rows),
        recomputed_f2_optimize_seconds_total=sum(row["methods"]["f2"].get("optimize_seconds", 0.) or 0. for row in rows))
    persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "affines_from", "predictions", "output"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--floor-safety-fraction", type=float, default=.95,
                        help="fixed predeclared.95 reserve; other values rejected")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("prediction_complete", "method_success_counts", "elapsed_seconds",
                                                 "recomputed_f2_complete_call_seconds_total")}))


if __name__ == "__main__":
    main()
