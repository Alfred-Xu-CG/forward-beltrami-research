"""Three predeclared paired F2 floor-reserve runs; no annotation access.

Only floor_safety_fraction changes (historical1 versus explicit.95). Reuse the
all20 image/affine/raw-match recipe, restart each from the same identity residual,
and retain rejected trials/incomplete budgets. No topology/evidence guard changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import numpy as np

from tools.coordinated_lung_all20 import make_configuration as all20_configuration
from tools.coordinated_real_case import optimize
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants


CASES = ("he_to_cc10", "he_to_ki67", "ki67_to_he")
FRACTIONS = (1., .95)


def make_configuration(name, args, fraction, *, production=True):
    if name not in CASES or fraction not in FRACTIONS:
        raise ValueError("only the three predeclared cases and fractions1/.95 are supported")
    fixed, moving = name.split("_to_")
    def image(stain):
        return args.canvas / ("cc10_fixed512.png" if stain == "he" else stain+"_moving512.png")
    pair = dict(name=name, fixed=image(fixed), moving=image(moving),
                affine=args.affines_from/(name+"_affine.npz"))
    configuration = all20_configuration(pair, "f2", args, production=production)
    suffix = "100" if fraction == 1. else "095"
    configuration.output = args.output/(name+"_f2_floor"+suffix+".npz")
    configuration.matches = args.predictions/(name+"_raw_matches.json")
    configuration.f2_floor_safety_fraction = fraction
    return configuration


def _actual_ratio(path):
    with np.load(path, allow_pickle=False) as archive:
        vertices = torch.as_tensor(archive["vertices"].copy(), dtype=torch.float64)
        reference = torch.as_tensor(archive["boundary_reference"].copy(), dtype=torch.float64)
    with torch.no_grad():
        q = q1_corner_determinants(vertices)
        qref = q1_corner_determinants(reference)
        if not bool(torch.isfinite(q).all() and torch.isfinite(qref).all() and (qref > 0).all()):
            raise ValueError("nonfinite saved corners or nonpositive material reference")
        return float((q/qref).min())


def run(args, *, production=True):
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError("new/empty comparison output directory required")
    if args.device not in ("cpu", "cuda") or isinstance(args.threads, bool) or args.threads < 1:
        raise ValueError("cpu/cuda device and positive thread count required")
    manifest_path = args.predictions/"predictions.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("prediction_complete") is not True or manifest.get("annotations_read") is not False:
        raise ValueError("completed label-free frozen prediction manifest required")
    source = manifest.get("rows", [])
    rows = {row["name"]: row for row in source}
    if len(rows) != len(source) or any(name not in rows for name in CASES):
        raise ValueError("unique source cases containing all three predeclared directions required")
    if production and (manifest.get("cohort_size") != 20 or len(rows) != 20):
        raise ValueError("production comparison must use the completed all20 cohort")
    configs = []
    for name in CASES:
        frozen = rows[name].get("raw_matches", {})
        if frozen.get("status") != "ok":
            raise ValueError("successful frozen raw matches required for "+name)
        path = Path(frozen["path"])
        path = path if path.is_absolute() else args.predictions/path
        for fraction in FRACTIONS:
            config = make_configuration(name, args, fraction, production=production)
            config.matches = path
            for input_path in (config.fixed, config.moving, config.affine, config.matches):
                if not input_path.is_file():
                    raise FileNotFoundError(input_path)
            configs.append((name, fraction, config))
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(protocol="predeclared HE→CC10, HE→Ki67, Ki67→HE; ONLY F2 floor reserve1 versus.95",
                  source_manifest=str(manifest_path), annotations_read=False,
                  paired_cases=3, attempts=6, fractions=list(FRACTIONS), rows=[],
                  production_recipe=production,
                  nominal_gradients_per_run=300 if production else 4,
                  initialization="same identity residual and unchanged original positive affine; no teacher/labels",
                  floor_scope="strict real-arithmetic reserve only; existing actual-rounded strict eta checks remain authoritative",
                  timing_scope="paired order1 then.95 within each case in one process; loading/features included in complete call, no cold-start equality claim")
    for name, fraction, config in configs:
        tick = time.perf_counter()
        expected = config.inner_steps*config.cycles*len(config.levels)
        record = dict(name=name, floor_safety_fraction=fraction, output=str(config.output),
                      report=str(config.output.with_suffix(".json")), expected_gradient_steps=expected,
                      frozen_matches=str(config.matches))
        try:
            report = optimize(config)
            if (report["configuration"]["f2_floor_safety_fraction"] != fraction
                    or report["f2_floor_safety_fraction"] != fraction):
                raise ValueError("optimizer report omitted/mismatched requested reserve")
            ratio = _actual_ratio(config.output)
            record.update(initial_total=report["initial"]["total"],
                          final_total=report["final"]["total"],
                          actual_minimum_corner_ratio=ratio,
                          actual_strict_floor_valid=ratio > config.minimum_jacobian,
                          budget_complete=report["gradient_steps"] == expected and report["failed_trials"] == 0,
                          **{key: report.get(key) for key in (
                              "gradient_steps", "failed_trials", "evaluations", "objective_evaluations",
                              "optimize_seconds", "end_to_end_seconds", "loading_seconds", "feature_seconds",
                              "peak_allocated_bytes", "saved_binary_certificate", "selected_stage", "output_selection")})
            if (not record["actual_strict_floor_valid"] or
                    record["saved_binary_certificate"].get("valid") is not True):
                raise ValueError("invalid export: strict actual floor and valid saved binary certificate required")
            record["status"] = "ok"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        record["complete_call_seconds"] = time.perf_counter()-tick
        result["rows"].append(record)
        (args.output/"comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(record, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "affines_from", "predictions", "output"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    result = run(parser.parse_args())
    print(json.dumps(dict(attempts=result["attempts"], successful=sum(row["status"] == "ok" for row in result["rows"]))))


if __name__ == "__main__":
    main()
