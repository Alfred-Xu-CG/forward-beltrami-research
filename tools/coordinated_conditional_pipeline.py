"""Measure accepted512 canvases + supplied A -> fresh SG/MA -> retained fusion300.

Not original-image end-to-end timing: rendering and initializer generation are
excluded. Run in an isolated process because the CUDA reset collector temporarily
wraps a process-global API. No source matcher/optimizer algorithm is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.coordinated_match_fusion import configuration, fuse_tables
from tools.coordinated_matchanything_batch import validate_table, validate_export
from tools.coordinated_stain_proxy import source_cases, write_scoring_manifests


class ResetAwarePeaks:
    """Maximum of all reset-delimited segments, including model setup transients."""
    _active = False

    def __init__(self, device, *, cuda_api=None):
        self.device = torch.device(device)
        self.cuda = torch.cuda if cuda_api is None else cuda_api
        self.enabled = self.device.type == "cuda"
        self.allocated = self.reserved = 0
        self.segments = 0
        self.original = None

    def capture(self):
        if self.enabled:
            self.allocated = max(self.allocated, int(self.cuda.max_memory_allocated(self.device)))
            self.reserved = max(self.reserved, int(self.cuda.max_memory_reserved(self.device)))
            self.segments += 1

    def __enter__(self):
        if not self.enabled:
            return self
        if type(self)._active:
            raise RuntimeError("isolated non-nested peak collector required")
        self.cuda.synchronize(self.device)
        self.original = self.cuda.reset_peak_memory_stats
        self.original(self.device)
        def reset(*args, **kwargs):
            self.capture()  # Capture BEFORE the existing routine destroys its peak.
            return self.original(*args, **kwargs)
        self.cuda.reset_peak_memory_stats = reset
        type(self)._active = True
        return self

    def __exit__(self, *exception):
        if self.enabled:
            try:
                self.cuda.synchronize(self.device)
                self.capture()
            finally:
                self.cuda.reset_peak_memory_stats = self.original
                type(self)._active = False

    def report(self):
        return dict(peak_allocated_bytes=self.allocated if self.enabled else None,
                    peak_reserved_bytes=self.reserved if self.enabled else None,
                    captured_segments=self.segments,
                    scope="whole timed process workload; maxima over reset-delimited segments, not sums; PyTorch allocator, not board VRAM")


TABLE_ARRAYS = ("source_points_unit", "target_points_unit", "confidence",
                "post_affine_matrix", "post_affine_offset")


def compare_tables(fresh, archived):
    """Report changes; never align, filter, reorder, or substitute archived rows."""
    result = {}
    for key in TABLE_ARRAYS:
        a, b = np.asarray(fresh[key]), np.asarray(archived[key])
        same_shape = a.shape == b.shape
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        result[key] = dict(fresh_shape=list(a.shape), archived_shape=list(b.shape),
                           exact=bool(same_shape and np.array_equal(a, b)), finite=finite,
                           maximum_absolute_difference=(float(np.max(np.abs(a.astype(float)-b.astype(float))))
                                                        if same_shape and finite and a.size else
                                                        0. if same_shape and finite else None))
    return dict(all_arrays_exact=all(r["exact"] for r in result.values()), arrays=result)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _archive_rows(path, names, *, ma):
    path = Path(path).resolve()
    if path.is_dir():
        path /= "predictions.json"
    value = _read(path)
    rows = value.get("rows", [])
    if (value.get("prediction_complete") is not True or value.get("annotations_read") is not False
            or [r.get("name") for r in rows] != names or any(r.get("status") != "ok" for r in rows)):
        raise ValueError("same ordered25 successful image-only archived rows required")
    paths = []
    for row in rows:
        candidate = Path(row["match_table"] if ma else row["configuration"]["matches"])
        paths.append(candidate if candidate.is_absolute() else path.parent/candidate)
    return paths


def run(miit_control, existing_control, ma_predictions, fusion_predictions, output, source_root,
        checkpoint, *, sg_extractor=None, model_factory=None, optimizer=None,
        export_validator=None, scoring_adapter=None, peak_factory=ResetAwarePeaks):
    # Import production dependencies before timing. Weight construction/loading is timed.
    if sg_extractor is None:
        import deeperhistreg  # noqa: F401
        import superpoint_superglue  # noqa: F401
        from tools.coordinated_image_matches import extract
        sg_extractor = extract
    if model_factory is None:
        from tools.coordinated_matchanything import FrozenMatchAnything
        model_factory = FrozenMatchAnything
    if optimizer is None:
        from tools.coordinated_real_case import optimize
        optimizer = optimize
    export_validator = validate_export if export_validator is None else export_validator
    scoring_adapter = write_scoring_manifests if scoring_adapter is None else scoring_adapter
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = source_cases(miit_control, existing_control)
    names = [r["name"] for r in rows]
    if len(names) != 25 or len(set(names)) != 25:
        raise ValueError("exactly25 distinct cases required")
    ma_paths = _archive_rows(ma_predictions, names, ma=True)
    fused_paths = _archive_rows(fusion_predictions, names, ma=False)
    for row, ma_path, fused_path in zip(rows, ma_paths, fused_paths, strict=True):
        name = row["name"]
        cfg = configuration(row["original_configuration"], "fusion", output/(name+"_analytic.npz"),
                            output/"fused_tables"/(name+".json"))
        row.update(status="pending", sg_status="pending", ma_status="pending", fusion_status="pending",
                   optimization_status="pending", output=cfg.output.name, report=cfg.output.with_suffix(".json").name,
                   configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()},
                   sg_table=str(output/"sg_tables"/(name+".json")),
                   ma_table=str(output/"ma_tables"/(name+".json")),
                   archived_sg_table=row["original_configuration"]["matches"],
                   archived_ma_table=str(ma_path), archived_fused_table=str(fused_path), costs={})
    devices = {r["configuration"]["device"] for r in rows}
    if len(devices) != 1:
        raise ValueError("single original device required")
    device = next(iter(devices))
    cuda = str(device).startswith("cuda")
    for part in ("sg_tables", "ma_tables", "fused_tables"):
        (output/part).mkdir(parents=True)
    manifest = dict(protocol="conditional accepted512 + supplied positive affine to fresh SG+MA fusion300 map",
                    cohort_size=25, specimen_count=4, attempt_denominator=25, rows=rows,
                    prediction_complete=False, extraction_complete=False, annotations_read=False,
                    changed_variables=["fresh SG/MA tables", "output"],
                    timing_scope="whole batch before any model setup through matcher close, fresh fusion, optimization, export/certification and scoring-manifest preparation; includes table comparisons and bookkeeping",
                    excluded_costs=["process/dependency imports before timer", "input-manifest validation/output-directory preparation", "model downloads", "original-image decode/render to accepted512", "supplied initializer estimation", "manual-label scoring"],
                    setup_policy="actual SG model build/load per case; MA one shared batch load; SG phase precedes MA; models released before optimization",
                    batch_schedule="all25 SG calls, one MA setup then all25 MA calls and close, then per-case fusion/optimization/export in order",
                    repeats_scope="one process and one25-case stage-batched workload; per-case component sums are not independent end-to-end cold/warm latencies",
                    sg_setup_attempts=0, ma_setup_attempts=0, ma_extraction_attempts=0,
                    time_to_first_saved_map=None,
                    saved_map_time_scope="elapsed from batch start until optimizer returns and saved-map export/certificate validation completes")
    def sync():
        if cuda:
            torch.cuda.synchronize(device)
    def persist():
        (output/"predictions.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    def fail(row, stage, error):
        row.update(status="failed", **{stage+"_status":"failed"})
        row.setdefault("errors", []).append(dict(stage=stage, error=f"{type(error).__name__}: {error}"))
    def compare(row, stage, fresh, archived_path):
        tick = time.perf_counter()
        try:
            row[stage+"_archive_comparison"] = compare_tables(fresh, _read(archived_path))
        except Exception as error:
            # Archive diagnostics never replace or discard a valid fresh table.
            row[stage+"_archive_comparison"] = dict(error=f"{type(error).__name__}: {error}", all_arrays_exact=False)
        row["costs"][stage+"_archive_comparison_seconds"] = time.perf_counter()-tick
    persist()  # All25 are declared before any model setup/extraction.
    started = time.perf_counter()
    with peak_factory(device) as peaks:
        for row in rows:
            cfg = argparse.Namespace(**row["configuration"])
            tick = time.perf_counter()
            try:
                manifest["sg_setup_attempts"] += 1
                record = sg_extractor(argparse.Namespace(fixed=Path(cfg.fixed),moving=Path(cfg.moving),
                    affine=Path(cfg.affine),output=Path(row["sg_table"]),image_side=512,device=device))
                sync()
                if record.get("status") != "ok":
                    raise ValueError("fresh raw SG extraction insufficient; no archived fallback")
                row.update(sg_status="ok", sg_diagnostics={k:v for k,v in record.items() if k not in TABLE_ARRAYS})
            except Exception as error:
                fail(row, "sg", error)
            row["costs"]["sg_complete_seconds"] = time.perf_counter()-tick
            if row["sg_status"] == "ok":
                compare(row, "sg", record, row["archived_sg_table"])
            persist()
        model = None
        tick = time.perf_counter()
        try:
            manifest["ma_setup_attempts"] += 1
            model = model_factory(Path(source_root), Path(checkpoint), device=device)
            sync()
            manifest.update(ma_setup_status="ok", ma_setup=copy.deepcopy(model.setup_report))
        except Exception as error:
            manifest.update(ma_setup_status="failed", ma_setup_error=f"{type(error).__name__}: {error}")
        manifest["ma_setup_seconds"] = time.perf_counter()-tick
        try:
            for row in rows:
                cfg = argparse.Namespace(**row["configuration"])
                tick = time.perf_counter()
                try:
                    if manifest["ma_setup_status"] != "ok":
                        raise RuntimeError(manifest["ma_setup_error"])
                    manifest["ma_extraction_attempts"] += 1
                    record = model.extract(Path(cfg.fixed), Path(cfg.moving), Path(cfg.affine), output=Path(row["ma_table"]))
                    sync()
                    if record.get("status") != "ok":
                        raise ValueError("fresh MA extraction insufficient; no archived fallback")
                    row.update(ma_status="ok", ma_diagnostics={k:v for k,v in record.items() if k not in TABLE_ARRAYS})
                except Exception as error:
                    fail(row, "ma", error)
                row["costs"]["ma_complete_seconds"] = time.perf_counter()-tick
                if row["ma_status"] == "ok":
                    compare(row, "ma", record, row["archived_ma_table"])
                persist()
        finally:
            tick = time.perf_counter()
            try:
                if model is not None:
                    model.close()
                sync()
                manifest["ma_close_status"] = "ok"
            except Exception as error:
                manifest.update(ma_close_status="failed", ma_close_error=f"{type(error).__name__}: {error}")
                # Do not optimize with a matcher whose release failed.
                for row in rows:
                    fail(row, "ma", RuntimeError("matcher release failed: "+manifest["ma_close_error"]))
            manifest["ma_close_seconds"] = time.perf_counter()-tick
        manifest["extraction_complete"] = True
        persist()
        for row in rows:
            cfg = configuration(row["original_configuration"], "fusion", output/row["output"],
                                Path(row["configuration"]["matches"]))
            if row["sg_status"] != "ok" or row["ma_status"] != "ok":
                row.update(fusion_status="skipped", optimization_status="skipped")
                persist()
                continue
            tick = time.perf_counter()
            try:
                fused = fuse_tables(_read(row["sg_table"]), _read(row["ma_table"]))
                fused.update(original_sg_table=row["sg_table"], original_ma_table=row["ma_table"])
                cfg.matches.write_text(json.dumps(fused, indent=2, allow_nan=False)+"\n", encoding="utf-8")
                row.update(fusion_status="ok", point_loader_metadata=validate_table(cfg))
            except Exception as error:
                fail(row, "fusion", error)
                row["optimization_status"] = "skipped"
            row["costs"]["fusion_complete_seconds"] = time.perf_counter()-tick
            if row["fusion_status"] == "ok":
                compare(row, "fused", fused, row["archived_fused_table"])
                tick = time.perf_counter()
                try:
                    result = optimizer(cfg)
                    sync()
                    ratio = export_validator(cfg, result)
                    row["completed_map_elapsed_from_batch_start"] = time.perf_counter()-started
                    if manifest["time_to_first_saved_map"] is None:
                        manifest["time_to_first_saved_map"] = row["completed_map_elapsed_from_batch_start"]
                    row.update({k:result.get(k) for k in ("gradient_steps","failed_trials","evaluations",
                        "saved_binary_certificate","peak_allocated_bytes","initial","final","loading_seconds",
                        "feature_seconds","serialization_seconds","certification_seconds","optimize_seconds","end_to_end_seconds")})
                    row.update(status="ok", optimization_status="ok", budget_complete=True, actual_minimum_corner_ratio=ratio)
                except Exception as error:
                    fail(row, "optimization", error)
                row["costs"]["optimizer_export_complete_seconds"] = time.perf_counter()-tick
            row["complete_call_seconds"] = sum(row["costs"].values())
            persist()
            print(json.dumps(dict(name=row["name"], status=row["status"])), flush=True)
        for row in rows:
            row["complete_call_seconds"] = sum(row["costs"].values())
        manifest.update(prediction_complete=True, successful=sum(r["status"]=="ok" for r in rows),
                        failed=sum(r["status"]=="failed" for r in rows))
        persist()
        scoring_adapter(manifest, output)
        # The generic MIIT adapter keeps old raw-match provenance; point it at this new fused table.
        miit_path = output/"miit_predictions.json"
        if miit_path.exists():
            value = _read(miit_path)
            for row in value["rows"]:
                row["archived_raw_matches"] = row.get("raw_matches")
                row["raw_matches"] = dict(path=row["methods"]["analytic"]["configuration"]["matches"], arm="fresh_fusion")
            miit_path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        sync()
    manifest.update(whole_batch_seconds=time.perf_counter()-started, whole_batch_memory=peaks.report(),
                    all_tables_exact=all(r.get(stage+"_archive_comparison",{}).get("all_arrays_exact") is True
                                         for r in rows for stage in ("sg","ma","fused")),
                    case_costs_scope="component SUM, not independent end-to-end or response latency; excludes shared MA setup/close and batch bookkeeping",
                    first_case=rows[0]["name"], later_cases=[r["name"] for r in rows[1:]])
    manifest["batch_seconds_per_attempt"] = manifest["whole_batch_seconds"]/25
    manifest["successful_maps_per_batch_second"] = manifest["successful"]/manifest["whole_batch_seconds"]
    persist()  # Final summary write itself is outside the reported batch interval.
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("miit-control", "existing-control", "ma-predictions", "fusion-predictions", "output", "source-root", "checkpoint"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run(**vars(args))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
