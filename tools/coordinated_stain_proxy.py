"""One fixed, contrast-calibrated hematoxylin proxy; no fitted stain vectors.

The scalar is not a measured concentration. Calibration is frozen on each
original canvas before the existing image pyramid and affine descriptor frame.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


STAIN_ROWS = np.array([[.65,.70,.29],[.07,.99,.11],[.27,.57,.78]], dtype=np.float64)


def hematoxylin_proxy(rgb, support):
    """Float RGB in [0,1]; support is the ORIGINAL inverted-gray > .04 mask."""
    rgb = np.asarray(rgb, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or support.shape != rgb.shape[:2]:
        raise ValueError("HWC RGB and matching original support required")
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError("finite RGB in [0,1] required")
    concentrations = -np.log(np.maximum(rgb, 1e-6)) @ np.linalg.inv(STAIN_ROWS)
    h = -np.expm1(-np.maximum(concentrations[..., 0], 0))
    raw_scale = float(np.quantile(h[support], .99, method="linear")) if support.any() else 0.
    scale = max(raw_scale, 1e-6)
    normalized = np.clip(h / scale, 0, 1).astype(np.float32)
    def fraction(values):
        return float(np.mean(values[support])) if support.any() else None
    diagnostics = dict(original_support_pixels=int(support.sum()),
        fractions_scope="each input's own original gray support; reporting only",
        negative_h_fraction=fraction(concentrations[..., 0] < 0),
        rgb_floor_channel_fraction=float(np.mean((rgb < 1e-6)[support])) if support.any() else None,
        rgb_floor_pixel_fraction=fraction(np.any(rgb < 1e-6, axis=-1)),
        h_zero_fraction=fraction(h == 0),
        raw_h_range=[float(h.min()), float(h.max())],
        s_raw=raw_scale, s=scale, numerical_floor_active=raw_scale < 1e-6,
        normalized_saturation_fraction=fraction(normalized >= 1))
    return normalized, diagnostics


def unwarped_mind_diagnostics(feature, support, levels=(32,64,128,256,512)):
    """Static original-mask weighted diagnostics, never overlap selection."""
    from tools.digital_mind_objective_probe import self_similarity
    result = {}
    for side in levels:
        image = F.interpolate(feature, size=(side,side), mode="area")
        weights = F.interpolate(support.float(), size=(side,side), mode="area").flatten()
        with torch.no_grad():
            _, variance = self_similarity(image)
        variance = variance.flatten()
        total = weights.sum()
        if float(total) == 0:
            median = incidence = None
        else:
            median = float(torch.quantile(variance[weights > 0], .5, interpolation="linear"))
            incidence = float((weights*(variance <= 1e-4)).sum()/total)
        result[str(side)] = dict(positive_support_linear_variance_median=median,
            original_mask_weighted_fraction_at_or_below_epsilon=incidence, epsilon=1e-4)
    return result


def load_proxy_image(path, original_gray, expected_side):
    if expected_side != 512:
        raise ValueError("approved stain proxy is calibrated once on the existing 512 canvas")
    with Image.open(path) as image:
        if image.size != (512,512) or image.mode != "RGB":
            raise ValueError("existing 512-square RGB canvas required; no extra resampling")
        rgb = np.asarray(image, dtype=np.uint8).astype(np.float64)/255.
    support = original_gray > .04
    values, diagnostics = hematoxylin_proxy(rgb, support[0,0].cpu().numpy())
    feature = torch.from_numpy(values[None,None])
    diagnostics["unwarped_normalized_mind_by_resolution"] = unwarped_mind_diagnostics(feature, support)
    return feature, diagnostics


def configuration(source, output):
    """Copy a completed shared-affine gray control, changing one evidence choice."""
    values = copy.deepcopy(source)
    required = dict(method="analytic", loss="mind", output_selection="best_full", grid_side=257,
        image_side=512, image_levels=[32,64,128,256,512], levels=[17,33,65,129,257],
        inner_steps=30, cycles=1, strain_weight=3., strain_model="p1_arap", shape_weight=1e-4,
        match_weight=.1, oob_weight=1., minimum_jacobian=.001, precision="float64",
        image_precision="float32", interpolation="p1_ac", mind_frame="shared_affine")
    defaults = dict(preprocessing="raw_inverted", mind_order="transport", image_weight=1.,
        seed_initializer="identity", fixed_mask=None, image_objective="continuation", capture_prefix="none",
        control_hierarchy="fixed", coordinate_mode="alternating", proposal_filter_steps=0, fine_patch_cells=0,
        inner_steps_by_level=[30]*5)
    if any(values.get(k)!=v for k,v in required.items()) or any(values.get(k,v)!=v for k,v in defaults.items()):
        raise ValueError("original shared-affine gray identity-start analytic300 recipe required")
    for key in ("fixed","moving","affine","matches"):
        if not isinstance(values.get(key), str) or not values[key]:
            raise ValueError("original input paths required")
        values[key] = Path(values[key])
    values.update(output=Path(output), preprocessing="hematoxylin_proxy")
    return argparse.Namespace(**values)


def source_cases(miit_control, existing_control):
    """Read only completed control manifests/reports, never label coordinates."""
    from tools.coordinated_dhr_existing_inputs import existing_rows
    expected = {"miit": ["miit_2_to_3","miit_7_to_8","miit_10_to_11"],
                "existing": [r["name"] for r in existing_rows(Path("unused"))]}
    cases = []
    for cohort, directory in (("miit",miit_control),("existing",existing_control)):
        path = Path(directory).resolve()
        if path.is_dir(): path = path/"predictions.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = manifest.get("rows", [])
        if (manifest.get("prediction_complete") is not True or manifest.get("annotations_read") is not False
                or [r.get("name") for r in rows] != expected[cohort]):
            raise ValueError("complete ordered image-only shared controls required")
        for row in rows:
            record = row["methods"]["analytic"] if cohort == "miit" else row
            if record.get("status") != "ok": raise ValueError("successful control required")
            report_path = Path(record["report"])
            if not report_path.is_absolute(): report_path = path.parent/report_path
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("gradient_steps") != 300 or report.get("failed_trials") != 0:
                raise ValueError("completed analytic300 control required")
            configuration(report["configuration"], Path("unused.npz"))
            cases.append(dict(name=row["name"], cohort=cohort, source_manifest=str(path),
                source_report=str(report_path), original_configuration=report["configuration"]))
    return cases


def run(miit_control, existing_control, output):
    from tools.coordinated_real_case import optimize
    from tools.coordinated_lung_all20 import _check_safe_export
    from tools.coordinated_f2_floor_compare import _actual_ratio
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    rows = source_cases(miit_control, existing_control)
    for row in rows:
        cfg = configuration(row["original_configuration"], output/(row["name"]+"_analytic.npz"))
        row.update(status="pending", output=cfg.output.name, report=cfg.output.with_suffix(".json").name,
            configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()})
    manifest = dict(protocol="ONE fixed contrast-calibrated hematoxylin proxy versus shared-affine gray",
        cohort_size=25, specimen_count=4, prediction_complete=False, annotations_read=False,
        changed_variables=["preprocessing","output"], rows=rows,
        scoring_policy="all25 terminal before any manual-label scoring; retain failed denominators",
        objective_caution="H and gray are different image functionals; total energies are not comparable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    def persist():
        manifest["elapsed_seconds"] = time.perf_counter()-started
        (output/"predictions.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    persist()
    for row in rows:
        tick = time.perf_counter()
        try:
            cfg = configuration(row["original_configuration"], output/row["output"])
            for key in ("fixed","moving","affine","matches"):
                if not getattr(cfg,key).is_file(): raise FileNotFoundError(getattr(cfg,key))
            result = optimize(cfg)
            if str(cfg.device).startswith("cuda"): torch.cuda.synchronize()
            with np.load(cfg.affine,allow_pickle=False) as saved:
                a,b = saved["post_affine_matrix"],saved["post_affine_offset"]
            _check_safe_export(cfg.output,result,a,b,cfg.grid_side)
            ratio = _actual_ratio(cfg.output)
            diagnostics = {k:result.get(k) for k in ("gradient_steps","failed_trials","evaluations",
                "objective_evaluations","saved_binary_certificate","peak_allocated_bytes","initial","final",
                "optimize_seconds","end_to_end_seconds","mind_frame","image_preprocessing")}
            json.dumps(diagnostics,allow_nan=False)
            row.update(diagnostics,actual_minimum_corner_ratio=ratio)
            if ratio <= cfg.minimum_jacobian or result["gradient_steps"] != 300 or result["failed_trials"] != 0:
                raise ValueError("strict exported floor and complete300 gradients required")
            row.update(status="ok",budget_complete=True)
        except Exception as error:
            row.update(status="failed",error=f"{type(error).__name__}: {error}")
        row["complete_call_seconds"] = time.perf_counter()-tick
        persist()
        print(json.dumps(dict(name=row["name"],status=row["status"])),flush=True)
    manifest["prediction_complete"] = True
    persist()
    write_scoring_manifests(manifest, output)
    return manifest


def write_scoring_manifests(manifest, output):
    """Adapt existing scorer inputs only after the entire experiment terminates."""
    if (manifest.get("prediction_complete") is not True or len(manifest["rows"]) != 25
            or any(r["status"] not in ("ok","failed") for r in manifest["rows"])):
        raise ValueError("all25 terminal before producing scoring inputs")
    output = Path(output).resolve()
    for cohort in ("miit","existing"):
        rows = [r for r in manifest["rows"] if r["cohort"] == cohort]
        source = Path(rows[0]["source_manifest"])
        result = json.loads(source.read_text(encoding="utf-8"))
        if cohort == "existing":
            result["rows"] = copy.deepcopy(rows)
        else:
            prefix = Path(os.path.relpath(source.parent,output))
            def relocate(value):
                path = Path(value)
                return str(path if path.is_absolute() else prefix/path)
            for old,new in zip(result["rows"],rows,strict=True):
                for key in ("fixed","moving","affine","layout"):
                    old[key] = relocate(old[key])
                for record,key in ((old.get("raw_matches",{}),"path"),(old.get("affine_estimation",{}),"report")):
                    if key in record: record[key] = relocate(record[key])
                for method in ("f2","dhr"):
                    record = old["methods"][method]
                    for key in ("output","report","output_directory","field","configuration","postprocessing_params"):
                        if isinstance(record.get(key),str): record[key] = relocate(record[key])
                    record["pilot_reference"] = "unchanged archived comparison; not rerun"
                old["methods"]["analytic"] = copy.deepcopy(new)
                old["status"] = "ok" if all(v["status"]=="ok" for v in old["methods"].values()) else "partial_failure"
        result.update(protocol=manifest["protocol"],all25_terminal=True,
            source_predictions=str(source),annotations_read=False,prediction_complete=True,
            changed_variables=manifest["changed_variables"])
        (output/(cohort+"_predictions.json")).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("miit-control","existing-control","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    result = run(args.miit_control,args.existing_control,args.output)
    if any(r["status"] != "ok" for r in result["rows"]): raise SystemExit(1)


if __name__ == "__main__": main()
