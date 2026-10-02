"""Released DHR fast/standard pipeline on the three original MIIT image pairs.

No supplied affine, matches, masks or annotations are accepted. Algorithm
settings are preserved from the installed preset; only device and image/output
I/O change. In particular native TIFFs load at ratio 1, then the preset's own
preprocessing chooses its resolution. This is a development reproduction, not
the original RegWSI challenge submission or a shared-objective ablation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

from PIL import Image
import torch


PAIRS = ((2, 3), (7, 8), (10, 11))  # moving/source, fixed/target


def configure(base, *, device, output):
    """Keep every algorithm hyperparameter; adapt only execution/I/O fields."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    result = copy.deepcopy(base)
    result.update(device=device, case_name="released_dhr",
                  logging_path=str(output / "deeperhistreg.log"),
                  save_final_images=False, save_final_displacement_field=True)
    result["loading_params"].update(loader="pil", source_resample_ratio=1., target_resample_ratio=1.)
    result["saving_params"]["final_saver"] = "pil"
    result["preprocessing_params"]["save_results"] = False
    result["initial_registration_params"].update(device=device, cuda=device == "cuda", save_results=False)
    result["nonrigid_registration_params"].update(device=device, save_results=False)
    return result


def _save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def run(args, *, dhr=None):
    if args.preset not in ("fast", "standard") or args.device not in ("cpu", "cuda"):
        raise ValueError("released fast/standard preset and cpu/cuda device required")
    if isinstance(args.threads, bool) or args.threads < 1:
        raise ValueError("positive threads required")
    output, source = Path(args.output).resolve(), Path(args.source_data).resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    import_start = time.perf_counter()
    if dhr is None:
        import deeperhistreg as dhr
    import_seconds = time.perf_counter() - import_start
    preset_name = "default_initial_nonrigid_fast" if args.preset == "fast" else "default_initial_nonrigid"
    base = getattr(dhr.configs, preset_name)()
    try:
        version = importlib.metadata.version("deeperhistreg")
    except importlib.metadata.PackageNotFoundError:
        version = "not available"
    output.mkdir(parents=True)
    _save(output / "released_preset.json", base)
    rows = [dict(name=f"miit_{m}_to_{f}", moving_section=m, fixed_section=f,
                 moving=str(source / str(m) / "images/image.tif"),
                 fixed=str(source / str(f) / "images/image.tif"), status="pending") for m, f in PAIRS]
    report = dict(preset=preset_name, package_version=version,
                  package_source=str(getattr(dhr, "__file__", "unknown")),
                  config_source=str(getattr(dhr.configs, "__file__", "unknown")),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  package_import_seconds=import_seconds, source_data=str(source),
                  prediction_complete=False, annotations_read=False, rows=rows,
                  scope="three development directions from one previously used prostate specimen",
                  initialization="unaltered released native initial-registration stage; no supplied affine",
                  io_adaptation="PIL native TIFF loading ratio1; preset preprocessing resolution unchanged; no image raster exports",
                  map_direction="fixed native pixel centers to moving native pixel centers",
                  timing_scope="complete native pipeline through displacement export; no final whole-slide image rendering; import separate",
                  interpolation="saved MHA displacement sampled bilinearly for landmark evaluation; no hard topology guarantee")
    started = time.perf_counter()
    def persist():
        report["elapsed_seconds"] = time.perf_counter() - started
        _save(output / "predictions.json", report)
    persist()
    for row in rows:
        folder = output / row["name"]
        folder.mkdir()
        params = configure(base, device=args.device, output=folder)
        _save(folder / "config.json", params)
        row["configuration"] = f"{row['name']}/config.json"
        tick = time.perf_counter()
        pipeline = None
        try:
            for key in ("fixed", "moving"):
                with Image.open(row[key]) as image:
                    if image.getexif().get(274, 1) != 1 or image.mode not in ("RGB", "L"):
                        raise ValueError("native L/RGB TIFF with identity orientation required")
                    row[key + "_original_wh"] = list(image.size)
            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            pipeline = dhr.direct_registration.DeeperHistReg_FullResolution(params)
            pipeline.run_registration(row["moving"], row["fixed"], str(folder))
            if args.device == "cuda":
                torch.cuda.synchronize()
            final = folder / "released_dhr/Results_Final"
            for key, name in (("field", "displacement_field.mha"), ("postprocessing_params", "postprocessing_params.json")):
                path = final / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                row[key] = path.relative_to(output).as_posix()
            row.update(status="ok", preprocessed_shape=list(pipeline.pre_source.shape),
                       field_shape=list(pipeline.current_displacement_field.shape),
                       initial_transform=(pipeline.initial_transform.detach().cpu().tolist()
                           if bool(torch.isfinite(pipeline.initial_transform).all()) else None),
                       initial_transform_frame="preprocessed normalized [-1,1] target-to-source affine; align_corners=False",
                       initial_transform_finite=bool(torch.isfinite(pipeline.initial_transform).all()),
                       initial_transform_determinant=(float(torch.linalg.det(pipeline.initial_transform[0, :, :2].double()))
                           if bool(torch.isfinite(pipeline.initial_transform).all()) else None),
                       loaded_padded_shape=list(pipeline.source.shape),
                       preprocessing_seconds=pipeline.preprocessing_time,
                       initial_seconds=pipeline.initial_registration_time,
                       nonrigid_seconds=pipeline.nonrigid_registration_time,
                       registration_seconds=pipeline.total_registration_time,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            row["complete_call_seconds"] = time.perf_counter() - tick
            del pipeline
            persist()
    report.update(prediction_complete=True, successful_pairs=sum(row["status"] == "ok" for row in rows), pair_denominator=3)
    persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("fast", "standard"), default="fast")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    result = run(parser.parse_args())
    print(json.dumps({"preset": result["preset"], "successful_pairs": result["successful_pairs"], "pair_denominator": 3}))


if __name__ == "__main__":
    main()
