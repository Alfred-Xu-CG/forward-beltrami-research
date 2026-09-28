"""Run one image-only DeeperHistReg development registration.

Landmarks are intentionally absent from this entry point.
"""

import argparse
import json
import time
from pathlib import Path

import deeperhistreg
import torch


def configure_device(params: dict, device: str) -> None:
    """Set every DHR stage's device for an isolated CPU/GPU comparison."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    params["device"] = device
    params["initial_registration_params"].update(
        device=device, cuda=(device == "cuda"),
    )
    params["nonrigid_registration_params"]["device"] = device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reuse-matcher", action="store_true",
                        help="isolate SuperPoint/SuperGlue network reuse within this process")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat in one process for cold/warmed cache comparison")
    parser.add_argument("--resample-ratio", type=float, default=0.1,
                        help="same source/target loading scale; use 1 for small original JPEGs")
    parser.add_argument("--case-name", default="HistoReg_CD68_to_CD4_development")
    parser.add_argument("--initial-only", action="store_true",
                        help="run the same feature-based initial stage but disable nonrigid fitting")
    args = parser.parse_args()
    if not 0 < args.resample_ratio <= 1:
        raise ValueError("resample ratio must be in (0,1]")
    if args.repeat < 1:
        raise ValueError("repeat must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    torch.set_num_threads(args.threads)
    matcher_cache = None
    if args.reuse_matcher:
        from tools.digital_dhr_cached_matcher import install_cached_matcher
        matcher_cache = install_cached_matcher()
    params = deeperhistreg.configs.default_initial_nonrigid_fast()
    configure_device(params, args.device)
    params["case_name"] = args.case_name
    params["logging_path"] = str(args.output / "deeperhistreg.log")
    params["loading_params"].update(
        loader="pil", source_resample_ratio=args.resample_ratio,
        target_resample_ratio=args.resample_ratio,
    )
    params["saving_params"]["final_saver"] = "pil"
    params["save_final_images"] = False
    params["save_final_displacement_field"] = True
    if args.initial_only:
        params["run_nonrigid_registration"] = False
    params["preprocessing_params"].update(initial_resolution=768, save_results=False)
    params["initial_registration_params"].update(
        save_results=False
    )
    params["nonrigid_registration_params"].update(
        save_results=False,
        registration_size=512,
        num_levels=5,
        used_levels=5,
        iterations=[30] * 5,
        learning_rates=[0.005, 0.0025, 0.0025, 0.0025, 0.0025],
        alphas=[1.5] * 5,
    )
    for repeat_index in range(args.repeat):
        run_output = args.output if args.repeat == 1 else args.output / f"run_{repeat_index}"
        run_output.mkdir(parents=True, exist_ok=True)
        if args.repeat > 1:
            params["case_name"] = f"{args.case_name}_run_{repeat_index}"
            params["logging_path"] = str(run_output / "deeperhistreg.log")
        (run_output / "config.json").write_text(
            json.dumps(params, indent=2, default=str), encoding="utf-8"
        )
        prior_calls = matcher_cache.calls if matcher_cache is not None else 0
        prior_builds = matcher_cache.model_builds if matcher_cache is not None else 0
        prior_build_seconds = matcher_cache.model_build_seconds if matcher_cache is not None else 0.0
        prior_inference_seconds = matcher_cache.inference_seconds if matcher_cache is not None else 0.0
        start = time.perf_counter()
        pipeline = deeperhistreg.direct_registration.DeeperHistReg_FullResolution(params)
        pipeline.run_registration(str(args.moving), str(args.fixed), str(run_output))
        runtime = {
            "moving": str(args.moving),
            "fixed": str(args.fixed),
            "runtime_seconds": time.perf_counter() - start,
            "registration_seconds": pipeline.total_registration_time,
            "preprocessing_seconds": pipeline.preprocessing_time,
            "initial_seconds": pipeline.initial_registration_time,
            "nonrigid_seconds": getattr(pipeline, "nonrigid_registration_time", 0.0),
            "preprocessed_shape": list(pipeline.pre_source.shape),
            "field_shape": list(pipeline.current_displacement_field.shape),
            "initial_only": args.initial_only,
            "device": args.device,
            "repeat_index": repeat_index,
        }
        if matcher_cache is not None:
            runtime["matcher_cache"] = {
                "calls": matcher_cache.calls - prior_calls,
                "model_builds": matcher_cache.model_builds - prior_builds,
                "model_build_seconds": matcher_cache.model_build_seconds - prior_build_seconds,
                "inference_seconds": matcher_cache.inference_seconds - prior_inference_seconds,
            }
        (run_output / "runtime.json").write_text(
            json.dumps(runtime, indent=2), encoding="utf-8"
        )
        print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
