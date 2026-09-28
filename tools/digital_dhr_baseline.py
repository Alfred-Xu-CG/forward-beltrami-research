"""Run one image-only DeeperHistReg development registration.

Landmarks are intentionally absent from this entry point.
"""

import argparse
import json
import time
from pathlib import Path

import deeperhistreg
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    params = deeperhistreg.configs.default_initial_nonrigid_fast()
    params["device"] = "cpu"
    params["case_name"] = "HistoReg_CD68_to_CD4_development"
    params["logging_path"] = str(args.output / "deeperhistreg.log")
    params["loading_params"].update(
        loader="pil", source_resample_ratio=0.1, target_resample_ratio=0.1
    )
    params["saving_params"]["final_saver"] = "pil"
    params["save_final_images"] = False
    params["save_final_displacement_field"] = True
    params["preprocessing_params"].update(initial_resolution=768, save_results=False)
    params["initial_registration_params"].update(
        device="cpu", cuda=False, save_results=False
    )
    params["nonrigid_registration_params"].update(
        device="cpu",
        save_results=False,
        registration_size=512,
        num_levels=5,
        used_levels=5,
        iterations=[30] * 5,
        learning_rates=[0.005, 0.0025, 0.0025, 0.0025, 0.0025],
        alphas=[1.5] * 5,
    )
    (args.output / "config.json").write_text(
        json.dumps(params, indent=2, default=str), encoding="utf-8"
    )
    start = time.perf_counter()
    pipeline = deeperhistreg.direct_registration.DeeperHistReg_FullResolution(params)
    pipeline.run_registration(str(args.moving), str(args.fixed), str(args.output))
    runtime = {
        "moving": str(args.moving),
        "fixed": str(args.fixed),
        "runtime_seconds": time.perf_counter() - start,
        "registration_seconds": pipeline.total_registration_time,
        "preprocessing_seconds": pipeline.preprocessing_time,
        "initial_seconds": pipeline.initial_registration_time,
        "nonrigid_seconds": pipeline.nonrigid_registration_time,
        "preprocessed_shape": list(pipeline.pre_source.shape),
        "field_shape": list(pipeline.current_displacement_field.shape),
    }
    (args.output / "runtime.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
