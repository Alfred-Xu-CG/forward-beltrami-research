"""Run selected ACROBAT canvases through initial/full DHR on idle assigned GPUs.

Research scratch on the remote host; copy the resulting teacher archives and
report back to D:. This runs no network training and reads no test labels.
Each GPU receives a serial queue, so jobs do not contend on one device.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _run_one(item: dict, root: Path, gpu: int,
             stages: tuple[str, ...]) -> dict:
    case, stain = item["case"], item["stain"]
    fixed = root / f"{case}_HE_physical512.png"
    moving = root / f"{case}_{stain}_physical512.png"
    if not fixed.is_file() or not moving.is_file():
        return {"case": case, "stain": stain, "gpu": gpu,
                "error": "missing rendered canvas"}
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    code_root = Path(__file__).resolve().parent.parent
    environment["PYTHONPATH"] = str(code_root) + os.pathsep + str(code_root / "src")
    record = {"case": case, "stain": stain, "gpu": gpu, "stages": {}}
    for stage in stages:
        target = root / f"{case}_DHR_physical_{stage}_teacher_affine.npz"
        if target.exists():
            with np.load(target) as archive:
                if (archive["raw_teacher_vertices"].shape != (1, 257, 257, 2)
                        or archive["post_affine_matrix"].shape != (2, 2)):
                    raise ValueError(f"invalid existing teacher archive: {target}")
            record["stages"][stage] = {"reused_archive": True}
            continue
        name = f"{case}_scaleup_{stage}"
        output = root / name
        command = [
            sys.executable, "-m", "tools.digital_dhr_baseline",
            "--moving", str(moving), "--fixed", str(fixed),
            "--output", str(output), "--device", "cuda",
            "--threads", "4", "--resample-ratio", "1",
            "--case-name", name,
        ]
        if stage == "initial":
            command.append("--initial-only")
        started = time.perf_counter()
        run = subprocess.run(command, env=environment, text=True,
                             capture_output=True, timeout=600)
        if run.returncode != 0:
            record["error"] = f"{stage} DHR returned {run.returncode}: {run.stderr[-1500:]}"
            return record
        field_dir = output / name / "Results_Final"
        prepare = [
            sys.executable, "-m", "tools.digital_q1_dhr_distill", "prepare",
            "--field", str(field_dir / "displacement_field.mha"),
            "--params", str(field_dir / "postprocessing_params.json"),
            "--moving-image", str(moving), "--fixed-image", str(fixed),
            "--output-target", str(target), "--side", "257", "--affine-factor",
        ]
        conversion = subprocess.run(prepare, env=environment, text=True,
                                    capture_output=True, timeout=180)
        if conversion.returncode != 0:
            record["error"] = (f"{stage} teacher preparation returned "
                               f"{conversion.returncode}: {conversion.stderr[-1500:]}")
            return record
        record["stages"][stage] = {
            "wall_seconds_including_prepare": time.perf_counter() - started,
            "registration": json.loads((output / "runtime.json").read_text()),
            "teacher": json.loads(conversion.stdout),
        }
    return record


def run_selected(selection: Path, root: Path, report_path: Path, *,
                 section: str, devices: list[int],
                 case_ids: list[int] | None = None,
                 available_pending_only: bool = False,
                 stages: tuple[str, ...] = ("initial", "full")) -> dict:
    if report_path.exists():
        raise FileExistsError(report_path)
    if section not in ("new_train", "new_confirmation") or not devices or (
        len(set(devices)) != len(devices) or any(gpu < 0 for gpu in devices)
    ) or not stages or any(stage not in ("initial", "full") for stage in stages):
        raise ValueError("valid section and distinct GPU indices required")
    contents = json.loads(selection.read_text(encoding="utf-8"))
    items = [item for rows in contents[section].values() for item in rows]
    if case_ids is not None:
        if len(set(case_ids)) != len(case_ids) or not set(case_ids) <= {
            item["case"] for item in items
        }:
            raise ValueError("case_ids must be unique selected cases")
        items = [item for item in items if item["case"] in set(case_ids)]
    if available_pending_only:
        items = [item for item in items if (
            (root / f'{item["case"]}_HE_physical512.png').is_file()
            and (root / f'{item["case"]}_{item["stain"]}_physical512.png').is_file()
            and not all((root / f'{item["case"]}_DHR_physical_{stage}_teacher_affine.npz').is_file()
                        for stage in stages)
        )]
    queues = [items[index::len(devices)] for index in range(len(devices))]

    def worker(gpu: int, jobs: list[dict]) -> list[dict]:
        records = []
        for item in jobs:
            try:
                record = _run_one(item, root, gpu, stages)
            except Exception as error:  # Preserve the selected case in denominator.
                record = {"case": item["case"], "stain": item["stain"],
                          "gpu": gpu, "error": repr(error)}
            records.append(record)
            print(json.dumps({"case": record["case"], "gpu": gpu,
                              "ok": "error" not in record,
                              "error": record.get("error")}), flush=True)
        return records

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, gpu, jobs)
                   for gpu, jobs in zip(devices, queues, strict=True)]
        records = [record for future in futures for record in future.result()]
    report = {"section": section, "selected_cases": len(items),
              "successful_cases": sum("error" not in record for record in records),
              "failed_cases": [record["case"] for record in records if "error" in record],
              "devices": devices, "stages": list(stages), "cases": records}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "root", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--section", choices=("new_train", "new_confirmation"),
                        default="new_train")
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--case-ids", type=int, nargs="+")
    parser.add_argument("--available-pending-only", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=("initial", "full"),
                        default=["initial", "full"])
    args = parser.parse_args()
    result = run_selected(args.selection, args.root, args.report,
                          section=args.section, devices=args.devices,
                          case_ids=args.case_ids,
                          available_pending_only=args.available_pending_only,
                          stages=tuple(args.stages))
    print(json.dumps({key: result[key] for key in
                      ("selected_cases", "successful_cases", "failed_cases")}))


if __name__ == "__main__":
    main()
