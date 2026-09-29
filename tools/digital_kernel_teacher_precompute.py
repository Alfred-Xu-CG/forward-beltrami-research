"""Precompute certified kernel maps and compare with slow pseudo-teachers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.digital_kernel_match_safe257 import run
from tools.digital_mind_amortized_network import split_ids


def map_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a.astype(np.float64)
                                     - b.astype(np.float64)) ** 2, axis=-1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "baselines", "matches", "teachers",
                 "output_dir", "report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    ids = json.loads(args.selection.read_text(encoding="utf-8"))["combined_train_ids"]
    train_ids, val_ids = split_ids(ids)
    rows = []
    for case in ids:
        if args.validation_only and case not in val_ids:
            continue
        bpath = args.baselines / f"{case}_old_safe257.npz"
        mpath = args.matches / f"{case}_alignedSG_matches.npz"
        tpath = args.teachers / f"{case}_matchopt4_safe257.npz"
        if not all(p.exists() for p in (bpath, mpath, tpath)):
            continue
        output = args.output_dir / f"{case}_kernelmatch4_safe257.npz"
        result = run(bpath, mpath, output, device_name=args.device)
        with np.load(bpath) as data:
            baseline = data["vertices"]
        with np.load(tpath) as data:
            teacher = data["vertices"]
        with np.load(output) as data:
            kernel = data["vertices"]
        rows.append({"case": case,
                     "split": "train" if case in train_ids else "validation",
                     "baseline_to_teacher_rmse_unit": map_rmse(baseline, teacher),
                     "kernel_to_teacher_rmse_unit": map_rmse(kernel, teacher),
                     "kernel_forward_seconds": result["forward_seconds"],
                     "kernel_full_vjp_seconds": result[
                         "baseline_source_target_vjp_seconds"]})
    summary = {}
    for group in ("train", "validation"):
        selected = [r for r in rows if r["split"] == group]
        summary[group] = {
            "count": len(selected),
            "baseline_to_teacher_rmse_unit": (None if not selected else float(np.mean([
                r["baseline_to_teacher_rmse_unit"] for r in selected]))),
            "kernel_to_teacher_rmse_unit": (None if not selected else float(np.mean([
                r["kernel_to_teacher_rmse_unit"] for r in selected]))),
            "kernel_closer_count": sum(
                r["kernel_to_teacher_rmse_unit"]
                < r["baseline_to_teacher_rmse_unit"] for r in selected),
        }
    report = {"method": "certified kernel maps versus strain-1 slow pseudo-teacher",
              "anatomical_labels_used": False,
              "validation_only": args.validation_only,
              "matched_input_teacher_and_kernel": True,
              "summary": summary, "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
