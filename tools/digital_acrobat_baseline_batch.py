"""Materialize case-split 257-square frozen maps from one declared checkpoint pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.digital_acrobat_teacher_probe import _case_images
from tools.digital_mind_dense257_predict import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "selection", "matches", "base", "one_head", "output_dir"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--partition", type=int, required=True)
    parser.add_argument("--partitions", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.partition < args.partitions:
        raise ValueError("invalid partition")
    ids = json.loads(args.selection.read_text(encoding="utf-8"))["combined_train_ids"]
    valid = []
    for case in ids:
        affine = args.root / f"{case}_directSG_affine.npz"
        match = args.matches / f"{case}_alignedSG_matches.npz"
        if not affine.exists() or not match.exists():
            continue
        with np.load(match) as data:
            if len(data["source_fixed_unit"]) >= 16:
                valid.append(case)
    assigned = [case for i, case in enumerate(valid)
                if i % args.partitions == args.partition]
    rows = []
    for case in assigned:
        fixed, moving = _case_images(args.root, case)
        output = args.output_dir / f"{case}_old_safe257.npz"
        report = predict(args.base, args.one_head, fixed, moving,
                         args.root / f"{case}_directSG_affine.npz", output,
                         device_name=args.device, repeats=1)
        rows.append({"case": case, "map": str(output),
                     "finite_full_parameter_vjp": report["finite_full_parameter_vjp"],
                     "feature_geometry": report["feature_geometry"]})
    report_file = args.output_dir / f"partition_{args.partition}.json"
    if report_file.exists():
        raise FileExistsError(report_file)
    report_file.write_text(json.dumps({
        "method": "case-split image-only directSG frozen safe257 baselines",
        "base_checkpoint": str(args.base), "one_head_checkpoint": str(args.one_head),
        "partition": args.partition, "partitions": args.partitions,
        "available": len(valid), "completed": len(rows), "rows": rows,
        "anatomical_labels_or_full_DHR_field_loaded": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"partition": args.partition,
                      "available": len(valid), "completed": len(rows)}))


if __name__ == "__main__":
    main()
