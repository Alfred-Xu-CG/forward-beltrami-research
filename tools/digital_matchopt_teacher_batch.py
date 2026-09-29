"""Generate training-only slow match-guided safe-map pseudo-teachers by case partition.

Every case uses the same frozen image-only baseline and the same declared
300-step objective. Partitioning by case ID permits independent GPU workers
without writing to the same output path. No anatomical labels are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.digital_acrobat_teacher_probe import _case_images
from tools.digital_match_guided_pair_optimize import optimize
from tools.digital_mind_dense257_predict import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--one-head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-from", type=Path,
                        help="Reuse a previously frozen map per case without regenerating it")
    parser.add_argument("--partition", type=int, required=True)
    parser.add_argument("--partitions", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--strain-weight", type=float, default=.1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not (0 <= args.partition < args.partitions):
        raise ValueError("invalid partition")
    ids = json.loads(args.selection.read_text(encoding="utf-8"))["combined_train_ids"]
    available = []
    for case in ids:
        match_path = args.matches / f"{case}_alignedSG_matches.npz"
        if not ((args.root / f"{case}_directSG_affine.npz").exists()
                and match_path.exists()):
            continue
        with np.load(match_path) as data:
            if len(data["source_fixed_unit"]) >= 16:
                available.append(case)
    assigned = [case for index, case in enumerate(available)
                if index % args.partitions == args.partition]
    rows = []
    for case in assigned:
        fixed, moving = _case_images(args.root, case)
        affine = args.root / f"{case}_directSG_affine.npz"
        points = args.matches / f"{case}_alignedSG_matches.npz"
        baseline = ((args.baseline_from or (args.output_dir / "baseline"))
                    / f"{case}_old_safe257.npz")
        teacher = args.output_dir / "optimized" / f"{case}_matchopt4_safe257.npz"
        if any(path.exists() for path in (teacher, teacher.with_suffix(".json"))):
            raise FileExistsError(f"case {case} outputs exist; do not overwrite")
        if args.baseline_from is None:
            if baseline.exists() or baseline.with_suffix(".json").exists():
                raise FileExistsError(baseline)
            predict(args.base, args.one_head, fixed, moving, affine, baseline,
                    device_name=args.device, repeats=1)
        elif not baseline.exists():
            raise FileNotFoundError(baseline)
        result = optimize(fixed, moving, baseline, points, teacher,
                          steps=args.steps, passes=4, match_weight=.05,
                          image_weight=1., strain_weight=args.strain_weight,
                          device_name=args.device)
        rows.append({"case": case, "baseline": str(baseline),
                     "teacher": str(teacher),
                     "seconds": result["training_seconds"],
                     "best_image": result["best_image"],
                     "best_match_robust_px": result["best_match_robust_px"]})
    report_path = args.output_dir / f"partition_{args.partition}.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps({
        "method": "offline unlabeled match-guided safe-map pseudo-teachers",
        "partition": args.partition, "partitions": args.partitions,
        "available_count": len(available), "assigned_count": len(assigned),
        "completed_count": len(rows), "rows": rows,
        "strain_weight": args.strain_weight,
        "anatomical_labels_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"available_count": len(available),
                      "assigned_count": len(assigned),
                      "completed_count": len(rows)}))


if __name__ == "__main__":
    main()
