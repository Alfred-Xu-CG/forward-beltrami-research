"""Build image-only direct-SuperGlue initializers for a listed ACROBAT subset.

No landmark, DHR initial transform, or dense DHR deformation is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_acrobat_teacher_probe import _case_images
from tools.digital_superglue_direct_affine import extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--id-key", choices=("combined_train_ids",
                                              "new_confirmation_ids"),
                        default="combined_train_ids")
    parser.add_argument("--match-threshold", type=float, default=.3)
    parser.add_argument("--retry-failed-from", type=Path)
    args = parser.parse_args()
    ids = json.loads(args.selection.read_text(encoding="utf-8"))[args.id_key]
    if args.retry_failed_from is not None:
        prior = json.loads(args.retry_failed_from.read_text(encoding="utf-8"))
        failed = {int(row["name"]) for row in prior["rows"]
                  if row["status"] != "ok"}
        ids = [case for case in ids if int(case) in failed]
    pairs = []
    for case in ids:
        fixed, moving = _case_images(args.root, int(case))
        pairs.append((str(case), fixed, moving,
                      args.root / f"{case}_directSG_affine.npz"))
    result = extract(pairs, args.report, device_name=args.device,
                     match_threshold=args.match_threshold)
    counts = {}
    for row in result["rows"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"count": len(pairs), "statuses": counts}))


if __name__ == "__main__":
    main()
