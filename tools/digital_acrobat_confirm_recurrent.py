"""Blind image-only confirmation of frozen one-pass and recurrent safe maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_acrobat_teacher_probe import _case_images
from tools.digital_mind_dense257_predict import predict as predict_one
from tools.digital_mind_recurrent_predict import predict as predict_recurrent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--one-head", type=Path, required=True)
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--match-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    ids = selection["new_confirmation_ids"]
    if set(ids) & set(selection["combined_train_ids"]):
        raise ValueError("confirmation and training IDs overlap")
    rows = []
    for case in ids:
        fixed, moving = _case_images(args.root, case)
        affine = args.root / f"{case}_directSG_affine.npz"
        if not affine.exists():
            rows.append({"case": case, "status": "missing_image_only_affine"})
            continue
        one = predict_one(args.base, args.one_head, fixed, moving, affine,
                          args.output_dir / f"{case}_onepass_safe257.npz",
                          device_name=args.device, repeats=3)
        rec = predict_recurrent(args.base, args.recurrent, fixed, moving, affine,
                                args.output_dir / f"{case}_recurrent_safe257.npz",
                                device_name=args.device, repeats=3,
                                matches_path=(None if args.match_dir is None else
                                              args.match_dir / f"{case}_alignedSG_matches.npz"))
        rows.append({"case": case, "status": "ok",
                     "one_image": one["dense_256_P1_descriptor_loss"],
                     "recurrent_image": rec["recurrent_256_P1_descriptor_loss"],
                     "one_forward_seconds": one["prewarp_feature_base_dense_seconds_median"],
                     "recurrent_forward_seconds": rec["prewarp_feature_base_recurrent_seconds_median"],
                     "one_vjp_seconds": one[
                         "base_dense_257_P1_forward_and_full_parameter_vjp_seconds_median"],
                     "recurrent_vjp_seconds": rec[
                         "base_recurrent_forward_and_full_parameter_vjp_seconds_median"],
                     "one_finite_vjp": one["finite_full_parameter_vjp"],
                     "recurrent_finite_vjp": rec["finite_full_parameter_vjp"]})
    good = [row for row in rows if row["status"] == "ok"]
    report = {"method": "frozen image-only-affine 8-case ACROBAT confirmation, no labels",
              "trained_model_ids_disjoint": True,
              "case_count": len(ids), "success_count": len(good), "rows": rows,
              "mean_one_image": (sum(row["one_image"] for row in good) / len(good)
                                 if good else None),
              "mean_recurrent_image": (
                  sum(row["recurrent_image"] for row in good) / len(good)
                  if good else None),
              "recurrent_better_count": sum(row["recurrent_image"] < row["one_image"]
                                            for row in good)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "case_count", "success_count", "mean_one_image",
        "mean_recurrent_image", "recurrent_better_count",
    )}))


if __name__ == "__main__":
    main()
