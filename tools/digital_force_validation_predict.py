"""Run a frozen force-recurrent P1 student on declared ACROBAT cohort IDs.

This is an image-only inference audit, not an anatomy-landmark evaluation. It
saves each predicted binary32 map and its exact Q1 certificate before summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from tools.digital_acrobat_teacher_probe import _case_images
from tools.digital_frozen_residual_predict import predict
from tools.digital_mind_amortized_network import split_ids


def run(root: Path, selection: Path, checkpoint: Path, base: Path, one_head: Path,
        output_dir: Path, report_path: Path, *, device: str, repeats: int,
        cohort: str = "validation", logit_gain: float = 1.,
        matches_dir: Path | None = None,
        tag: str = "force_recurrent_strain02",
        match_feedback_gain: float = 0.) -> dict:
    if report_path.exists():
        raise FileExistsError(report_path)
    saved = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    if not tag or any(character in tag for character in "/\\"):
        raise ValueError("tag must be one filename component")
    requires_matches = (int(saved.get("match_channels", 0)) > 0 and
                        saved.get("evidence_mode") not in (
                            "force_only", "force_recurrent"))
    if requires_matches != (matches_dir is not None):
        raise ValueError("match directory presence disagrees with checkpoint")
    selection_data = json.loads(selection.read_text(encoding="utf-8"))
    selected = selection_data["combined_train_ids"]
    train_ids, val_ids = split_ids(selected)
    if cohort == "train":
        checkpoint_metadata = torch.load(checkpoint, map_location="cpu",
                                         weights_only=False)
        ids = [int(case) for case in checkpoint_metadata["train_ids"]]
        allowed = set(train_ids)
    elif cohort == "validation":
        ids = [int(item["case"]) for item in saved["validation_final"]]
        allowed = set(val_ids)
    elif cohort == "confirmation":
        ids = [int(case) for case in selection_data["new_confirmation_ids"]]
        allowed = set(ids)
        if allowed & set(selected):
            raise ValueError("confirmation IDs overlap all selected training-source IDs")
    else:
        raise ValueError("cohort must be train, validation or confirmation")
    if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(allowed):
        raise ValueError("cohort IDs are not unique or outside the declared split")
    if cohort != "train" and set(ids) & set(train_ids):
        raise ValueError("evaluation cohort intersects model-training IDs")
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in ids:
        fixed, moving = _case_images(root, case)
        affine = root / f"{case}_directSG_affine.npz"
        output = output_dir / f"{case}_{tag}_safe257.npz"
        matches = (None if matches_dir is None else
                   matches_dir / f"{case}_alignedSG_matches.npz")
        result = predict(base, one_head, checkpoint, fixed, moving, affine,
                         matches, output, device_name=device, repeats=repeats,
                         logit_gain=logit_gain,
                         match_feedback_gain=match_feedback_gain)
        cert = result["saved_binary_certificate"]
        with np.load(output) as archive:
            vertices = torch.from_numpy(archive["vertices"].copy())
        corner = q1_corner_determinants(vertices).double() * 256**2
        rows.append({
            "case": case,
            "fixed": str(fixed), "moving": str(moving),
            "map": str(output),
            "frozen_image": result["frozen_image"],
            "student_image": result["residual_image"],
            "student_better_image": result["residual_image"] < result["frozen_image"],
            "forward_seconds": result["full_forward_seconds_median"],
            "forward_vjp_seconds": result[
                "full_forward_and_parameter_match_vjp_seconds_median"],
            "forward_peak_bytes": result["forward_peak_torch_cuda_allocated_bytes"],
            "vjp_peak_bytes": result["vjp_peak_torch_cuda_allocated_bytes"],
            "finite_vjp": result["finite_full_vjp"],
            "match_affine_frame_machine_checked": result[
                "match_affine_frame_machine_checked"],
            "q1_certificate_valid": cert["valid"],
            "q1_min_normalized_corner": float(corner.amin()),
            "q1_corners_below_point1": int((corner < 0.1).sum()),
        })
    report = {
        "method": "frozen recurrent student, ACROBAT image-only inference",
        "cohort": cohort,
        "checkpoint": str(checkpoint), "tag": tag,
        "logit_gain": logit_gain,
        "match_feedback_gain": match_feedback_gain,
        "cohort_id_source": (str(checkpoint) if cohort == "train" else
                             str(checkpoint.with_suffix(".json")) if cohort ==
                             "validation" else str(selection)),
        "selection": str(selection),
        "selection_validation_count": len(val_ids),
        "no_selected_matches_supplied_to_student": matches_dir is None,
        "matches_dir": None if matches_dir is None else str(matches_dir),
        "affine_is_external_image_only_direct_superglue": True,
        "landmarks_or_teacher_maps_read_at_inference": False,
        "case_count": len(rows),
        "student_better_image_count": sum(row["student_better_image"] for row in rows),
        "all_saved_q1_valid": all(row["q1_certificate_valid"] for row in rows),
        "all_finite_vjp": all(row["finite_vjp"] for row in rows),
        "rows": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "selection", "checkpoint", "base", "one_head", "output_dir", "report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path,
                            required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cohort", choices=("train", "validation", "confirmation"),
                        default="validation")
    parser.add_argument("--logit-gain", type=float, default=1.)
    parser.add_argument("--matches-dir", type=Path)
    parser.add_argument("--tag", default="force_recurrent_strain02")
    parser.add_argument("--match-feedback-gain", type=float, default=0.)
    args = parser.parse_args()
    report = run(args.root, args.selection, args.checkpoint, args.base, args.one_head,
                 args.output_dir, args.report, device=args.device,
                 repeats=args.repeats, cohort=args.cohort,
                 logit_gain=args.logit_gain, matches_dir=args.matches_dir,
                 tag=args.tag,
                 match_feedback_gain=args.match_feedback_gain)
    print(json.dumps({key: report[key] for key in (
        "case_count", "student_better_image_count", "all_saved_q1_valid",
        "all_finite_vjp",
    )}))


if __name__ == "__main__":
    main()
