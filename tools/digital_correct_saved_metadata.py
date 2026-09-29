"""Mechanical repair of two independently identified report metadata mistakes.

Numerical arrays, timings, losses, maps, and scores are never changed.
This script only changes direct-SG affine provenance flags and the HistoReg
layout's inherited BIRL-scale description; it is safe to rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas", type=Path, required=True)
    parser.add_argument("--lung-canvas", type=Path, required=True)
    args = parser.parse_args()
    changed = []
    for directory in (args.canvas, args.lung_canvas):
        for path in sorted(directory.glob("*directSG_mind_dense257*.json")):
            if "eval_p1" in path.stem:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            if not any(token in Path(report["affine"]).stem
                       for token in ("directSG", "direct_superglue")):
                raise ValueError(f"not a directSG affine report: {path}")
            if report.get("DHR_derived_initial_affine_used") is False:
                continue
            if report.get("DHR_derived_initial_affine_used") is not True:
                raise ValueError(f"unexpected provenance field: {path}")
            report["DHR_derived_initial_affine_used"] = False
            report["affine_provenance"] = (
                "direct image-only SuperPoint/SuperGlue similarity; "
                "matcher weights distributed by DHR package")
            report["feature_geometry"] = "legacy_align_corners"
            report["metadata_correction"] = (
                "provenance and old feature-geometry labels corrected after "
                "independent review; numerical result unchanged")
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            changed.append(str(path))
    layout_path = args.canvas / "histo_layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    expected = "same pixel scale within BIRL scale-5pc pair; no scanner spacing"
    corrected = ("HistoReg full-resolution CD4/CD68 native JPEGs at assumed "
                 "common pixel scale; not BIRL scale-5pc")
    if layout["scale_assumption"] == expected:
        layout["scale_assumption"] = corrected
        layout_path.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
        changed.append(str(layout_path))
    elif layout["scale_assumption"] != corrected:
        raise ValueError("unexpected HistoReg scale assumption")
    for path in sorted(args.canvas.glob("histo*eval_p1.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("scale_assumption") == expected:
            report["scale_assumption"] = corrected
            report["metadata_correction"] = (
                "HistoReg dataset scale description corrected; scores unchanged")
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            changed.append(str(path))
        elif report.get("scale_assumption") not in (None, corrected):
            raise ValueError(f"unexpected HistoReg score scale: {path}")
    oracle_report = args.lung_canvas.parent / "four_pair_LABEL_ORACLE_F1x4_500_eval_p1.json"
    if oracle_report.exists():
        report = json.loads(oracle_report.read_text(encoding="utf-8"))
        if report.get("training_tuning_labels_used") is False:
            report["training_tuning_labels_used"] = True
            report["not_deployable_label_oracle"] = True
            report["metadata_correction"] = (
                "labels were explicitly loaded and minimized by the oracle; "
                "numerical scores unchanged")
            oracle_report.write_text(json.dumps(report, indent=2) + "\n",
                                     encoding="utf-8")
            changed.append(str(oracle_report))
    print(json.dumps({"changed_count": len(changed), "paths": changed}))


if __name__ == "__main__":
    main()
