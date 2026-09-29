"""Independent-annotator (same lung lesion_3 specimen) proSPC read-only score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_lung_lesion3_score import score_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lung = args.root / "lung_lesion3_eval"
    arms = {
        "frozen_cnn": (lung / "canvas", "directSG_mind_dense257"),
        "slow_strain_0p1": (lung / "match_optimized_eval", "matchopt4_safe257"),
        "slow_strain_1": (lung / "match_optimized_strain1_eval",
                          "matchopt4_strain1_safe257"),
        "kernel_k4": (lung / "kernel_match_eval", "kernelmatch4_safe257"),
        "kernel_k8": (lung / "kernel_match8_eval", "kernelmatch8_safe257"),
        "student": (lung / "matchopt_distilled_eval",
                    "matchoptdistill4_safe257"),
    }
    reports = {name: score_case(lung / "canvas", args.annotations,
                                "prospc", "proSPC-4", map_dir=directory,
                                map_suffix=suffix)
               for name, (directory, suffix) in arms.items()}
    result = {
        "protocol": "same lung-lesion_3 proSPC/HE images, distinct JB annotator",
        "not_new_specimen": True,
        "post_label_access_exploratory": True,
        "landmark_count": 80,
        "metric": "mean fixed-to-moving 5%-scale native JPEG Euclidean pixel TRE",
        "initial_affine_mean_px": reports["frozen_cnn"]["initial_affine"]["mean_px"],
        "per_arm_mean_px": {name: report["dense257_p1"]["mean_px"]
                            for name, report in reports.items()},
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"initial_affine_mean_px":
                      result["initial_affine_mean_px"],
                      "per_arm_mean_px": result["per_arm_mean_px"]}))


if __name__ == "__main__":
    main()
