"""Predict all ordered stain directions of the known lung-lesion-3 specimen.

This script reads images only. Anatomical annotations must be scored afterward
by a separate command; no landmark path or landmark-derived affine is accepted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

from tools.digital_frozen_residual_predict import predict as predict_dynamic
from tools.digital_mind_dense257_predict import predict as predict_frozen
from tools.digital_superglue_direct_affine import extract


STAINS = ("he", "cc10", "cd31", "ki67", "prospc")


def image_path(canvas: Path, stain: str) -> Path:
    return canvas / ("cc10_fixed512.png" if stain == "he" else
                     f"{stain}_moving512.png")


def run(canvas: Path, output: Path, base: Path, one_head: Path,
        residual: Path, *, device: str, logit_gain: float = 1.,
        affines_from: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("new all-20 output directory required")
    pairs = []
    for fixed, moving in itertools.permutations(STAINS, 2):
        name = f"{fixed}_to_{moving}"
        first, second = image_path(canvas, fixed), image_path(canvas, moving)
        if not first.is_file() or not second.is_file():
            raise FileNotFoundError((first, second))
        pairs.append((name, first, second, output / f"{name}_affine.npz"))
    if affines_from is None:
        affine_report = extract(pairs, output / "affines.json", device_name=device)
    else:
        if not (affines_from / "affines.json").is_file():
            raise FileNotFoundError(affines_from / "affines.json")
        affine_report = json.loads((affines_from / "affines.json").read_text())
        if {row["name"] for row in affine_report["rows"]} != {
            name for name, _, _, _ in pairs
        }:
            raise ValueError("reused affine cohort does not match all 20 directions")
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(affines_from / "affines.json", output / "affines.json")
        for name, _, _, affine in pairs:
            shutil.copy2(affines_from / f"{name}_affine.npz", affine)
    rows = []
    for (name, first, second, affine), match_row in zip(
        pairs, affine_report["rows"], strict=True
    ):
        if match_row["status"] != "ok":
            rows.append({"name": name, "status": match_row["status"]})
            continue
        frozen = predict_frozen(base, one_head, first, second, affine,
                                output / f"{name}_frozen_safe257.npz",
                                device_name=device, repeats=1)
        dynamic = predict_dynamic(base, one_head, residual, first, second,
                                  affine, None,
                                  output / f"{name}_dynamic_safe257.npz",
                                  device_name=device, repeats=1,
                                  logit_gain=logit_gain)
        rows.append({
            "name": name, "status": "ok",
            "frozen_image": frozen["dense_256_P1_descriptor_loss"],
            "dynamic_image": dynamic["residual_image"],
            "frozen_q1": frozen["saved_binary_certificate"]["valid"],
            "dynamic_q1": dynamic["saved_binary_certificate"]["valid"],
            "frozen_finite_vjp": frozen["finite_full_parameter_vjp"],
            "dynamic_finite_vjp": dynamic["finite_full_vjp"],
        })
        print(json.dumps(rows[-1]), flush=True)
    result = {
        "protocol": "all 20 ordered directions in one previously viewed specimen",
        "stains": STAINS,
        "checkpoint": str(residual),
        "logit_gain": logit_gain,
        "affines_reused_from": None if affines_from is None else str(affines_from),
        "annotations_read": False,
        "match_raster_supplied_to_student": False,
        "rows": rows,
    }
    (output / "predictions.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas", "output", "base", "one_head", "residual"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path,
                            required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logit-gain", type=float, default=1.)
    parser.add_argument("--affines-from", type=Path)
    args = parser.parse_args()
    report = run(args.canvas, args.output, args.base, args.one_head,
                 args.residual, device=args.device, logit_gain=args.logit_gain,
                 affines_from=args.affines_from)
    print(json.dumps({"pair_count": len(report["rows"]),
                      "success_count": sum(r["status"] == "ok"
                                           for r in report["rows"])}))


if __name__ == "__main__":
    main()
