"""Posthoc score the saved native DHR initial affine, never its dense field.

This only decomposes an already completed development baseline. No registration,
model selection, initializer transfer or parameter optimization is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.coordinated_dhr_released_baseline import PAIRS
from tools.coordinated_dhr_released_score import summary
from tools.coordinated_miit_score import read_points, validate_layout


def map_initial_native(points, theta, params, preprocessed_hw):
    """Fixed native pixel centers -> moving native pixel centers through theta."""
    points, theta = np.asarray(points, dtype=float), np.asarray(theta, dtype=float)
    extent = np.asarray(preprocessed_hw, dtype=float)[::-1]
    if theta.shape == (1, 2, 3):
        theta = theta[0]
    sf, sm = float(params["target_resample_ratio"]), float(params["source_resample_ratio"])
    ratio = float(params.get("initial_resample_ratio", 1))
    pf = np.asarray([params["pad_2"][1][0], params["pad_2"][0][0]], dtype=float)
    pm = np.asarray([params["pad_1"][1][0], params["pad_1"][0][0]], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or theta.shape != (2, 3)
            or extent.shape != (2,) or np.any(extent < 2)
            or not all(np.isfinite(v).all() for v in (points, theta, extent, pf, pm, [sf, sm, ratio]))
            or min(sf, sm, ratio) <= 0 or np.any(pf < 0) or np.any(pm < 0)):
        raise ValueError("finite native points/2x3 theta, positive ratios and valid preprocessed frame required")
    normalized = 2 * ((points + .5) * sf + pf) / ratio / extent - 1
    mapped = normalized @ theta[:, :2].T + theta[:, 2]
    return ((mapped + 1) * .5 * extent * ratio - pm) / sm - .5


def score(predictions, source_data, comparison_directory):
    predictions, source_data, comparison_directory = map(Path, (predictions, source_data, comparison_directory))
    if predictions.is_dir():
        predictions = predictions / "predictions.json"
    manifest = json.loads(predictions.read_text(encoding="utf-8"))
    if manifest.get("prediction_complete") is not True or manifest.get("annotations_read") is not False:
        raise ValueError("completed native image-only predictions required before scoring")
    expected = [(f"miit_{m}_to_{f}", m, f) for m, f in PAIRS]
    if [(r.get("name"), r.get("moving_section"), r.get("fixed_section")) for r in manifest["rows"]] != expected:
        raise ValueError("all three declared native pair attempts required")
    rows = []
    for record in manifest["rows"]:
        row = dict(name=record["name"], status="failed")
        rows.append(row)
        try:
            if record["status"] != "ok":
                raise ValueError("native pipeline did not complete")
            if record.get("initial_transform_frame") != "preprocessed normalized [-1,1] target-to-source affine; align_corners=False":
                raise ValueError("explicit saved native initial affine frame required")
            points = {}
            for role in ("fixed", "moving"):
                section = record[role + "_section"]
                with Image.open(source_data / str(section) / "images/image.tif") as image:
                    if list(image.size) != record[role + "_original_wh"] or image.getexif().get(274, 1) != 1:
                        raise ValueError("native image dimensions/orientation disagree with saved prediction")
                points[role] = read_points(source_data / str(section) / "landmarks" / f"{section:02d}.csv",
                    record[role + "_original_wh"], allow_missing_inf=True)
            if set(points["fixed"]) != set(points["moving"]):
                raise ValueError("all nominal labels required in both images")
            nominal = sorted(points["fixed"])
            absent = {role: [key for key in nominal if np.isposinf(points[role][key]).all()]
                      for role in ("fixed", "moving")}
            missing = set(absent["fixed"]) | set(absent["moving"])
            ids = [key for key in nominal if key not in missing]
            fixed = np.stack([points["fixed"][key] for key in ids])
            target = np.stack([points["moving"][key] for key in ids])
            params = json.loads((predictions.parent / record["postprocessing_params"]).read_text())
            mapped = map_initial_native(fixed, record["initial_transform"], params, record["preprocessed_shape"][-2:])
            layout = validate_layout(json.loads((comparison_directory / (record["name"] + "_layout.json")).read_text()))
            for role in ("fixed", "moving"):
                if layout[role]["original_wh"] != record[role + "_original_wh"]:
                    raise ValueError("comparison canvas is not the same original image")
            # The SAME existing canvas metric: native displacement times the
            # two exact PIL-resize scales. Padding/half-pixel cancel in errors.
            error = mapped - target
            canvas_error = error * np.asarray(layout["moving"]["effective_original_to_canvas_scale_xy"])
            row.update(status="ok", nominal_landmarks=len(nominal), scored_landmarks=len(ids),
                available_pair_labels=ids, unavailable_pair_labels=sorted(missing),
                unavailable_fixed_labels=absent["fixed"], unavailable_moving_labels=absent["moving"],
                predicted_moving_native_pixels=dict(zip(ids, mapped.tolist(), strict=True)),
                metrics=dict(native_moving_pixels=summary(np.linalg.norm(error, axis=1), ids),
                             canvas_pixels=summary(np.linalg.norm(canvas_error, axis=1), ids)))
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
    ok = [row for row in rows if row["status"] == "ok"]
    aggregate = {unit: dict(mean_pair_mean=float(np.mean([row["metrics"][unit]["mean"] for row in ok])),
        mean_pair_p90=float(np.mean([row["metrics"][unit]["p90"] for row in ok])))
        for unit in ("native_moving_pixels", "canvas_pixels")} if len(ok) == 3 else None
    return dict(prediction_manifest=str(predictions), preset=manifest["preset"], rows=rows,
        pair_denominator=3, scored_pairs=len(ok), failed_pairs=3-len(ok), all_three=aggregate,
        scope="native initial affine only; same previously used MIIT specimen and available paired labels",
        coordinate_convention="CSV zero-based native pixel centers; exact publisher origin unverified",
        initial_transform_evaluation="float64 analytic evaluation of saved float32 theta; not a rerasterized dense field",
        final_dense_fields_read=False, registration_or_selection_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "source-data", "comparison-directory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = score(args.predictions, args.source_data, args.comparison_directory)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(scored_pairs=result["scored_pairs"], all_three=result["all_three"])))


if __name__ == "__main__":
    main()
