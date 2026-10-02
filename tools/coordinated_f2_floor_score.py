"""Post-prediction, read-only scoring of the three fixed F2 reserve pairs.

Reuse the all20 frame/certificate/metric helpers. This is ONE already viewed
specimen, not an independent clinical cohort or label-selected inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from tools.coordinated_f2_floor_compare import CASES, FRACTIONS
from tools.coordinated_lung_all20_score import (
    PREFIX, STAIN_NAME, _json, _load_predictions, _layouts, _path, _affine,
    _safe_map, _metrics, scaled_landmarks, original_pixel_to_canvas_unit,
    p1_at_queries_numpy,
)


def _comparison(path):
    manifest = path/"comparison.json" if path.is_dir() else path
    value = _json(manifest)
    rows = value.get("rows")
    if (value.get("annotations_read") is not False or value.get("attempts") != 6
            or value.get("paired_cases") != 3 or not isinstance(rows, list) or len(rows) != 6):
        raise ValueError("six completed label-free comparison attempts required before labels")
    expected = [(name, fraction) for name in CASES for fraction in FRACTIONS]
    for row, key in zip(rows, expected, strict=True):
        if (not isinstance(row, dict) or isinstance(row.get("floor_safety_fraction"), bool)
                or (row.get("name"), row.get("floor_safety_fraction")) != key
                or row.get("status") not in ("ok", "failed", "skipped")):
            raise ValueError("exact predeclared six ordered terminal case/fraction rows required")
    return value, manifest.parent


def _local_basename(value, directory):
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty comparison artifact path required")
    name = Path(value.replace("\\", "/")).name
    if not name or name in (".", ".."):
        raise ValueError("comparison artifact basename required")
    return directory/name


def _aggregate(rows, method):
    ok = [row for row in rows if row["methods"][method]["status"] == "ok"]
    conditional = {}
    for unit in ("canvas_pixels", "original_moving_5pc_pixels"):
        values = [row["methods"][method]["metrics"][unit] for row in ok]
        means = np.array([v["mean"] for v in values])
        conditional[unit] = None if not values else dict(
            mean_of_direction_means=float(means.mean()), p90_of_direction_means=float(np.percentile(means, 90)),
            max_of_direction_means=float(means.max()), mean_of_direction_p90=float(np.mean([v["p90"] for v in values])),
            maximum_direction_landmark_error=float(max(v["max"] for v in values)))
    worse = sum(row["methods"]["common_affine"]["status"] == "ok" and
                row["methods"][method]["metrics"]["canvas_pixels"]["mean"] >
                row["methods"]["common_affine"]["metrics"]["canvas_pixels"]["mean"] for row in ok)
    return dict(direction_denominator=3, scored_directions=len(ok), failed_or_skipped_directions=3-len(ok),
                all3_equal_direction_metrics=conditional if len(ok) == 3 else None,
                successful_directions_only=conditional, worse_than_common_affine_count=int(worse),
                worse_than_affine_denominator=3, unscored_worse_than_affine_directions=3-len(ok))


def score(canvas, annotations, predictions, comparison, *, affines_from=None):
    comparison_record, directory = _comparison(Path(comparison))
    original, original_directory = _load_predictions(Path(predictions))
    source = {row["name"]: row for row in original["rows"]}
    layouts = _layouts(Path(canvas))
    prepared = []
    for index, name in enumerate(CASES):
        fixed, moving = name.split("_to_")
        row = dict(name=name, fixed_stain=fixed, moving_stain=moving, methods={})
        try:
            affine_path = (Path(affines_from)/(name+"_affine.npz") if affines_from is not None else
                           _path(source[name].get("affine"), original_directory))
            a, b = _affine(affine_path)
            row["methods"]["common_affine"] = dict(status="ok", data=(a, b), path=str(affine_path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            a = b = None
            row["methods"]["common_affine"] = dict(status="failed", error=f"{type(error).__name__}: {error}")
        for offset, fraction in enumerate(FRACTIONS):
            prediction = comparison_record["rows"][2*index+offset]
            arm = "floor100" if fraction == 1. else "floor095"
            item = dict(status=prediction["status"], comparison_record=prediction)
            row["methods"][arm] = item
            if item["status"] != "ok":
                continue
            try:
                if a is None:
                    raise ValueError("common affine unavailable")
                output = _local_basename(prediction.get("output"), directory)
                report = _local_basename(prediction.get("report"), directory)
                config = _json(report)
                if (config.get("configuration", {}).get("f2_floor_safety_fraction") != fraction
                        or config.get("f2_floor_safety_fraction") != fraction):
                    raise ValueError("optimizer report does not match the requested floor-reserve arm")
                vertices, geometry = _safe_map(dict(output=str(output.resolve()), report=str(report.resolve())), directory, a, b)
                item.update(data=vertices, geometry=geometry)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                item.update(status="failed", error=f"{type(error).__name__}: {error}")
        prepared.append(row)

    # No annotation was opened above; both complete prediction declarations checked.
    relevant = ("he", "cc10", "ki67")
    points = {stain: scaled_landmarks(Path(annotations)/(PREFIX+STAIN_NAME[stain]+"-les3.csv"))
              for stain in relevant}
    ids = set(points["he"])
    if len(ids) != 80 or any(set(points[stain]) != ids for stain in relevant):
        raise ValueError("exact same eighty IDs required in each stain; no intersection")
    ids = sorted(ids, key=int)
    for stain in relevant:
        values = np.stack([points[stain][key] for key in ids])
        if (not np.isfinite(values).all() or np.any(values < -.5)
                or np.any(values > np.asarray(layouts[stain]["original_wh"])-.5)):
            raise ValueError("landmark outside native 5pc JPEG frame")
    for row in prepared:
        fixed, moving = row["fixed_stain"], row["moving_stain"]
        query = original_pixel_to_canvas_unit(np.stack([points[fixed][key] for key in ids]), layouts[fixed], 512)
        target = np.stack([points[moving][key] for key in ids])
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > 1):
            raise ValueError("fixed landmark outside unit canvas")
        row["landmark_count"] = 80
        for name, item in row["methods"].items():
            if item["status"] != "ok":
                continue
            data = item.pop("data")
            try:
                mapped = query@data[0].T+data[1] if name == "common_affine" else p1_at_queries_numpy(data, query, "ac")
                item["metrics"] = _metrics(mapped, target, layouts[moving], ids)
            except (ValueError, RuntimeError) as error:
                item.update(status="failed", error=f"{type(error).__name__}: {error}")
    return dict(protocol="three predeclared paired directions of ONE previously viewed lung-lesion3 specimen; development scoring, not independent patients",
                comparison=str(Path(comparison)), original_predictions=str(Path(predictions)),
                canvas=str(Path(canvas)), annotations=str(Path(annotations)),
                affines_from=None if affines_from is None else str(Path(affines_from)),
                predictor_annotations_read=False, scorer_annotations_read=True,
                chronology_scope="completion/no-label declarations checked, not proof against historical earlier label access",
                direction_denominator=3, paired_attempts=6, specimen_count=1, canvas_side=512,
                landmark_count_per_direction=80, map_direction="fixed to moving P1ac then the common positive affine",
                landmark_conversion="50pc CSV to5pc JPEG: (coordinate+.5)/10-.5; nominal scale convention with subpixel uncertainty",
                metric_scope="equal direction weights; all eighty IDs per direction, correlated directions not patients",
                failure_policy="failed arms retained, no fallback/cohort drop; no all3 numeric aggregate if an arm is unscored",
                rows=prepared, aggregates={arm: _aggregate(prepared, arm) for arm in ("common_affine", "floor100", "floor095")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("canvas", "annotations", "predictions", "comparison", "affines_from", "output"):
        parser.add_argument("--"+key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = score(args.canvas, args.annotations, args.predictions, args.comparison, affines_from=args.affines_from)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["aggregates"], allow_nan=False))


if __name__ == "__main__":
    main()
