"""Post-prediction native-coordinate scoring of released DHR MIIT runs.

Evaluation only: this module never registers images or selects optimizer states.
The same metadata-selected pairs and annotation availability convention as the
earlier MIIT comparison apply. A native MHA is not a certified P1 output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from tools.coordinated_dhr_released_baseline import PAIRS
from tools.coordinated_miit_score import read_points, validate_layout
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit
from tools.digital_compare_appearance import dhr_map_at_unit_queries


def summary(errors, ids):
    errors = np.asarray(errors, dtype=float)
    if errors.shape != (len(ids),) or not len(ids) or not np.isfinite(errors).all():
        raise ValueError("finite error for every available label required")
    return dict(mean=float(errors.mean()), p90=float(np.percentile(errors, 90)),
                maximum=float(errors.max()), per_label=dict(zip(ids, errors.tolist(), strict=True)))


def field_corner_diagnostics(field, chunk_rows=128):
    """All four local cell determinants of identity+saved displacement.

    Saved displacement is in its own raster's pixel units. Reference determinant
    is one. This checks local bilinear-cell orientation on that pixel-center
    rectangle, NOT outer boundary injectivity or the different257-grid P1 class.
    """
    field = np.asarray(field)
    if field.ndim != 3 or field.shape[0] != 2 or min(field.shape[1:]) < 2:
        raise ValueError("expected a (2,H,W) displacement")
    if not np.isfinite(field).all() or chunk_rows < 1:
        raise ValueError("finite field and positive chunk size required")
    h, w = field.shape[1:]
    nonpositive = 0
    minimum = float("inf")
    det = lambda a, b: a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    for first in range(0, h-1, chunk_rows):
        last = min(first+chunk_rows, h-1)
        y, x = np.meshgrid(np.arange(first, last+1, dtype=float),
                           np.arange(w, dtype=float), indexing="ij")
        vertices = np.moveaxis(field[:, first:last+1], 0, -1).astype(float) + np.stack((x, y), -1)
        a, b, c, d = vertices[:-1, :-1], vertices[:-1, 1:], vertices[1:, 1:], vertices[1:, :-1]
        for q in (det(b-a, d-a), det(b-a, c-b), det(c-d, c-b), det(c-d, d-a)):
            minimum = min(minimum, float(q.min()))
            nonpositive += int((q <= 0).sum())
    return dict(field_rows=h, field_columns=w, query_lattice_vertices=h*w,
                checked_corner_count=4*(h-1)*(w-1), nonpositive_corners=nonpositive,
                minimum_corner_ratio=minimum, local_orientation_valid=nonpositive == 0,
                global_homeomorphism_certified=False,
                scope="four corners on native saved displacement pixel-center cells; boundary injectivity NOT checked")


def score(predictions, source_data, comparison_directory):
    import SimpleITK as sitk
    predictions = Path(predictions)
    if predictions.is_dir():
        predictions = predictions / "predictions.json"
    manifest = json.loads(predictions.read_text(encoding="utf-8"))
    if manifest.get("prediction_complete") is not True or manifest.get("annotations_read") is not False:
        raise ValueError("complete image-only predictions required before evaluation")
    expected = [f"miit_{m}_to_{f}" for m, f in PAIRS]
    if [r.get("name") for r in manifest.get("rows", [])] != expected:
        raise ValueError("all three predefined pair attempts required")
    directory, source_data = predictions.parent, Path(source_data)
    rows = []
    for record, (moving_id, fixed_id) in zip(manifest["rows"], PAIRS, strict=True):
        row = dict(name=record["name"], moving_section=moving_id, fixed_section=fixed_id,
                   status=record["status"], complete_call_seconds=record.get("complete_call_seconds"),
                   peak_allocated_bytes=record.get("peak_allocated_bytes"))
        rows.append(row)
        if record["status"] not in ("ok", "failed"):
            raise ValueError("terminal native prediction required")
        if record["status"] != "ok":
            row["error"] = record.get("error")
            continue
        try:
            if (record["moving_section"], record["fixed_section"]) != (moving_id, fixed_id):
                raise ValueError("fixed-to-moving pair direction mismatch")
            wh = {}
            points = {}
            for role, section in (("fixed", fixed_id), ("moving", moving_id)):
                with Image.open(source_data / str(section) / "images/image.tif") as image:
                    wh[role] = np.asarray(image.size)
                    if list(image.size) != record[role+"_original_wh"] or image.getexif().get(274, 1) != 1:
                        raise ValueError("native TIFF dimensions/orientation disagree with prediction")
                points[role] = read_points(source_data / str(section) / "landmarks" / f"{section:02d}.csv",
                                          wh[role], allow_missing_inf=True)
            if set(points["fixed"]) != set(points["moving"]):
                raise ValueError("all nominal IDs must match")
            all_ids = sorted(points["fixed"])
            absent = {role: [label for label in all_ids if np.isposinf(points[role][label]).all()]
                      for role in ("fixed", "moving")}
            missing = set(absent["fixed"]) | set(absent["moving"])
            ids = [label for label in all_ids if label not in missing]
            fixed = np.stack([points["fixed"][label] for label in ids])
            target = np.stack([points["moving"][label] for label in ids])
            field = sitk.GetArrayFromImage(sitk.ReadImage(str(directory / record["field"])))
            params = json.loads((directory / record["postprocessing_params"]).read_text(encoding="utf-8"))
            query = torch.from_numpy(((fixed+.5)/wh["fixed"]).astype(np.float32)).reshape(1, 1, -1, 2)
            mapped = dhr_map_at_unit_queries(field, params, fixed_size=tuple(wh["fixed"]),
                                             moving_size=tuple(wh["moving"]), query=query)[0, 0].double().numpy()
            predicted = mapped*wh["moving"]-.5
            layout = validate_layout(json.loads((Path(comparison_directory) / (record["name"]+"_layout.json")).read_text()))
            for role in ("fixed", "moving"):
                if layout[role]["original_wh"] != wh[role].tolist():
                    raise ValueError("comparison layout is not the same native image")
            predicted_canvas = original_pixel_to_canvas_unit(predicted, layout["moving"], 512)
            target_canvas = original_pixel_to_canvas_unit(target, layout["moving"], 512)
            row.update(status="ok", nominal_landmarks=len(all_ids), scored_landmarks=len(ids),
                       available_pair_labels=ids, unavailable_pair_labels=sorted(missing),
                       unavailable_fixed_labels=absent["fixed"], unavailable_moving_labels=absent["moving"],
                       metrics=dict(native_moving_pixels=summary(np.linalg.norm(predicted-target, axis=1), ids),
                                    canvas_pixels=summary(512*np.linalg.norm(predicted_canvas-target_canvas, axis=1), ids)),
                       topology=field_corner_diagnostics(field))
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
    successful = [r for r in rows if r["status"] == "ok"]
    aggregate = {unit: dict(mean_pair_mean=float(np.mean([r["metrics"][unit]["mean"] for r in successful])),
                            mean_pair_p90=float(np.mean([r["metrics"][unit]["p90"] for r in successful])))
                 for unit in ("native_moving_pixels", "canvas_pixels")} if len(successful) == 3 else None
    return dict(preset=manifest["preset"], prediction_manifest=str(predictions), rows=rows,
                pair_denominator=3, scored_pairs=len(successful), failed_pairs=3-len(successful), all_three=aggregate,
                units="native moving pixels and the SAME previously used512 canvas; physical spacing unknown",
                coordinate_convention="native CSV zero-based pixel centers; publisher exact origin unverified",
                missing_policy="same annotation-only (+inf,+inf) absence for every method; prediction failures retained",
                scope="ONE development specimen, three correlated pairs; no blind or official leaderboard claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "source-data", "comparison-directory", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = score(args.predictions, args.source_data, args.comparison_directory)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(scored_pairs=result["scored_pairs"], all_three=result["all_three"])))


if __name__ == "__main__":
    main()
