"""Post-prediction native DHR STANDARD scoring of the existing22-case expansion.

Uses the validated native saved-field evaluator, never the old hardcoded512 DHR
wrappers. All22 prediction attempts must be terminal before any manual labels
are opened. This is development evidence from three previously viewed specimens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from tools.coordinated_dhr_existing_inputs import existing_rows, STAINS, PREFIX
from tools.coordinated_dhr_released_score import summary, field_corner_diagnostics
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit
from tools.digital_compare_appearance import dhr_map_at_unit_queries
from tools.digital_dhr_field_eval import _landmarks
from tools.digital_lung_lesion3_score import scaled_landmarks


def _basename(path):
    return Path(str(path).replace("\\", "/")).name


def _load_predictions(path, root):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "predictions.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = existing_rows(root)
    if (value.get("prediction_complete") is not True or value.get("annotations_read") is not False
            or value.get("preset") != "default_initial_nonrigid" or value.get("pair_denominator") != 22):
        raise ValueError("complete image-only STANDARD predictions for all22 required before labels")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 22:
        raise ValueError("exactly22 prediction attempts required")
    for row, case in zip(rows, expected, strict=True):
        if (not isinstance(row, dict) or row.get("name") != case["name"]
                or row.get("status") not in ("ok", "failed")
                or any(_basename(row.get(role, "")) != _basename(case[role]) for role in ("fixed", "moving"))):
            raise ValueError("all22 ordered terminal cases with the exact fixed-to-moving image identities required")
    return value, path


def _layout(role, wh):
    original = np.asarray(role["original_wh"])
    resized = np.asarray(role["resized_wh"], dtype=float)
    scale = np.asarray(role["effective_original_to_canvas_scale_xy"], dtype=float)
    pad = np.asarray(role["padding_xy"], dtype=float)
    if (any(a.shape != (2,) for a in (original, resized, scale, pad))
            or not np.array_equal(original, wh)
            or not all(np.isfinite(a).all() for a in (original, resized, scale, pad))
            or np.any(original <= 0) or np.any(resized <= 0) or np.any(scale <= 0)
            or np.any(pad < 0) or np.any(pad+resized > 512)
            or not np.array_equal(scale, resized/original)):
        raise ValueError("same native image dimensions and exact saved512 resize/pad layout required")
    return role


def _case_data(record, root):
    """Existing dataset-specific readers, directions and declared ID policies."""
    root = Path(root)
    case = next(r for r in existing_rows(root) if r["name"] == record["name"])
    wh = {}
    for role in ("fixed", "moving"):
        with Image.open(case[role]) as image:
            wh[role] = image.size
            if (image.getexif().get(274, 1) != 1 or image.mode not in ("RGB", "L")
                    or list(image.size) != record[role+"_original_wh"]):
                raise ValueError("prediction native dimensions/orientation disagree with local original")
    if record["name"] in ("histo", "rat_kidney"):
        layout = json.loads((root / f"birl_anhir_dev/canvas/{record['name']}_layout.json").read_text(encoding="utf-8"))
        if layout.get("side") != 512:
            raise ValueError("original512 comparison layout required")
        if record["name"] == "histo":
            directory = root / "HistoReg_CD68_CD4"
            labels = dict(fixed=directory/"Landmarks_CD4.csv", moving=directory/"Landmarks_CD68.csv")
        else:
            directory = root / "birl_anhir_dev/labels_eval_only/rat-kidney_/scale-5pc"
            labels = dict(fixed=directory/"Rat-Kidney_HE.csv", moving=directory/"Rat-Kidney_PanCytokeratin.csv")
        layouts = {role: _layout(layout[role], wh[role]) for role in ("fixed", "moving")}
        points = {role: _landmarks(labels[role], wh[role]) for role in ("fixed", "moving")}
        expected_count = 77 if record["name"] == "histo" else 69
    else:
        fixed, moving = record["name"].split("_to_")
        layouts, points = {}, {}
        for role, stain in (("fixed", fixed), ("moving", moving)):
            filename = "cc10" if stain == "he" else stain
            saved = json.loads((root / f"lung_lesion3_eval/canvas/{filename}_layout.json").read_text(encoding="utf-8"))
            if saved.get("side") != 512:
                raise ValueError("original512 lung comparison layout required")
            layouts[role] = _layout(saved["fixed" if stain == "he" else "moving"], wh[role])
            points[role] = scaled_landmarks(root / f"lung_lesion3_eval/annotations50/{PREFIX}{STAINS[stain]}-les3.csv")
        expected_count = 80
    for role in ("fixed", "moving"):
        if _basename(layouts[role]["source"]) != _basename(case[role]):
            raise ValueError("saved layout source is a different native image")
    fixed_only = sorted(set(points["fixed"])-set(points["moving"]))
    moving_only = sorted(set(points["moving"])-set(points["fixed"]))
    ids = sorted(set(points["fixed"]) & set(points["moving"]))
    expected_fixed_only = ["70", "71"] if record["name"] == "rat_kidney" else []
    if len(ids) != expected_count or fixed_only != expected_fixed_only or moving_only:
        raise ValueError("existing77/69/80 paired-ID policy changed; no new intersection or label dropping")
    arrays = {role: np.stack([points[role][label] for label in ids]) for role in ("fixed", "moving")}
    for role, values in arrays.items():
        if not np.isfinite(values).all() or np.any(values < -.5) or np.any(values > np.asarray(wh[role])-.5):
            raise ValueError("evaluation label outside native pixel-center image frame")
    return wh, layouts, arrays, ids, fixed_only, moving_only


def native_metrics(field, params, fixed, target, fixed_wh, moving_wh, moving_layout, ids):
    """Native fixed pixels -> native moving pixels -> unchanged512 moving frame."""
    fixed, target = np.asarray(fixed), np.asarray(target)
    if (fixed.shape != (len(ids), 2) or target.shape != fixed.shape
            or not np.isfinite(fixed).all() or not np.isfinite(target).all()):
        raise ValueError("finite fixed/target coordinate for every declared ID required")
    query = torch.tensor((fixed+.5)/np.asarray(fixed_wh), dtype=torch.float32).reshape(1, 1, -1, 2)
    mapped = dhr_map_at_unit_queries(field, params, fixed_size=tuple(fixed_wh),
                                   moving_size=tuple(moving_wh), query=query)[0, 0].double().numpy()
    predicted = mapped*np.asarray(moving_wh)-.5
    canvas = original_pixel_to_canvas_unit(predicted, moving_layout, 512)
    expected = original_pixel_to_canvas_unit(target, moving_layout, 512)
    return dict(native_moving_pixels=summary(np.linalg.norm(predicted-target, axis=1), ids),
                canvas_pixels=summary(512*np.linalg.norm(canvas-expected, axis=1), ids))


def _field_scalars(field):
    maximum = 0.
    for first in range(0, field.shape[1], 128):
        chunk = field[:, first:first+128].astype(np.float64)
        maximum = max(maximum, float(np.hypot(chunk[0], chunk[1]).max()))
    return dict(shape=list(field.shape), dtype=str(field.dtype),
                minimum_dx=float(field[0].min()), maximum_dx=float(field[0].max()),
                minimum_dy=float(field[1].min()), maximum_dy=float(field[1].max()),
                maximum_displacement_norm=maximum, units="saved-field lattice pixels")


def _score_record(record, directory, root):
    import SimpleITK as sitk
    field = sitk.GetArrayFromImage(sitk.ReadImage(str((directory / record["field"]).resolve())))
    params = json.loads((directory / record["postprocessing_params"]).resolve().read_text(encoding="utf-8"))
    topology = field_corner_diagnostics(field)
    wh, layouts, points, ids, fixed_only, moving_only = _case_data(record, root)
    return dict(status="ok", scored_landmarks=len(ids), available_pair_labels=ids,
                fixed_only_ids=fixed_only, moving_only_ids=moving_only,
                native_fixed_wh=list(wh["fixed"]), native_moving_wh=list(wh["moving"]),
                metrics=native_metrics(field, params, points["fixed"], points["moving"],
                                       wh["fixed"], wh["moving"], layouts["moving"], ids),
                topology=topology, field_scalars=_field_scalars(field))


def _aggregate(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    def values(unit):
        return dict(mean_pair_mean=float(np.mean([r["metrics"][unit]["mean"] for r in ok])),
                    mean_pair_p90=float(np.mean([r["metrics"][unit]["p90"] for r in ok])))
    conditional = {unit: values(unit) for unit in ("canvas_pixels", "native_moving_pixels")} if ok else None
    return dict(direction_denominator=len(rows), scored_directions=len(ok),
                failed_directions=len(rows)-len(ok), all_directions=conditional if len(ok) == len(rows) else None,
                successful_directions_only=conditional)


def score(predictions, data_root):
    root = Path(data_root).resolve()
    manifest, path = _load_predictions(predictions, root)  # all22 terminal before label access
    rows = []
    for record in manifest["rows"]:
        row = {key: record.get(key) for key in ("name", "status", "configuration", "complete_call_seconds",
                "registration_seconds", "preprocessing_seconds", "initial_seconds", "nonrigid_seconds",
                "peak_allocated_bytes", "cuda_free_bytes_before_case", "cuda_total_bytes")}
        rows.append(row)
        if record["status"] == "failed":
            row["error"] = record.get("error")
            continue
        try:
            row.update(_score_record(record, path.parent, root))
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
    lung = _aggregate(rows[:20])
    histo, kidney = _aggregate(rows[20:21]), _aggregate(rows[21:])
    equal = None
    if all(group["all_directions"] is not None for group in (lung, histo, kidney)):
        equal = {key: float(np.mean([group["all_directions"]["canvas_pixels"][key] for group in (lung, histo, kidney)]))
                 for key in ("mean_pair_mean", "mean_pair_p90")}
    return dict(prediction_manifest=str(path), data_root=str(root), preset=manifest["preset"],
                pair_denominator=22, scored_pairs=sum(r["status"] == "ok" for r in rows), rows=rows,
                lung_all20=lung, histo=histo, rat_kidney=kidney, equal_specimen_canvas=equal,
                scope="three previously viewed development specimens; twenty lung directions are correlated, not twenty patients",
                map_direction="fixed native pixels to moving native pixels, then the existing512 comparison layout",
                coordinate_convention="lung50pc CSV to5pc JPEG (x+.5)/10-.5; Histo/kidney existing native CSV reader; nominal pixel origins",
                units="same512 moving-canvas pixels and native moving-image pixels; physical spacing unknown; no cross-specimen native-pixel average",
                failure_policy="all22 terminal predictions required; failed outputs retained with no fallback; complete-cohort aggregates withheld on any failure",
                input_identity_check="existing filenames, dimensions and layout metadata; relocated predictions supported, not a historical content-identity proof",
                predictor_annotations_read=False, scoring_is_post_prediction=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("predictions", "data-root", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = score(args.predictions, args.data_root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(scored_pairs=result["scored_pairs"], equal_specimen_canvas=result["equal_specimen_canvas"])))


if __name__ == "__main__":
    main()
