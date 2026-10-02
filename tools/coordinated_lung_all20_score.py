"""Read-only post-prediction scoring of ONE known lung-lesion3 specimen.

Twenty correlated stain directions are not twenty patients. Labels are opened
only after the predictor declares all twenty attempts complete. No map, affine,
selection, or annotation is modified; missing/invalid exports remain failures.
"""
from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction
from pathlib import Path

import torch
import numpy as np

from tools.digital_lung_all20_score import STAIN_NAME
from tools.digital_lung_lesion3_score import PREFIX, scaled_landmarks
from tools.digital_birl_landmark_score import (
    original_pixel_to_canvas_unit, canvas_unit_to_original_pixel,
)
from tools.coordinated_score_case import p1_at_queries_numpy
from tools.digital_q1_real_eval import load_effective_vertices
from tools.digital_compare_appearance import dhr_map_at_unit_queries
from tools.digital_dhr_field_eval import q1_corner_determinants


PAIRS = tuple(itertools.permutations(STAIN_NAME, 2))
METHODS = ("analytic", "f2", "dhr")
SIDE = 512
TERMINAL = {"ok", "failed", "skipped"}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _path(value, directory: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty artifact path required")
    path = Path(value)
    # Relocated manifests can chain ../archive/../ references. Canonicalize
    # before opening: their uncollapsed spelling can exceed Windows MAX_PATH
    # even when the actual existing artifact has a short enough native path.
    return (path if path.is_absolute() else directory / path).resolve()


def _load_predictions(predictions: Path) -> tuple[dict, Path]:
    manifest = predictions / "predictions.json" if predictions.is_dir() else predictions
    value = _json(manifest)
    if value.get("prediction_complete") is not True:
        raise ValueError("all prediction attempts must be complete before opening labels")
    if value.get("annotations_read") is not False or value.get("cohort_size") != 20:
        raise ValueError("label-free predictor declaration and cohort_size=20 required")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("exactly twenty ordered unique prediction rows required")
    for row, (fixed, moving) in zip(rows, PAIRS, strict=True):
        if not isinstance(row, dict) or (row.get("name"), row.get("fixed_stain"),
                                       row.get("moving_stain")) != (
                f"{fixed}_to_{moving}", fixed, moving):
            raise ValueError("prediction cohort must match the predeclared ordered twenty directions")
        if row.get("input_status") not in TERMINAL:
            raise ValueError("pending/unknown input status is not a completed prediction")
        raw = row.get("raw_matches")
        if not isinstance(raw, dict) or raw.get("status") not in TERMINAL:
            raise ValueError("pending/unknown matcher status is not a completed prediction")
        methods = row.get("methods")
        if not isinstance(methods, dict) or set(methods) != set(METHODS):
            raise ValueError("analytic, f2 and dhr statuses required for every direction")
        if any(not isinstance(methods[k], dict) or methods[k].get("status") not in TERMINAL
               for k in METHODS):
            raise ValueError("pending/unknown method status is not a completed prediction")
    return value, manifest.parent


def _layouts(canvas: Path) -> dict:
    layouts = {}
    for stain in STAIN_NAME:
        filename = "cc10_layout.json" if stain == "he" else f"{stain}_layout.json"
        value = _json(canvas / filename)
        if value.get("side") != SIDE:
            raise ValueError("this predeclared cohort uses original 512 canvases only")
        layout = value["fixed" if stain == "he" else "moving"]
        wh = np.asarray(layout["original_wh"], dtype=np.float64)
        resized = np.asarray(layout["resized_wh"], dtype=np.float64)
        scale = np.asarray(layout["effective_original_to_canvas_scale_xy"], dtype=np.float64)
        pad = np.asarray(layout["padding_xy"], dtype=np.float64)
        if any(x.shape != (2,) or not np.isfinite(x).all() for x in (wh, resized, scale, pad)):
            raise ValueError("finite two-axis native frame metadata required")
        if (np.any(wh <= 0) or np.any(resized <= 0) or np.any(scale <= 0)
                or np.any(pad < 0) or np.any(pad + resized > SIDE)
                or not np.array_equal(scale, resized / wh)):
            raise ValueError("inconsistent native-size/resize/padding scale metadata")
        layouts[stain] = layout
    return layouts


def _affine(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        a, b = archive["post_affine_matrix"], archive["post_affine_offset"]
    if (a.shape != (2, 2) or b.shape != (2,) or a.dtype not in (np.float32, np.float64)
            or b.dtype not in (np.float32, np.float64)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("finite floating-point common affine required")
    x, y, z, w = (Fraction.from_float(float(v)) for v in a.flat)
    if x * w - y * z <= 0:
        raise ValueError("common affine must have strictly positive exact binary determinant")
    return a, b


def _safe_map(method: dict, directory: Path, a, b) -> tuple[np.ndarray, dict]:
    path = _path(method.get("output"), directory)
    report_path = _path(method.get("report"), directory)
    report = _json(report_path)
    if (report.get("configuration", {}).get("interpolation") != "p1_ac"
            or report.get("landmarks_used") is not False):
        raise ValueError("prediction report must declare P1ac and no landmark use")
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["interpolation"].item()) != "p1_ac":
            raise ValueError("saved interpolation must be explicitly P1ac")
        if not (np.array_equal(archive["post_affine_matrix"], a)
                and np.array_equal(archive["post_affine_offset"], b)):
            raise ValueError("saved map differs from the common affine")
    vertices, certificate = load_effective_vertices(path)
    if not certificate["composite_representation_valid"] or not np.isfinite(vertices).all():
        raise ValueError("saved residual boundary/all-four-corner certificate is invalid")
    metadata = dict(path=str(path), report=str(report_path), interpolation="p1_ac",
                    certificate=certificate,
                    certificate_scope="stored residual all-four-corner signs and exact rectangle boundary plus exact positive affine; independent P1 query evaluation",
                    gradient_steps=report.get("gradient_steps"),
                    failed_trials=report.get("failed_trials"),
                    objective_evaluations=report.get("objective_evaluations"),
                    optimize_seconds=report.get("optimize_seconds"),
                    output_selection=report.get("output_selection"),
                    selected_stage=report.get("selected_stage"))
    return vertices, metadata


def _native(method: dict, directory: Path, a, b) -> tuple[np.ndarray, dict, dict]:
    import SimpleITK as sitk

    field_path = _path(method.get("field"), directory)
    params_path = _path(method.get("postprocessing_params"), directory)
    report_path = _path(method.get("report"), directory)
    configuration_path = _path(method.get("configuration"), directory)
    report, params, configuration = _json(report_path), _json(params_path), _json(configuration_path)
    if (report.get("image_side") != SIDE or report.get("initial_resample_ratio") != 1
            or not np.array_equal(np.asarray(report.get("post_affine_matrix")), a)
            or not np.array_equal(np.asarray(report.get("post_affine_offset")), b)):
        raise ValueError("native DHR runtime must confirm the same affine and unresampled512 frame")
    loading = configuration.get("loading_params", {})
    if loading.get("loader") != "pil" or any(loading.get(k) != 1 for k in (
            "source_resample_ratio", "target_resample_ratio")):
        raise ValueError("native DHR configuration must use the declared common-canvas PIL frame")
    if any(params.get(k) != 1 for k in (
            "source_resample_ratio", "target_resample_ratio", "initial_resample_ratio")):
        raise ValueError("native DHR exported frame must have unit resample ratios")
    for key in ("pad_1", "pad_2"):
        padding = np.asarray(params.get(key))
        if padding.shape != (2, 2) or not np.all(padding == 0):
            raise ValueError("native DHR exported common canvases must be unpadded")
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(field_path)))
    if field.shape != (2, SIDE, SIDE) or not np.isfinite(field).all():
        raise ValueError("finite (2,512,512) native saved pixel-displacement field required")
    corners = q1_corner_determinants(field)
    if not np.isfinite(corners).all():
        raise ValueError("native Q1 corner determinants are nonfinite")
    metadata = dict(field=str(field_path), postprocessing_params=str(params_path),
                    report=str(report_path), configuration=str(configuration_path),
                    runtime_seconds=report.get("runtime_seconds"),
                    field_shape=list(field.shape), minimum_corner_determinant=float(corners.min()),
                    nonpositive_corner_count=int((corners <= 0).sum()),
                    cells_with_nonpositive_corner=int((corners <= 0).any(0).sum()),
                    geometry_scope="local signs of saved native pixel-center Q1 displacement; NOT a global/boundary homeomorphism certificate, no repair or scoring exclusion",
                    objective_scope="native CLAHE/NCC/regularizer/free-boundary objective, not the common coordinated objective")
    return field, params, metadata


def _metrics(mapped, target_px, layout, ids) -> dict:
    mapped = np.asarray(mapped, dtype=np.float64)
    expected = original_pixel_to_canvas_unit(target_px, layout, SIDE)
    if mapped.shape != expected.shape or not np.isfinite(mapped).all():
        raise ValueError("all eighty predicted queries must be finite; no query dropping")
    canvas = np.linalg.norm((mapped - expected) * SIDE, axis=-1)
    native = np.linalg.norm(canvas_unit_to_original_pixel(mapped, layout, SIDE) - target_px, axis=-1)
    def summary(values):
        return dict(mean=float(values.mean()), p90=float(np.percentile(values, 90)),
                    max=float(values.max()), per_landmark=dict(zip(ids, values.tolist(), strict=True)))
    return dict(canvas_pixels=summary(canvas), original_moving_5pc_pixels=summary(native),
                outside_unit_canvas_queries=int(((mapped < 0) | (mapped > 1)).any(-1).sum()))


def _aggregate(rows, method: str) -> dict:
    ok = [row for row in rows if row["methods"][method]["status"] == "ok"]
    def summarize(unit):
        if not ok:
            return None
        values = [r["methods"][method]["metrics"][unit] for r in ok]
        means = np.asarray([v["mean"] for v in values])
        return dict(mean_of_direction_means=float(means.mean()),
                    p90_of_direction_means=float(np.percentile(means, 90)),
                    max_of_direction_means=float(means.max()),
                    mean_of_direction_p90=float(np.mean([v["p90"] for v in values])),
                    maximum_direction_landmark_error=float(max(v["max"] for v in values)))
    conditional = {unit: summarize(unit) for unit in ("canvas_pixels", "original_moving_5pc_pixels")}
    worse = sum(row["methods"]["common_affine"]["status"] == "ok" and
                row["methods"][method]["metrics"]["canvas_pixels"]["mean"] >
                row["methods"]["common_affine"]["metrics"]["canvas_pixels"]["mean"] for row in ok)
    return dict(direction_denominator=20, scored_directions=len(ok), failed_or_skipped_directions=20-len(ok),
                all20_equal_direction_metrics=conditional if len(ok) == 20 else None,
                successful_directions_only=conditional,
                worse_than_common_affine_count=int(worse), worse_than_affine_denominator=20,
                unscored_worse_than_affine_directions=20-len(ok),
                failure_policy="failed/skipped directions retained; no affine fallback; no all20 numeric aggregate when any direction is unscored")


def score(canvas: Path, annotations: Path, predictions: Path, *, affines_from: Path | None = None) -> dict:
    """Score completed exports. A local affine-directory override aids moved copies.

    Overrides do not alter prediction maps and must still match every exported
    affine numerically exactly. Relative artifact paths use the manifest parent.
    """
    manifest, directory = _load_predictions(Path(predictions))  # MUST precede any label access.
    layouts = _layouts(Path(canvas))
    prepared = []
    for row in manifest["rows"]:
        item = dict(name=row["name"], fixed_stain=row["fixed_stain"], moving_stain=row["moving_stain"],
                    methods={})
        try:
            affine_path = (Path(affines_from) / f"{row['name']}_affine.npz" if affines_from is not None
                           else _path(row.get("affine"), directory))
            a, b = _affine(affine_path)
            item["methods"]["common_affine"] = dict(status="ok", data=(a, b), path=str(affine_path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            a = b = None
            item["methods"]["common_affine"] = dict(status="failed", error=f"{type(error).__name__}: {error}")
        for name in METHODS:
            prediction = row["methods"][name]
            if prediction["status"] != "ok":
                item["methods"][name] = dict(status=prediction["status"], prediction=prediction)
                continue
            try:
                if a is None:
                    raise ValueError("common affine unavailable; equality cannot be verified")
                data = _native(prediction, directory, a, b) if name == "dhr" else _safe_map(prediction, directory, a, b)
                item["methods"][name] = dict(status="ok", data=data, prediction=prediction)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
                item["methods"][name] = dict(status="failed", prediction=prediction,
                                             error=f"{type(error).__name__}: {error}")
        prepared.append(item)

    # First label access is below: all prediction statuses and exports were examined.
    landmarks = {stain: scaled_landmarks(Path(annotations) / f"{PREFIX}{original}-les3.csv")
                 for stain, original in STAIN_NAME.items()}
    ids = set(landmarks["he"])
    if len(ids) != 80 or any(set(points) != ids for points in landmarks.values()):
        raise ValueError("exact same eighty IDs required in every stain; no intersection/cohort dropping")
    ids = sorted(ids, key=int)
    for stain, points in landmarks.items():
        coordinates = np.stack([points[key] for key in ids])
        if (not np.isfinite(coordinates).all() or np.any(coordinates < -.5)
                or np.any(coordinates > np.asarray(layouts[stain]["original_wh"]) - .5)):
            raise ValueError(f"landmark outside native 5pc JPEG frame: {stain}")
    for row in prepared:
        fixed, moving = row["fixed_stain"], row["moving_stain"]
        source = np.stack([landmarks[fixed][key] for key in ids])
        target = np.stack([landmarks[moving][key] for key in ids])
        query = original_pixel_to_canvas_unit(source, layouts[fixed], SIDE)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > 1):
            raise ValueError("fixed landmark outside unit canvas")
        row["landmark_count"] = 80
        for name, item in row["methods"].items():
            if item["status"] != "ok":
                continue
            data = item.pop("data")
            try:
                if name == "common_affine":
                    a, b = data
                    mapped = query @ a.T + b
                elif name == "dhr":
                    field, params, item["geometry"] = data
                    mapped = dhr_map_at_unit_queries(field, params, fixed_size=(SIDE, SIDE), moving_size=(SIDE, SIDE),
                        query=torch.tensor(query, dtype=torch.float32).reshape(1, 1, -1, 2))[0, 0].numpy()
                else:
                    vertices, item["geometry"] = data
                    mapped = p1_at_queries_numpy(vertices, query, "ac")
                item["metrics"] = _metrics(mapped, target, layouts[moving], ids)
            except (ValueError, RuntimeError) as error:
                item.update(status="failed", error=f"{type(error).__name__}: {error}")
    return dict(protocol="all20 ordered directions in ONE previously viewed physical lung-lesion3 specimen; post-prediction development scoring, not blind/independent patients",
                prediction_manifest=str(Path(predictions)), prediction_complete=True,
                canvas=str(Path(canvas)), annotations=str(Path(annotations)),
                affines_from=None if affines_from is None else str(Path(affines_from)),
                predictor_annotations_read=False, scorer_annotations_read=True,
                chronology_scope="relies on predictor completion/no-label declarations; does not prove historical absence of earlier label access",
                map_direction="fixed to moving", canvas_side=SIDE, direction_denominator=20,
                landmark_count_per_direction=80, specimen_count=1,
                landmark_conversion="50pc CSV to5pc JPEG: (coordinate+.5)/10-.5; nominal scale convention with subpixel uncertainty",
                cohort_statistics_scope="equal weight per ordered direction; correlated directions, not1600 independent patients",
                rows=prepared, aggregates={name: _aggregate(prepared, name) for name in ("common_affine", *METHODS)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("canvas", "annotations", "predictions", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--affines-from", type=Path, help="optional local copies of unchanged {name}_affine.npz archives")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = score(args.canvas, args.annotations, args.predictions, affines_from=args.affines_from)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["aggregates"], allow_nan=False))


if __name__ == "__main__":
    main()
