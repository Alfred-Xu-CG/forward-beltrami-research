"""Direct-original integer canvas scaling, preserving the frozen normalized frame.

No matcher, landmarks or optimizer is loaded. This does NOT upsample old PNGs:
it uses the same RGB/PIL BILINEAR rendering as digital_birl_pair_canvas, with
EXACT integer multiples of the already accepted resize dimensions and padding.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


_CASES = ("histo", "lesions", "rat_kidney")
_CENTER_CONVENTION = "canvas_xy=(original_xy+0.5)*effective_scale_xy-0.5+padding_xy"


def _path(value):
    if not isinstance(value, (str, Path)) or (isinstance(value, str) and not value):
        raise ValueError("nonempty string or Path required")
    return Path(value).resolve()


def _json(path):
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"JSON object required: {path}")
    return result


def _integers(values, name, minimum=1):
    if (not isinstance(values, list) or len(values) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in values)):
        raise ValueError(f"{name} must contain two integers >= {minimum}")
    return np.asarray(values, dtype=np.int64)


def _vector(values, name, positive=False):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if result.shape != (2,) or not np.isfinite(result).all() or (positive and (result <= 0).any()):
        raise ValueError(f"finite {'positive ' if positive else ''}two-axis {name} required")
    return result


def _basename(value):
    if not isinstance(value, str) or not value:
        raise ValueError("recorded raster/affine path must be a nonempty string")
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _plan(case, canvas_dir, matches_dir, output_dir, factor):
    layout_path = canvas_dir / f"{case}_layout.json"
    affine_path = canvas_dir / f"{case}_initial_affine.npz"
    match_path = matches_dir / f"{case}_common_sg_raw_matches.json"
    old = _json(layout_path)
    side = old.get("side")
    if isinstance(side, bool) or not isinstance(side, int) or side < 2:
        raise ValueError("layout side must be an integer >=2")
    if old.get("landmarks_used") is not False:
        raise ValueError("layout must explicitly exclude landmark-based preparation")
    new_side = factor * side
    new = copy.deepcopy(old)
    new.update(side=new_side, origin_layout=str(layout_path), scale_factor=factor,
               render_method="direct original RGB raster; PIL BILINEAR; exact integer layout scaling")
    sources = []
    for role in ("moving", "fixed"):
        item = old.get(role)
        if not isinstance(item, dict):
            raise ValueError("moving/fixed layout objects required")
        original = _integers(item.get("original_wh"), "original_wh")
        resized = _integers(item.get("resized_wh"), "resized_wh")
        padding = _integers(item.get("padding_xy"), "padding_xy", minimum=0)
        if (padding + resized > side).any():
            raise ValueError("resize/padding does not fit the old canvas")
        scale = _vector(item.get("effective_original_to_canvas_scale_xy"), "effective scale", positive=True)
        if not np.array_equal(scale, resized / original):
            raise ValueError("effective resize scale must match resized_wh/original_wh")
        _vector(item.get("original_mpp_xy"), "original pixel scale", positive=True)
        mpp = item.get("canvas_mpp")
        if isinstance(mpp, bool) or not isinstance(mpp, (int, float)) or not np.isfinite(mpp) or mpp <= 0:
            raise ValueError("positive finite canvas_mpp required")
        if item.get("original_center_to_canvas_center") != _CENTER_CONVENTION:
            raise ValueError("unsupported original pixel-center convention")
        source = _path(item.get("source"))
        with Image.open(source) as image:
            if image.size != tuple(original):
                raise ValueError("actual original image dimensions disagree with layout")
            if image.getexif().get(274, 1) != 1:
                raise ValueError("nontrivial EXIF orientation is unsupported")
        target_wh = factor * resized
        if (target_wh > original).any():
            raise ValueError("native-detail upsampling is unsupported: target exceeds original dimensions")
        target = output_dir / f"{case}_{role}{new_side}.png"
        current = new[role]
        current.update(resized_wh=target_wh.tolist(), padding_xy=(factor*padding).tolist(),
                       effective_original_to_canvas_scale_xy=(factor*scale).tolist(),
                       canvas_mpp=mpp/factor, canvas_png=str(target))
        sources.append((source, tuple(original), target, current))
    with np.load(affine_path, allow_pickle=False) as archive:
        try:
            matrix, offset = archive["post_affine_matrix"], archive["post_affine_offset"]
        except KeyError as error:
            raise ValueError("post-affine matrix/offset required") from error
        if (matrix.shape != (2, 2) or offset.shape != (2,) or
                matrix.dtype not in (np.float32, np.float64) or offset.dtype not in (np.float32, np.float64) or
                not np.isfinite(matrix).all() or not np.isfinite(offset).all() or
                np.linalg.det(matrix.astype(np.float64)) <= 0):
            raise ValueError("finite positive frozen affine required")
    record = _json(match_path)
    if record.get("targets_manual_landmarks_or_dense_teacher_loaded") is not False:
        raise ValueError("matches must explicitly exclude targets/manual/dense teachers")
    if (isinstance(record.get("image_side"), bool) or not isinstance(record.get("image_side"), int) or
            record.get("image_side") != side):
        raise ValueError("match prediction frame does not agree with old canvas side")
    for role in ("fixed", "moving"):
        if _basename(record.get(role)) != _basename(old[role].get("canvas_png")):
            raise ValueError("match raster does not agree with old layout")
    if _basename(record.get("affine")) != affine_path.name:
        raise ValueError("match affine path does not agree with old initializer")
    if (not np.array_equal(np.asarray(record.get("post_affine_matrix")), matrix) or
            not np.array_equal(np.asarray(record.get("post_affine_offset")), offset)):
        raise ValueError("match affine values do not agree with frozen initializer")
    try:
        source_points = np.asarray(record.get("source_points_unit"), dtype=np.float64)
        target_points = np.asarray(record.get("target_points_unit"), dtype=np.float64)
        confidence = np.asarray(record.get("confidence"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid normalized correspondence table") from error
    if (source_points.ndim != 2 or source_points.shape[-1] != 2 or len(source_points) < 1 or
            target_points.shape != source_points.shape or confidence.shape != (len(source_points),) or
            not np.isfinite(source_points).all() or not np.isfinite(target_points).all() or
            not np.isfinite(confidence).all() or (source_points < 0).any() or (source_points > 1).any() or
            (target_points < 0).any() or (target_points > 1).any() or
            (confidence < 0).any() or (confidence > 1).any() or confidence.sum() <= 0):
        raise ValueError("finite in-domain points and nonnegative confidence required")
    if (isinstance(record.get("raw_matches"), bool) or not isinstance(record.get("raw_matches"), int) or
            record.get("raw_matches") != len(source_points)):
        raise ValueError("raw match count disagrees with table")
    prediction_side = record.get("prediction_side", side)
    if isinstance(prediction_side, bool) or not isinstance(prediction_side, int) or prediction_side < 1:
        raise ValueError("original matcher prediction side must be a positive integer")
    new_affine = output_dir / affine_path.name
    new_match = output_dir / match_path.name
    transported = copy.deepcopy(record)
    transported.update(fixed=new["fixed"]["canvas_png"], moving=new["moving"]["canvas_png"],
        affine=str(new_affine), image_side=new_side, origin_match_record=str(match_path),
        prediction_side=prediction_side, origin_image_side=side,
        transport_factor=factor, physical_robust_scale_factor=factor,
        transport_method="exact integer normalized-frame transport; points/confidences copied, not fresh matcher predictions")
    outputs = [entry[2] for entry in sources] + [output_dir/layout_path.name, new_affine, new_match]
    if any(path.exists() for path in outputs):
        raise FileExistsError("all scaled output paths must be nonexistent")
    return dict(case=case, side=new_side, sources=sources, layout=new,
                layout_path=outputs[2], affine_source=affine_path, affine_output=new_affine,
                matches=transported, match_output=new_match)


def prepare_scaled_canvases(canvas_dir, matches_dir, output_dir, *, cases=("histo", "rat_kidney"), factor=2):
    """Prepare exact-frame direct-original canvases without overwriting material.

    Output matches are TRANSPORTED old predictions, not a fresh high-resolution
    matcher. Under integer doubling, normalized points/affine remain unchanged;
    multiply a canvas-pixel robust scale by ``factor`` to preserve its physical
    meaning. Header/metadata/collision preflight completes before any output;
    later I/O failures may leave explicit partial outputs, never overwritten.
    """
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive Python integer")
    if (not isinstance(cases, (list, tuple)) or not cases or any(
            not isinstance(case, str) or case not in _CASES for case in cases) or len(set(cases)) != len(cases)):
        raise ValueError("distinct known case names required: histo, lesions, rat_kidney")
    canvas_dir, matches_dir, output_dir = map(_path, (canvas_dir, matches_dir, output_dir))
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError("output directory is an existing file")
    plans = [_plan(case, canvas_dir, matches_dir, output_dir, factor) for case in cases]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for plan in plans:
        for source, expected_size, target, item in plan["sources"]:
            with Image.open(source) as image:
                if image.size != expected_size or image.getexif().get(274, 1) != 1:
                    raise ValueError("source dimensions/orientation changed after preflight")
                raster = image.convert("RGB")
            resized = raster.resize(tuple(item["resized_wh"]), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (plan["side"], plan["side"]), (255, 255, 255))
            canvas.paste(resized, tuple(item["padding_xy"]))
            with target.open("xb") as handle:
                canvas.save(handle, format="PNG")
            del raster, resized, canvas
        # Exclusive creation plus byte-copy preserves ALL archive arrays exactly.
        with plan["affine_source"].open("rb") as source, plan["affine_output"].open("xb") as target:
            shutil.copyfileobj(source, target)
        for target, record in ((plan["layout_path"], plan["layout"]), (plan["match_output"], plan["matches"])):
            with target.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")
        results.append(dict(case=plan["case"], side=plan["side"], layout=str(plan["layout_path"]),
                            affine=str(plan["affine_output"]), matches=str(plan["match_output"])))
    return dict(render="original RGB/PIL BILINEAR; exact integer layout", matcher_reexecuted=False,
                normalized_frame_unchanged=True, factor=factor, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canvas_dir", "matches_dir", "output_dir"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=_CASES, default=["histo", "rat_kidney"])
    parser.add_argument("--factor", type=int, default=2)
    print(json.dumps(prepare_scaled_canvases(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
