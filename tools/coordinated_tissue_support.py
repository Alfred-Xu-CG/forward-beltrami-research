"""Prepare fixed MIIT tissue support in an existing image canvas frame.

Evidence is images PLUS released semi-manual tissue masks. No landmarks,
moving-image overlap or candidate maps are read. Native binary support is
resized by nearest neighbor and padded with zero; the optimizer's existing
area pyramid then supplies fractional fixed weights at coarser resolutions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


PAIRS = ((2, 3), (7, 8), (10, 11))
SCOPE = "images plus released semi-manual fixed tissue masks; no manual correspondence landmarks"


def _basename(path):
    return Path(str(path).replace("\\", "/")).name


def prepare_tissue_support(native_mask, native_fixed, layout_path, fixed_canvas, output):
    native_mask, native_fixed, layout_path, fixed_canvas, output = map(
        Path, (native_mask, native_fixed, layout_path, fixed_canvas, output))
    if output.exists():
        raise FileExistsError(output)
    if output.suffix != ".npz":
        raise ValueError("prepared support output must use .npz suffix")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    side, fixed = layout["side"], layout["fixed"]
    if isinstance(side, bool) or not isinstance(side, int) or side < 2:
        raise ValueError("positive integer canvas side required")
    original = np.asarray(fixed["original_wh"])
    resized = np.asarray(fixed["resized_wh"])
    padding = np.asarray(fixed["padding_xy"])
    scale = np.asarray(fixed["effective_original_to_canvas_scale_xy"], dtype=float)
    if any(v.shape != (2,) for v in (original, resized, padding, scale)):
        raise ValueError("two-axis saved layout required")
    if (not all(np.issubdtype(v.dtype, np.integer) for v in (original, resized, padding))
            or np.any(original < 1) or np.any(resized < 1) or np.any(padding < 0)
            or np.any(resized + padding > side)
            or not np.array_equal(scale, resized / original)):
        raise ValueError("exact integer resize/padding layout and saved scale required")
    if _basename(fixed["canvas_png"]) != fixed_canvas.name:
        raise ValueError("fixed canvas basename differs from saved layout")
    if _basename(fixed["source"]) != native_fixed.name:
        raise ValueError("native fixed image basename differs from saved layout")
    for path, expected in ((native_fixed, tuple(original)), (fixed_canvas, (side, side))):
        with Image.open(path) as image:
            if image.size != expected or image.getexif().get(274, 1) != 1:
                raise ValueError("fixed image dimensions/orientation disagree with saved layout")
    with Image.open(native_mask) as image:
        if image.size != tuple(original) or image.getexif().get(274, 1) != 1:
            raise ValueError("native mask dimensions/orientation disagree with fixed image")
        native = np.asarray(image)
    values = np.unique(native)
    if (native.ndim != 2 or not np.isfinite(native).all()
            or not (set(values.tolist()) <= {0, 1} or set(values.tolist()) <= {0, 255})):
        raise ValueError("two-dimensional binary released tissue mask required")
    binary = (native > 0).astype(np.uint8)
    small = np.asarray(Image.fromarray(binary).resize(tuple(resized), Image.Resampling.NEAREST))
    canvas = np.zeros((side, side), dtype=np.uint8)
    x, y = padding
    width, height = resized
    canvas[y:y+height, x:x+width] = small
    if not canvas.any():
        raise ValueError("nonempty fixed tissue support required")
    with Image.open(fixed_canvas) as image:
        original_threshold = 1 - np.asarray(image.convert("L"), dtype=float) / 255 > .04
    metadata = dict(evidence_scope=SCOPE, native_mask=str(native_mask.resolve()),
        native_fixed=str(native_fixed.resolve()), layout=str(layout_path.resolve()),
        fixed_canvas_basename=fixed_canvas.name, side=side,
        original_wh=original.tolist(), resized_wh=resized.tolist(), padding_xy=padding.tolist(),
        effective_original_to_canvas_scale_xy=scale.tolist(), native_orientation=1,
        native_values=values.tolist(), rasterization="native binary >0; PIL nearest resize; zero padding",
        pyramid="existing optimizer area interpolation; fixed fractional weights, fixed denominator at each resolution",
        tissue_pixels=int(canvas.sum()),
        original_threshold_pixels=int(original_threshold.sum()),
        threshold_outside_tissue_pixels=int((original_threshold & ~canvas.astype(bool)).sum()),
        tissue_outside_threshold_pixels=int((canvas.astype(bool) & ~original_threshold).sum()),
        manual_correspondence_landmarks_read=False, moving_overlap_or_candidate_map_read=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, mask=canvas, metadata_json=json.dumps(metadata, allow_nan=False))
    return metadata


def load_tissue_support(path, fixed_path, image_side, *, device="cpu", dtype=torch.float32):
    """Load one prepared binary support without resizing or image-dependent changes."""
    with np.load(path, allow_pickle=False) as data:
        mask = np.array(data["mask"], copy=True)
        metadata = json.loads(str(data["metadata_json"]))
    if (metadata.get("side") != image_side or mask.shape != (image_side, image_side)
            or metadata.get("fixed_canvas_basename") != Path(fixed_path).name):
        raise ValueError("prepared tissue support does not match fixed canvas frame")
    if (metadata.get("evidence_scope") != SCOPE
            or metadata.get("manual_correspondence_landmarks_read") is not False
            or metadata.get("moving_overlap_or_candidate_map_read") is not False
            or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all() or not mask.any()):
        raise ValueError("finite nonempty binary released fixed tissue support required")
    return torch.as_tensor(mask, device=device, dtype=dtype)[None, None], metadata


def prepare_cohort(args):
    source, predictions, output = map(Path, (args.source_data, args.predictions, args.output))
    if predictions.is_dir():
        predictions = predictions / "predictions.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(predictions.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    if len(rows) != 3 or [(r["moving_section"], r["fixed_section"]) for r in rows] != list(PAIRS):
        raise ValueError("the three declared MIIT canvas pairs required")
    records = []
    for row, (_, fixed) in zip(rows, PAIRS, strict=True):
        resolve = lambda name: Path(name) if Path(name).is_absolute() else predictions.parent / name
        path = output / (row["name"] + "_fixed_tissue.npz")
        metadata = prepare_tissue_support(source / str(fixed) / "masks/tissue_mask.tif",
            source / str(fixed) / "images/image.tif", resolve(row["layout"]), resolve(row["fixed"]), path)
        records.append(dict(name=row["name"], fixed_section=fixed, fixed_mask=path.name, **metadata))
    report = dict(evidence_scope=SCOPE, source_data=str(source.resolve()),
                  canvas_predictions=str(predictions.resolve()), rows=records, annotations_read=False,
                  supplied_tissue_annotation=True,
                  annotation_flag_scope="annotations_read=False means evaluation correspondence coordinates; released tissue masks ARE supplied annotations")
    (output / "support.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def load_support_paths(path):
    """Resolve the three reusable supports, with no annotation-coordinate access."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("rows", [])
    if (report.get("evidence_scope") != SCOPE or report.get("annotations_read") is not False
            or [(r.get("name"), r.get("fixed_section")) for r in rows]
            != [(f"miit_{m}_to_{f}", f) for m, f in PAIRS]):
        raise ValueError("three declared fixed MIIT tissue supports required")
    result = {}
    for row in rows:
        support = Path(row["fixed_mask"])
        support = support if support.is_absolute() else path.parent / support
        if not support.is_file():
            raise FileNotFoundError(support)
        result[row["name"]] = support
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="existing three-pair canvas manifest")
    parser.add_argument("--output", type=Path, required=True, help="new reusable support directory")
    result = prepare_cohort(parser.parse_args())
    print(json.dumps({"evidence_scope": result["evidence_scope"], "masks": len(result["rows"])}))


if __name__ == "__main__":
    main()
