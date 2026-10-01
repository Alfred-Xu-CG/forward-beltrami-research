"""Separate read-only development landmark scoring; never called by optimizer."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit, canvas_unit_to_original_pixel, q1_at_queries
from tools.digital_dhr_field_eval import _landmarks
from tools.digital_q1_real_eval import load_effective_vertices


def score(layout_path, fixed_path, moving_path, maps, affine, output):
    if output.exists():
        raise FileExistsError(output)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    loaded = {name: load_effective_vertices(path) for name,path in maps.items()}
    if not all(report["composite_representation_valid"] for _,report in loaded.values()):
        raise ValueError("invalid saved map")
    with Image.open(layout["fixed"]["source"]) as image:
        fixed_size = image.size
    with Image.open(layout["moving"]["source"]) as image:
        moving_size = image.size
    fixed = _landmarks(fixed_path,fixed_size)
    moving = _landmarks(moving_path,moving_size)
    ids = sorted(fixed.keys() & moving.keys())
    if not ids:
        raise ValueError("no shared landmark IDs")
    f = np.stack([fixed[key] for key in ids]); m = np.stack([moving[key] for key in ids])
    query = original_pixel_to_canvas_unit(f,layout["fixed"],layout["side"])
    expected = original_pixel_to_canvas_unit(m,layout["moving"],layout["side"])
    predicted = {name:q1_at_queries(vertices,query) for name,(vertices,_) in loaded.items()}
    with np.load(affine) as data:
        predicted["common_affine"] = query @ data["post_affine_matrix"].astype(np.float64).T + data["post_affine_offset"].astype(np.float64)
    results = {}
    for name,p in predicted.items():
        canvas_error = np.linalg.norm((p-expected)*layout["side"],axis=-1)
        native = np.linalg.norm(canvas_unit_to_original_pixel(p,layout["moving"],layout["side"])-m,axis=-1)
        results[name] = dict(mean_canvas_px=float(canvas_error.mean()),p90_canvas_px=float(np.percentile(canvas_error,90)),
                             p95_canvas_px=float(np.percentile(canvas_error,95)),max_canvas_px=float(canvas_error.max()),
                             mean_native_moving_px=float(native.mean()),per_landmark_canvas_px=dict(zip(ids,canvas_error.tolist())))
    report = dict(protocol="reused public development specimen, not independent or official ACROBAT/ANHIR test",
                  interpolation="Q1",map_direction="fixed to moving",landmark_count=len(ids),
                  fixed_only_ids=sorted(fixed.keys()-moving.keys()),moving_only_ids=sorted(moving.keys()-fixed.keys()),
                  layout=str(layout_path),results=results)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("layout","fixed_landmarks","moving_landmarks","affine","output"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    parser.add_argument("--map",action="append",required=True,help="name=absolute NPZ path")
    args = parser.parse_args()
    maps = dict(item.split("=",1) for item in args.map)
    result = score(args.layout,args.fixed_landmarks,args.moving_landmarks,
                   {name:Path(path) for name,path in maps.items()},args.affine,args.output)
    print(json.dumps({name:{key:value for key,value in values.items() if key != "per_landmark_canvas_px"}
                      for name,values in result["results"].items()}))


if __name__ == "__main__":
    main()
