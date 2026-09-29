"""Render only the frozen selected ACROBAT TIFF pairs to common 512² canvases.

No teacher, landmark, network, or optimizer is read. Existing outputs are
preserved and not silently regenerated; the original TIFFs remain on D:.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_acrobat_pair_canvas import _inspect, _render, compute_layout


def render_selection(selection_path: Path, source_root: Path, output_root: Path, *,
                     section: str = "new_train", side: int = 512,
                     skip_missing: bool = False) -> dict:
    if section not in ("new_train", "new_confirmation"):
        raise ValueError("unknown selection section")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    items = [item for rows in selection[section].values() for item in rows]
    rendered = reused = missing = 0
    for index, item in enumerate(items, 1):
        case, stain = item["case"], item["stain"]
        moving, fixed = (source_root / f"{case}_{stain}.tiff",
                         source_root / f"{case}_HE.tiff")
        moving_out = output_root / f"{case}_{stain}_physical512.png"
        fixed_out = output_root / f"{case}_HE_physical512.png"
        meta_out = output_root / f"{case}_physical512_layout.json"
        outputs = (moving_out, fixed_out, meta_out)
        if not moving.is_file() or not fixed.is_file():
            if not skip_missing:
                raise FileNotFoundError(f"missing source TIFF for case {case}")
            missing += 1
            continue
        if all(path.exists() for path in outputs):
            reused += 1
            print(f"{index}/{len(items)} reused {case} {stain}", flush=True)
            continue
        if any(path.exists() for path in outputs):
            raise FileExistsError(f"partial existing canvas output for case {case}")
        metadata = [_inspect(moving), _inspect(fixed)]
        layouts = compute_layout([
            (record["original_hw"][0], record["original_hw"][1],
             record["mpp_xy"][1], record["mpp_xy"][0])
            for record in metadata
        ], side=side)
        for record, layout, source, output in zip(
            metadata, layouts, (moving, fixed), (moving_out, fixed_out), strict=True,
        ):
            record.update(layout)
            record["pyramid_level_read"] = _render(source, output, layout, side=side)
            record["canvas_png"] = str(output)
        meta = {"side": side, "moving": metadata[0], "fixed": metadata[1],
                "warning": "common physical pixel scale and aspect, no slide alignment"}
        meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        rendered += 1
        print(f"{index}/{len(items)} rendered {case} {stain}", flush=True)
    return {"section": section, "cases": len(items),
            "rendered": rendered, "reused": reused, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--section", choices=("new_train", "new_confirmation"),
                        default="new_train")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render_selection(args.selection, args.source_root,
                                      args.output_root,
                                      section=args.section,
                                      skip_missing=args.skip_missing)))


if __name__ == "__main__":
    main()
