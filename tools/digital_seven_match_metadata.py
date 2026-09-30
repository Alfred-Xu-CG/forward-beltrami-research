"""Attach affine-frame metadata to seven existing image-derived match archives.

The source JSON names the affine used to prewarp the moving image. This script
checks that its point arrays equal the compact archive and that the named
affine agrees with the saved registration map, then writes a new compact
archive with explicit affine metadata. It cannot prove the historical
matcher actually used the affine named in its own JSON provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data, maps, out = args.data_root, args.map_dir, args.output_dir
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for case in ("cc10", "cd31", "ki67", "prospc",
                 "lesions", "rat_kidney", "histo"):
        if case in ("cc10", "cd31", "ki67", "prospc"):
            directory = data / "lung_lesion3_eval/canvas"
            json_path = directory / f"{case}_alignedSG_matches.json"
            match_path = directory / f"{case}_alignedSG_matches.npz"
            affine_path = directory / f"{case}_directSG_affine.npz"
        else:
            directory = data / "birl_anhir_dev/directSG_aligned_matches"
            json_path = directory / f"{case}_matches.json"
            match_path = directory / f"{case}_alignedSG_matches.npz"
            affine_path = (data / "birl_anhir_dev/canvas"
                           / f"{case}_direct_superglue_affine.npz")
        source = json.loads(json_path.read_text(encoding="utf-8"))
        if (source["status"] != "ok" or
                Path(source["affine_map"]).name != affine_path.name or
                source["ransac_inliers"] < 16):
            raise ValueError(f"JSON provenance mismatch: {case}")
        json_fixed = np.asarray(source["source_points_unit"], dtype=np.float32)
        json_target = np.asarray(source["target_points_unit"], dtype=np.float32)
        with np.load(match_path) as archive:
            fixed = archive["source_fixed_unit"].copy()
            target = archive["target_aligned_unit"].copy()
        if (not np.array_equal(json_fixed, fixed) or
                not np.array_equal(json_target, target) or
                fixed.shape != (source["ransac_inliers"], 2) or
                not np.isfinite(fixed).all() or not np.isfinite(target).all()):
            raise ValueError(f"JSON and compact coordinates disagree: {case}")
        with np.load(affine_path) as archive:
            matrix = archive["post_affine_matrix"].copy()
            offset = archive["post_affine_offset"].copy()
        with np.load(maps / f"{case}_imageforce_unet1000_safe257.npz") as archive:
            if (not np.array_equal(matrix, archive["post_affine_matrix"]) or
                    not np.array_equal(offset, archive["post_affine_offset"])):
                raise ValueError(f"saved-map and source affine differ: {case}")
        rows.append({
            "case": case, "points": len(fixed),
            "json_affine_map": source["affine_map"],
            "json_points_equal_compact_binary32": True,
            "affine_equal_saved_map_binary32": True,
            "output": f"{case}_alignedSG_matches_with_affine.npz",
        })
    out.mkdir(parents=True)
    for row in rows:
        case = row["case"]
        if case in ("cc10", "cd31", "ki67", "prospc"):
            directory = data / "lung_lesion3_eval/canvas"
            affine_path = directory / f"{case}_directSG_affine.npz"
        else:
            directory = data / "birl_anhir_dev/directSG_aligned_matches"
            affine_path = (data / "birl_anhir_dev/canvas"
                           / f"{case}_direct_superglue_affine.npz")
        with np.load(directory / f"{case}_alignedSG_matches.npz") as archive:
            fixed = archive["source_fixed_unit"].copy()
            target = archive["target_aligned_unit"].copy()
        with np.load(affine_path) as archive:
            matrix = archive["post_affine_matrix"].copy()
            offset = archive["post_affine_offset"].copy()
        np.savez_compressed(out / row["output"],
                            source_fixed_unit=fixed, target_aligned_unit=target,
                            post_affine_matrix=matrix, post_affine_offset=offset)
    (out / "provenance.json").write_text(json.dumps({
        "mode": "seven existing image-only matches with explicit affine metadata",
        "limit": "source JSON is provenance, not independent proof of historical matcher execution",
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "all_json_points_equal": True,
                      "all_affines_equal_saved_map": True}))


if __name__ == "__main__":
    main()
