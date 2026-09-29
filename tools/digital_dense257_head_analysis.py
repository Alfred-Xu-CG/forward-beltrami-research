"""Measure what a trained 257² recurrent head adds to its frozen parent map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _q1_upsample(coarse: np.ndarray) -> np.ndarray:
    fine = np.empty((2 * coarse.shape[0] - 1, 2 * coarse.shape[1] - 1, 2),
                    dtype=np.float64)
    fine[::2, ::2] = coarse
    fine[1::2, ::2] = (coarse[:-1] + coarse[1:]) * .5
    fine[::2, 1::2] = (coarse[:, :-1] + coarse[:, 1:]) * .5
    fine[1::2, 1::2] = (coarse[:-1, :-1] + coarse[:-1, 1:]
                       + coarse[1:, :-1] + coarse[1:, 1:]) * .25
    return fine


def analyze(parent: Path, candidate: Path, ids: list[int], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for case in ids:
        for arm in ("actual", "blank"):
            with np.load(parent / f"{case}_{arm}_safe_q1.npz") as data:
                old = data["vertices"].astype(np.float64)
                old_affine = (data["post_affine_matrix"].copy(),
                              data["post_affine_offset"].copy())
            with np.load(candidate / f"{case}_{arm}_safe_q1.npz") as data:
                new = data["vertices"].astype(np.float64)
                new_affine = (data["post_affine_matrix"].copy(),
                              data["post_affine_offset"].copy())
            if old.shape != (1, 257, 257, 2) or new.shape != old.shape or any(
                not np.array_equal(a, b) for a, b in zip(old_affine, new_affine, strict=True)
            ):
                raise ValueError(f"incompatible case {case} {arm}")
            delta = (new - old)[0]
            high = delta - _q1_upsample(delta[::2, ::2])
            lengths = np.linalg.norm(delta, axis=-1)
            rms = float(np.sqrt(np.mean(delta**2)))
            high_rms = float(np.sqrt(np.mean(high**2)))
            rows.append({
                "case": case, "arm": arm,
                "mean_vertex_motion": float(lengths.mean()),
                "p95_vertex_motion": float(np.percentile(lengths, 95)),
                "max_vertex_motion": float(lengths.max()),
                "coordinate_rms_change": rms,
                "fine_component_rms": high_rms,
                "fine_fraction_of_rms": high_rms / rms if rms else None,
            })
    report = {"question": "does the new 257-side full-vertex head change its parent",
              "normalization": "unit canvas; fine component is delta minus Q1 prolongation of delta at 129² even-index vertices",
              "rows": rows}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.parent, args.candidate, args.ids, args.output)
    for arm in ("actual", "blank"):
        rows = [row for row in report["rows"] if row["arm"] == arm]
        print(arm, "mean coordinate RMS:",
              np.mean([row["coordinate_rms_change"] for row in rows]),
              "mean fine fraction:",
              np.mean([row["fine_fraction_of_rms"] for row in rows]))


if __name__ == "__main__":
    main()
