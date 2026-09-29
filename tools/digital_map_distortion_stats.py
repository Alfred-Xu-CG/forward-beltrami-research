"""Read-only normalized P1 triangle Jacobian distribution for saved 257² maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def triangle_jacobians(vertices: np.ndarray) -> np.ndarray:
    if vertices.shape != (1, 257, 257, 2):
        raise ValueError("one 257-square vertex map required")
    y = vertices[0].astype(np.float64)
    a, b, c, d = y[:-1, :-1], y[:-1, 1:], y[1:, 1:], y[1:, :-1]

    def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    jac1 = 256.**2 * cross(b - a, c - a)
    jac2 = 256.**2 * cross(c - a, d - a)
    return np.concatenate((jac1.ravel(), jac2.ravel()))


def analyze(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        with np.load(path) as archive:
            vertices = archive["vertices"]
        j = triangle_jacobians(vertices)
        rows.append({
            "map": str(path), "triangle_count": len(j),
            "nonpositive_count": int(np.sum(j <= 0)),
            "min": float(j.min()), "p001": float(np.percentile(j, .1)),
            "p01": float(np.percentile(j, 1)),
            "p50": float(np.median(j)),
            "p99": float(np.percentile(j, 99)),
            "max": float(j.max()),
            "fraction_below_0p05": float(np.mean(j < .05)),
            "fraction_below_0p1": float(np.mean(j < .1)),
            "fraction_above_5": float(np.mean(j > 5.)),
        })
    return {"method": "saved-P1 normalized two-triangle Jacobian distribution",
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"map": row["map"], "min": row["min"],
                       "p01": row["p01"], "fraction_below_0p1": row["fraction_below_0p1"]}
                      for row in result["rows"]]))


if __name__ == "__main__":
    main()
