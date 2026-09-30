"""Render the already audited same-specimen affine counterfactual comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = sorted(report["rows"], key=lambda row:
                  row["affine_counterfactual_mean_TRE_native_moving_px"])
    if len(rows) != 20 or report["direction_count"] != 20:
        raise ValueError("expected the twenty-direction frozen report")
    fig, ax = plt.subplots(figsize=(10.8, 7.8), dpi=170)
    for index, row in enumerate(rows):
        affine = row["affine_counterfactual_mean_TRE_native_moving_px"]
        full = row["full_dynamic_mean_TRE_native_moving_px"]
        ax.plot((full, affine), (index, index), color="#a9b5c3", lw=1.6,
                zorder=1)
    index = list(range(20))
    ax.scatter([row["affine_counterfactual_mean_TRE_native_moving_px"]
                for row in rows], index, s=30, color="#b05640",
               label="Vertex-L2 affine correction of frozen map", zorder=3)
    ax.scatter([row["full_dynamic_mean_TRE_native_moving_px"]
                for row in rows], index, s=30, color="#17699a",
               label="Full learned 257² P1 map", zorder=4)
    ax.set_yticks(index, [row["direction"].replace("_to_", " → ") for row in rows])
    ax.set_xlabel("Mean manual-landmark TRE (native moving 5%-JPEG pixels; lower is better)")
    ax.set_title("20 stain directions of ONE previously viewed lung specimen\n"
                 "Affine correction fitted to map vertices, never to landmarks")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
