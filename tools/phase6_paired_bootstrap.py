"""Exploratory paired uncertainty intervals for existing Phase VI per-case JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("image_mse", "query_map_mse", "face_beltrami_mse",
           "source_centroid_beltrami_mse")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--cluster-size", type=int, default=1,
                        help="resample contiguous groups of this many cases")
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")

    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    for key in ("seed", "count", "target_family", "image_side",
                "test_seed", "fine_cycles", "fine_side"):
        if key in before and key in after and before[key] != after[key]:
            raise ValueError(f"unpaired metadata {key}: {before[key]} != {after[key]}")
    first = before.get("samples", before.get("final_heldout", {}).get("samples"))
    second = after.get("samples", after.get("final_heldout", {}).get("samples"))
    if first is None or second is None:
        raise ValueError("both records must contain per-case samples")
    if len(first) != len(second) or not first:
        raise ValueError("per-case lists must have equal nonzero length")
    if args.cluster_size < 1 or len(first) % args.cluster_size:
        raise ValueError("cluster size must be positive and divide sample count")

    rng = np.random.default_rng(args.seed)
    cluster_count = len(first) // args.cluster_size
    indices = rng.integers(0, cluster_count,
                           size=(args.draws, cluster_count))
    out = {"before": str(args.before), "after": str(args.after),
           "paired_count": len(first), "bootstrap_draws": args.draws,
           "bootstrap_seed": args.seed, "cluster_size": args.cluster_size,
           "cluster_count": cluster_count, "metrics": {}}
    for key in METRICS:
        if key not in first[0] or key not in second[0]:
            continue
        a = np.asarray([row[key] for row in first], dtype=np.float64)
        b = np.asarray([row[key] for row in second], dtype=np.float64)
        difference = b - a
        cluster_means = difference.reshape(cluster_count,
                                           args.cluster_size).mean(axis=1)
        bootstrap_means = cluster_means[indices].mean(axis=1)
        out["metrics"][key] = {
            "mean_before": float(a.mean()), "mean_after": float(b.mean()),
            "mean_difference_after_minus_before": float(difference.mean()),
            "paired_mean_difference_95_percentile_interval":
                np.quantile(bootstrap_means, [0.025, 0.975]).tolist(),
            "after_lower_count": int(np.count_nonzero(b < a)),
            "cluster_mean_after_lower_count":
                int(np.count_nonzero(cluster_means < 0)),
        }
    print(json.dumps(out, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
