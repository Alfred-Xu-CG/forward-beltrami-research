"""Test one scalar 64-band gate threshold across disjoint photo contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _scores(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    low = np.asarray(result["photographic_high32"]["fraction64"], dtype=np.float64)
    high = np.asarray(result["photographic_high64"]["fraction64"], dtype=np.float64)
    if low.ndim != 1 or high.shape != low.shape or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("invalid paired photo scores")
    if len(low) != len(result["photo_names"]) * result["photo_variants_per_name"]:
        raise ValueError("photo count does not match metadata")
    return low, high, result["photo_names"]


def _rates(low: np.ndarray, high: np.ndarray, threshold: float) -> dict[str, float]:
    false_positive = float(np.mean(low > threshold))
    true_positive = float(np.mean(high > threshold))
    return {
        "false_positive_rate_high32": false_positive,
        "true_positive_rate_high64": true_positive,
        "balanced_accuracy": 0.5*(1-false_positive+true_positive),
    }


def _fit_threshold(low: np.ndarray, high: np.ndarray) -> tuple[float, dict[str, float]]:
    ordered = np.unique(np.concatenate((low, high)))
    candidates = np.concatenate(([-1.0], (ordered[:-1]+ordered[1:])/2, [1.0]))
    metrics = [_rates(low, high, float(value)) for value in candidates]
    best = max(range(len(candidates)), key=lambda i: metrics[i]["balanced_accuracy"])
    return float(candidates[best]), metrics[best]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    args = parser.parse_args()
    old_low, old_high, old_names = _scores(args.old)
    new_low, new_high, new_names = _scores(args.new)
    output = {"score": "fraction64", "classify_high64_if": "score>threshold",
              "old_photo_names": old_names, "new_photo_names": new_names,
              "directions": {}}
    for label, train_low, train_high, test_low, test_high in (
        ("old6_to_new4", old_low, old_high, new_low, new_high),
        ("new4_to_old6", new_low, new_high, old_low, old_high),
    ):
        threshold, train = _fit_threshold(train_low, train_high)
        output["directions"][label] = {
            "train_count_per_family": len(train_low),
            "test_count_per_family": len(test_low),
            "threshold": threshold,
            "train": train,
            "test": _rates(test_low, test_high, threshold),
        }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
