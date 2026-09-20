"""Non-periodic scattered Beurling treecode versus blocked direct quadrature."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_direct import direct_beurling_apply
from qcopt.forward.beurling_treecode import treecode_beurling_apply


def _case(rng: np.random.Generator, count: int, *, near: bool) -> dict[str, float | int | str]:
    source = 0.05 + 0.42 * rng.random(count) + 1j * (0.05 + 0.9 * rng.random(count))
    if near:
        target = source + (0.018 + 0.012 * rng.random(count)) * np.exp(2j * np.pi * rng.random(count))
    else:
        target = 0.55 + 0.4 * rng.random(count) + 1j * (0.05 + 0.9 * rng.random(count))
    values = rng.normal(size=count) + 1j * rng.normal(size=count)
    weights = (0.4 + 0.8 * rng.random(count)) / count
    t0 = time.perf_counter()
    reference = direct_beurling_apply(source, values, weights, target_points=target, block_size=512)
    direct_seconds = time.perf_counter() - t0
    rows = []
    for theta, order in ((0.4, 6), (0.3, 8), (0.25, 10), (0.2, 12)):
        t0 = time.perf_counter()
        approximation = treecode_beurling_apply(source, values, weights, target_points=target, theta=theta, order=order, max_leaf=32)
        tree_seconds = time.perf_counter() - t0
        rows.append({
            "theta": theta,
            "order": order,
            "elapsed_seconds": tree_seconds,
            "relative_l2_error": float(np.linalg.norm(approximation - reference) / max(np.linalg.norm(reference), 1e-30)),
            "max_abs_error": float(np.max(np.abs(approximation - reference))),
            "speedup_vs_direct": direct_seconds / tree_seconds,
        })
    return {"case": "near" if near else "separated", "points": count, "direct_seconds": direct_seconds, "records": rows}


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(20260919)
    result = {
        "source_target_model": "nonuniform scattered points; whole-plane kernel; no periodic wrapping",
        "cases": [_case(rng, 4096, near=False), _case(rng, 4096, near=True)],
        "elapsed_seconds": 0.0,
        "limitation": "Barnes-Hut Taylor treecode control, not a production FMM or singular quadrature theorem",
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    output = Path("D:/QC_optimization/artifacts/beurling_treecode_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "beurling_treecode_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
