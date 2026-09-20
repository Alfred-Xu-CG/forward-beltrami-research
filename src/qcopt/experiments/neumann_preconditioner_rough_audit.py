"""Cost-inclusive GMRES stress test for rough Fourier Beltrami coefficients."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.beurling_gmres import periodic_beltrami_gmres


def _rough_random_field(n: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Build a deterministic bounded field with modes spanning a rough band."""
    if n < 8 or not np.isfinite(amplitude) or not 0.0 < amplitude < 1.0:
        raise ValueError("n must be at least 8 and amplitude must lie in (0,1)")
    x = np.arange(n, dtype=np.float64)[None, :] / n
    y = np.arange(n, dtype=np.float64)[:, None] / n
    field = np.zeros((n, n), dtype=np.complex128)
    max_mode = max(2, n // 8)
    modes: list[tuple[int, int]] = []
    for kx in range(-max_mode, max_mode + 1):
        for ky in range(-max_mode, max_mode + 1):
            if (kx, ky) != (0, 0) and max(abs(kx), abs(ky)) >= max_mode // 2:
                modes.append((kx, ky))
    # Keep the number of exponentials bounded so generation cost is negligible
    # compared with the FFT solve, while still covering high frequencies.
    selected = modes[: min(48, len(modes))]
    for kx, ky in selected:
        coefficient = rng.normal() + 1j * rng.normal()
        phase = rng.uniform(0.0, 2.0 * np.pi)
        field += coefficient * np.exp(2j * np.pi * (kx * x + ky * y) + 1j * phase)
    scale = np.max(np.abs(field))
    return np.ascontiguousarray(amplitude * field / max(scale, np.finfo(float).eps))


def run(output_dir: Path, n: int = 512, cases: int = 2) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260919)
    records = []
    for index in range(cases):
        amplitude = float(rng.uniform(0.25, 0.72))
        mu = _rough_random_field(n, amplitude, rng)
        started = time.perf_counter()
        cold = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=160)
        cold_seconds = time.perf_counter() - started
        case = {
            "case": index,
            "amplitude": amplitude,
            "cold_iterations": cold.iterations,
            "cold_seconds": cold_seconds,
            "cold_residual": cold.operator_residual,
            "cold_converged": cold.converged,
        }
        for order in (1, 2, 3):
            started = time.perf_counter()
            result = periodic_beltrami_gmres(mu, rtol=1e-9, maxiter=160, preconditioner_order=order)
            elapsed = time.perf_counter() - started
            case[f"order{order}"] = {
                "iterations": result.iterations,
                "elapsed_seconds": elapsed,
                "speedup": cold_seconds / elapsed,
                "iteration_reduction": cold.iterations - result.iterations,
                "residual": result.operator_residual,
                "root_difference": float(np.max(np.abs(result.h - cold.h))),
                "converged": bool(result.converged),
            }
        records.append(case)
    result = {
        "grid": f"{n}x{n}",
        "cases": cases,
        "records": records,
        "speedup_summary": {
            f"order{order}_median": float(np.median([r[f"order{order}"]["speedup"] for r in records]))
            for order in (1, 2, 3)
        },
        "scope": "cost-inclusive matrix-free Neumann preconditioner on high-frequency rough Fourier fields",
        "limitation": "finite rough-band control, not a theorem for arbitrary measurable or near-degenerate coefficients",
    }
    (output_dir / "neumann_preconditioner_rough_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--cases", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.cases), indent=2))


if __name__ == "__main__":
    main()
