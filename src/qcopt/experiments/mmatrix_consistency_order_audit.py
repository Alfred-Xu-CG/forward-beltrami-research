"""Manufactured-solution consistency audit for positive integer wide stencils."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.mmatrix import beltrami_conductivity, integer_wide_stencil_conductances


def _field(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(n, dtype=np.float64) / n
    y = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, y, indexing="xy")
    a = 2.0 * np.pi
    b = 2.0 * np.pi
    c = 6.0 * np.pi
    d = 2.0 * np.pi
    u = np.sin(a * xx) * np.cos(b * yy) + 0.4 * np.cos(c * xx + 0.2) * np.sin(d * yy)
    ux = a * np.cos(a * xx) * np.cos(b * yy) - 0.4 * c * np.sin(c * xx + 0.2) * np.sin(d * yy)
    uy = -b * np.sin(a * xx) * np.sin(b * yy) + 0.4 * d * np.cos(c * xx + 0.2) * np.cos(d * yy)
    uxx = -a * a * np.sin(a * xx) * np.cos(b * yy) - 0.4 * c * c * np.cos(c * xx + 0.2) * np.sin(d * yy)
    uyy = -b * b * np.sin(a * xx) * np.cos(b * yy) - 0.4 * d * d * np.cos(c * xx + 0.2) * np.sin(d * yy)
    uxy = -a * b * np.cos(a * xx) * np.sin(b * yy) - 0.4 * c * d * np.sin(c * xx + 0.2) * np.cos(d * yy)
    return u, uxx, uxy, uyy


def run(output_dir: Path, n_values: tuple[int, ...] = (128, 256, 512)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mu_values = (0.92 * np.exp(0.73j), 0.96 * np.exp(1.11j))
    records: list[dict] = []
    for mu in mu_values:
        tensor = beltrami_conductivity(complex(mu))
        for max_step in (1, 2, 3, 4, 6):
            fit = integer_wide_stencil_conductances(tensor, max_step=max_step)
            by_n = []
            for n in n_values:
                t0 = time.perf_counter()
                u, uxx, uxy, uyy = _field(n)
                h = 1.0 / n
                discrete = np.zeros_like(u)
                # Reconstruct primitive integer vectors from the normalized dictionary.
                # The ordering is deterministic and matches integer_wide_stencil_directions.
                vectors = []
                for p0 in range(max_step + 1):
                    for q0 in range(-max_step, max_step + 1):
                        if p0 == 0 and q0 <= 0 or p0 == 0 and q0 == 0:
                            continue
                        if np.gcd(p0, abs(q0)) != 1:
                            continue
                        vectors.append((p0, q0))
                for weight, (p0, q0) in zip(fit.values, vectors):
                    length2 = float(p0 * p0 + q0 * q0)
                    second = (np.roll(u, (q0, p0), axis=(0, 1)) - 2.0 * u + np.roll(u, (-q0, -p0), axis=(0, 1))) / (h * h * length2)
                    discrete += weight * second
                truth = tensor[0, 0] * uxx + 2.0 * tensor[0, 1] * uxy + tensor[1, 1] * uyy
                err = float(np.sqrt(np.mean((discrete - truth) ** 2)))
                by_n.append({"n": n, "l2_error": err, "elapsed_seconds": time.perf_counter() - t0})
            orders = []
            for old, new in zip(by_n, by_n[1:]):
                orders.append(float(np.log(old["l2_error"] / new["l2_error"]) / np.log(2.0)))
            records.append({
                "mu": [float(mu.real), float(mu.imag)],
                "max_step": max_step,
                "directions": len(fit.directions),
                "tensor_fit_residual": float(fit.residual),
                "all_nonnegative": bool(np.all(fit.values >= -1e-12)),
                "resolution_errors": by_n,
                "empirical_orders": orders,
            })
    result = {
        "n_values": list(n_values),
        "records": records,
        "scope": "periodic manufactured-solution consistency of positive integer wide stencils; not a boundary or arbitrary unstructured-mesh theorem",
    }
    (output_dir / "mmatrix_consistency_order_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
