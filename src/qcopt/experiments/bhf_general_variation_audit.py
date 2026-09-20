"""Audit the arbitrary-base BHF variation reference operator.

The kernel is evaluated from the published three-simple-pole formula.  This
experiment intentionally reports direct blocked-quadrature cost and a
source/evaluation collision case; it is not a claim of a production PV
quadrature or a nonlinear flow integrator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.bhf_variation import arbitrary_base_bhf_variation


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for n_source, n_eval in ((4096, 1024), (16384, 4096)):
        rng = np.random.default_rng(5100 + n_source)
        source = (rng.random(n_source) + 1j * rng.random(n_source)) * 1.8 - 0.9 + 0.17j
        evaluation = (rng.random(n_eval) + 1j * rng.random(n_eval)) * 1.8 - 0.9
        b = 0.22 + 0.08j
        a = 1.0 - b
        source_image = a * source + b * np.conjugate(source)
        evaluation_image = a * evaluation + b * np.conjugate(evaluation)
        fz = np.full(n_source, a, dtype=np.complex128)
        variation = 0.25 * np.exp(-np.abs(source - 0.23 - 0.29j) ** 2 / 0.55)
        weights = np.full(n_source, 3.24 / n_source)
        t0 = time.perf_counter()
        velocity = arbitrary_base_bhf_variation(
            evaluation,
            source,
            evaluation_image,
            source_image,
            fz,
            variation,
            weights,
            block_size=256,
        )
        elapsed = time.perf_counter() - t0
        records.append(
            {
                "source_points": n_source,
                "evaluation_points": n_eval,
                "elapsed_seconds": elapsed,
                "pair_evaluations": n_source * n_eval,
                "max_velocity": float(np.max(np.abs(velocity))),
                "finite": bool(np.all(np.isfinite(velocity))),
            }
        )

    # The normalized map fixes 0 and 1, while a source/evaluation collision
    # exercises the explicit diagonal omission path.
    source = np.array([-0.7 + 0.2j, -0.15 + 0.4j, 0.42 + 0.22j, 1.3 - 0.2j])
    evaluation = np.array([source[1], 0.0 + 0.0j, 1.0 + 0.0j])
    source_image = source.copy()
    fz = np.ones(source.shape, dtype=np.complex128)
    variation = np.array([0.1 + 0.03j, -0.2 + 0.08j, 0.05 - 0.1j, 0.07 + 0.02j])
    weights = np.full(source.shape, 0.2)
    collision_velocity = arbitrary_base_bhf_variation(
        evaluation,
        source,
        evaluation,
        source_image,
        fz,
        variation,
        weights,
    )
    result = {
        "records": records,
        "collision_case": {
            "velocity": [[float(v.real), float(v.imag)] for v in collision_velocity],
            "normalization_abs": [float(abs(collision_velocity[1])), float(abs(collision_velocity[2]))],
        },
        "scope": "arbitrary-base BHF first variation using direct blocked quadrature",
        "limitations": [
            "O(N_source*N_eval) reference cost; no NUFFT/FMM acceleration",
            "principal-value diagonal term is omitted when source_image equals evaluation_image",
            "does not yet integrate the nonlinear BHF ODE or provide a sphere atlas discretization",
        ],
    }
    (output_dir / "bhf_general_variation_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

