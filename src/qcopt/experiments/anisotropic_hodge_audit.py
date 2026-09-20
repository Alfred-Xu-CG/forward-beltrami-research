"""Phase-II audit of a positive diagonal/diamond anisotropic Hodge stencil."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcopt.forward.anisotropic_hodge import diamond_stencil_decomposition
from qcopt.forward.mmatrix import beltrami_conductivity


def run(output_dir: Path, *, radii: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9), angles: int = 4096) -> dict:
    """Measure the angular fraction where the four-direction stencil is positive."""
    if angles < 4:
        raise ValueError("angles must be at least four")
    output_dir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)
    records = []
    max_reconstruction_error = 0.0
    for radius in radii:
        feasible = 0
        minimum_coefficient = np.inf
        for angle in theta:
            tensor = beltrami_conductivity(radius * np.exp(1j * angle))
            result = diamond_stencil_decomposition(tensor)
            feasible += int(result.feasible)
            minimum_coefficient = min(minimum_coefficient, float(np.min(result.conductances)))
            max_reconstruction_error = max(
                max_reconstruction_error, float(np.max(np.abs(result.reconstructed - tensor)))
            )
        records.append(
            {
                "radius": float(radius),
                "angles": int(angles),
                "feasible_fraction": float(feasible / angles),
                "minimum_conductance_over_angles": float(minimum_coefficient),
            }
        )
    result = {
        "schema": "forward-beltrami-anisotropic-hodge-audit-v1",
        "records": records,
        "max_tensor_reconstruction_error": max_reconstruction_error,
        "scope": "fixed axis plus two diagonal directions; exact tensor representation with a positive-conductance audit",
        "limitation": "a negative coefficient means this fixed stencil is not a positive graph Hodge star; rotating or enlarging the direction dictionary may improve coverage but changes the mesh",
    }
    (output_dir / "anisotropic_hodge_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angles", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, angles=args.angles), indent=2))


if __name__ == "__main__":
    main()
