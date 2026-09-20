"""BHF arbitrary interior-target near-field subdivision audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.bhf_variation import interior_point_near_far_bhf_variation
from qcopt.mesh import structured_rectangle


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n in (64, 128):
        mesh = structured_rectangle(n, n)
        source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
        triangles = source[mesh.faces]
        b = 0.21 + 0.09j
        a = 1.0 - b
        image = a * source + b * np.conjugate(source)
        fz = np.full(mesh.n_faces, a, dtype=np.complex128)
        variation = 0.12 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.42 - 0.37j) ** 2 / 0.15)
        target_face = (n // 2) * n + n // 2
        bary = np.asarray([0.2, 0.3, 0.5])
        values = []
        for order in (8, 16, 24):
            t0 = time.perf_counter()
            value = interior_point_near_far_bhf_variation(
                source, image, mesh.faces, fz, variation, target_face, bary, near_order=order
            )
            values.append(
                {
                    "order": order,
                    "elapsed_seconds": time.perf_counter() - t0,
                    "velocity_real": float(value.real),
                    "velocity_imag": float(value.imag),
                }
            )
        last = complex(values[-1]["velocity_real"], values[-1]["velocity_imag"])
        records.append(
            {
                "grid": f"{n}x{n} cells",
                "faces": int(mesh.n_faces),
                "target_face": int(target_face),
                "records": values,
                "order16_minus_order8_abs": float(
                    abs(complex(values[1]["velocity_real"], values[1]["velocity_imag"]) - complex(values[0]["velocity_real"], values[0]["velocity_imag"]))
                ),
                "order24_minus_order16_abs": float(
                    abs(last - complex(values[1]["velocity_real"], values[1]["velocity_imag"]))
                ),
            }
        )
    result = {
        "records": records,
        "scope": "Duffy subdivision for a target strictly inside one affine source face",
        "limitation": "edge/vertex multi-face PV cancellation, nonlinear integration, and atlas gluing remain open",
    }
    (output_dir / "bhf_interior_target_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
