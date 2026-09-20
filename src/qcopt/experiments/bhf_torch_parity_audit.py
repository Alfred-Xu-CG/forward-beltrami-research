"""Compare GPU float32 BHF assembly with the NumPy near/far reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_jacobians
from qcopt.forward.bhf_variation import all_vertex_near_far_bhf_variation
from qcopt.mesh import structured_rectangle


def run(gpu_velocity: Path, output: Path, n: int = 32) -> dict:
    mesh = structured_rectangle(n, n)
    source = mesh.vertices[:, 0] + 1j * mesh.vertices[:, 1]
    base = 0.18 + 0.07j
    image = (1.0 - base) * source + base * np.conjugate(source)
    image_xy = np.column_stack((image.real, image.imag))
    jacobian = face_jacobians(mesh, image_xy)
    fz = 0.5 * ((jacobian[:, 0, 0] + jacobian[:, 1, 1]) + 1j * (jacobian[:, 1, 0] - jacobian[:, 0, 1]))
    triangles = source[mesh.faces]
    variation = 0.04 * np.exp(-np.abs(np.mean(triangles, axis=1) - 0.38 - 0.42j) ** 2 / 0.12)
    reference = all_vertex_near_far_bhf_variation(
        source, image, mesh.faces, fz, variation, near_order=16, far_block_size=128
    )
    candidate = np.asarray(np.load(gpu_velocity), dtype=np.complex128)
    error = candidate - reference
    result = {
        "n": n,
        "vertices": int(source.size),
        "faces": int(mesh.n_faces),
        "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-30)),
        "max_abs_error": float(np.max(np.abs(error))),
        "reference_l2": float(np.linalg.norm(reference)),
        "gpu_l2": float(np.linalg.norm(candidate)),
        "finite": bool(np.all(np.isfinite(candidate))),
        "scope": "GPU float32 BHF assembly versus NumPy Duffy near/far reference",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-velocity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.gpu_velocity, args.output, args.n), indent=2))


if __name__ == "__main__":
    main()
