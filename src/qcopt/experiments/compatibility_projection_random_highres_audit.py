"""Realistic-resolution projection audit for incompatible random facewise mu."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.beltrami import face_beltrami
from qcopt.forward.compatibility_projection import project_facewise_mu
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def _random_target(mesh, rng: np.random.Generator, amplitude: float) -> np.ndarray:
    target = rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces)
    target *= float(amplitude) / max(float(np.max(np.abs(target))), np.finfo(float).eps)
    return np.ascontiguousarray(target.astype(np.complex128))


def run(
    output_dir: Path,
    n: int = 256,
    amplitude: float = 0.35,
    max_nfev: int = 8,
    seed: int = 20260919,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    rng = np.random.default_rng(seed)
    target = _random_target(mesh, rng, amplitude)
    initial = mesh.vertices.copy()
    started = time.perf_counter()
    projection = project_facewise_mu(
        mesh,
        target,
        boundary_map=initial,
        initial_map=initial,
        max_nfev=max_nfev,
    )
    elapsed = time.perf_counter() - started
    target_norm = float(np.linalg.norm(np.concatenate((target.real, target.imag))))
    relative_residual = float(projection.residual_norm / max(target_norm, np.finfo(float).eps))
    mapped = projection.map
    report = audit_injectivity(mesh, mapped, rectangle=True)
    record = {
        "grid": f"{n}x{n} cells",
        "faces": int(mesh.n_faces),
        "target_amplitude": float(np.max(np.abs(target))),
        "target_norm": target_norm,
        "residual_norm": float(projection.residual_norm),
        "relative_residual": relative_residual,
        "nfev": int(projection.nfev),
        "success": bool(projection.success),
        "elapsed_seconds": float(elapsed),
        "finite": bool(np.all(np.isfinite(mapped))),
        "flipped_faces": int(len(report.flipped_faces)),
        "minimum_signed_area": float(report.minimum_signed_area_ratio),
        "injectivity_certified": bool(report.certified),
        "boundary_orientation_ok": bool(report.boundary_orientation_ok),
    }
    result = {
        "record": record,
        "scope": "random incompatible facewise Beltrami projection on a realistic structured rectangle",
        "limitation": "a finite random-field stress is not a theorem for all incompatible fields or all boundary data",
    }
    (output_dir / "compatibility_projection_random_highres_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--max-nfev", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.amplitude, args.max_nfev, args.seed), indent=2))


if __name__ == "__main__":
    main()
