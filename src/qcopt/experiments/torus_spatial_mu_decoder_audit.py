"""Audit a spatial Beltrami-to-positive-weight torus decoder.

The rectangular periodic Tutte solver only exposes axis-aligned positive edge
weights.  This experiment therefore tests two deliberately separated cases:
an exactly representable separable map and a shear field whose conductivity
has an off-diagonal component.  The latter is a control showing that positive
axis weights are not an arbitrary spatially varying Beltrami decoder.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.torus_tutte import periodic_torus_face_determinants, periodic_tutte_embedding


def _lifted_cells(values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return p00,p10,p01 for all periodic cells, with seam lifts."""
    p00 = values.reshape(n, n, 2)
    p10 = np.roll(p00, -1, axis=1).copy()
    p01 = np.roll(p00, -1, axis=0).copy()
    p10[:, -1] += np.array([1.0, 0.0])
    p01[-1, :] += np.array([0.0, 1.0])
    return p00, p10, p01


def _face_mu(values: np.ndarray, n: int) -> np.ndarray:
    p00, p10, p01 = _lifted_cells(values, n)
    fx = (p10 - p00) * n
    fy = (p01 - p00) * n
    z_x = fx[..., 0] + 1j * fx[..., 1]
    z_y = fy[..., 0] + 1j * fy[..., 1]
    fz = 0.5 * (z_x - 1j * z_y)
    fzb = 0.5 * (z_x + 1j * z_y)
    return (fzb / fz).reshape(-1)


def _separable_target(n: int, ax: float, ay: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float64) / n
    x = t + ax * np.sin(2.0 * np.pi * t) / (2.0 * np.pi)
    y = t + ay * np.sin(2.0 * np.pi * t) / (2.0 * np.pi)
    dx = np.roll(x, -1) - x
    dx[-1] += 1.0
    dy = np.roll(y, -1) - y
    dy[-1] += 1.0
    wx = np.broadcast_to(1.0 / dx[None, :], (n, n)).copy()
    wy = np.broadcast_to(1.0 / dy[:, None], (n, n)).copy()
    target = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1).reshape(-1, 2)
    return target, wx, wy


def _shear_target(n: int, amplitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(t, t, indexing="xy")
    target = np.stack((xx + amplitude * np.sin(2.0 * np.pi * yy) / (2.0 * np.pi), yy), axis=-1).reshape(-1, 2)
    # Axis weights cannot encode the off-diagonal conductivity; this is the
    # positive isotropic control used to quantify the resulting mismatch.
    wx = np.ones((n, n), dtype=np.float64)
    wy = np.ones((n, n), dtype=np.float64)
    return target, wx, wy


def _case(name: str, n: int, target: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> dict:
    t0 = time.perf_counter()
    solved = periodic_tutte_embedding(n, n, wx, wy)
    elapsed = time.perf_counter() - t0
    target_mu = _face_mu(target, n)
    solved_mu = _face_mu(solved, n)
    det = periodic_torus_face_determinants(solved, n, n)
    return {
        "name": name,
        "vertices": n * n,
        "periodic_faces": 2 * n * n,
        "elapsed_seconds": elapsed,
        "target_mu_rms": float(np.sqrt(np.mean(np.abs(target_mu) ** 2))),
        "mu_relative_l2_error": float(np.linalg.norm(solved_mu - target_mu) / max(np.linalg.norm(target_mu), 1e-15)),
        "mu_absolute_linf_error": float(np.max(np.abs(solved_mu - target_mu))),
        "map_relative_l2_error": float(np.linalg.norm(solved - target) / max(np.linalg.norm(target), 1e-15)),
        "min_lifted_face_determinant": float(np.min(det)),
        "flipped_faces": int(np.sum(det <= 0.0)),
        "finite": bool(np.all(np.isfinite(solved_mu)) and np.all(np.isfinite(det))),
    }


def run(output_dir: Path, n: int = 256) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    sep_target, sep_wx, sep_wy = _separable_target(n, 0.28, -0.22)
    shear_target, shear_wx, shear_wy = _shear_target(n, 0.30)
    result = {
        "grid": f"{n}x{n} periodic vertices",
        "cases": [
            _case("separable_axis_aligned", n, sep_target, sep_wx, sep_wy),
            _case("off_diagonal_shear_control", n, shear_target, shear_wx, shear_wy),
        ],
        "scope": "spatial Beltrami decoder control for positive axis-aligned torus weights",
        "interpretation": "the separable subclass is recovered to solver precision, while the shear control exposes the missing off-diagonal/triangular coupling",
        "limitation": "not an arbitrary triangulated-torus QC decoder and not evidence of general spatial Beltrami inversion",
    }
    (output_dir / "torus_spatial_mu_decoder_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n), indent=2))


if __name__ == "__main__":
    main()
