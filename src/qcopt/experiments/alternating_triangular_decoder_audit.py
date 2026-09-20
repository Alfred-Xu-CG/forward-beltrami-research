"""High-resolution non-triangular coupling via alternating triangular layers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.coupled_decoder import triangular_spatial_monotone_inverse, triangular_spatial_monotone_map
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def _reverse_triangular_map(points: np.ndarray, y_increments: np.ndarray, x_increments: np.ndarray) -> np.ndarray:
    swapped = triangular_spatial_monotone_map(points[:, ::-1], y_increments, x_increments)
    return swapped[:, ::-1]


def _reverse_triangular_inverse(points: np.ndarray, y_increments: np.ndarray, x_increments: np.ndarray) -> np.ndarray:
    swapped = triangular_spatial_monotone_inverse(points[:, ::-1], y_increments, x_increments)
    return swapped[:, ::-1]


def run(output_dir: Path, n: int = 512, layers: int = 4) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260919)
    mesh = structured_rectangle(n, n)
    points = mesh.vertices.copy()
    layer_data = []
    for layer in range(layers):
        scalar = np.exp(0.10 * rng.normal(size=32))
        conditional = np.exp(0.10 * rng.normal(size=(32, 64)))
        layer_data.append((scalar, conditional))
    started = time.perf_counter()
    mapped = points.copy()
    for layer, (scalar, conditional) in enumerate(layer_data):
        if layer % 2 == 0:
            mapped = triangular_spatial_monotone_map(mapped, scalar, conditional)
        else:
            mapped = _reverse_triangular_map(mapped, scalar, conditional)
    recovered = mapped.copy()
    for layer in range(layers - 1, -1, -1):
        scalar, conditional = layer_data[layer]
        if layer % 2 == 0:
            recovered = triangular_spatial_monotone_inverse(recovered, scalar, conditional)
        else:
            recovered = _reverse_triangular_inverse(recovered, scalar, conditional)
    report = audit_injectivity(mesh, mapped, rectangle=True)
    vectors = np.stack(
        (mapped[mesh.faces[:, 1]] - mapped[mesh.faces[:, 0]], mapped[mesh.faces[:, 2]] - mapped[mesh.faces[:, 0]]),
        axis=-1,
    )
    result = {
        "grid": f"{n}x{n} cells / {mesh.n_faces} faces",
        "vertices": int(len(points)),
        "layers": layers,
        "control_rows": 32,
        "control_intervals": 64,
        "max_round_trip_error": float(np.max(np.abs(recovered - points))),
        "min_sampled_face_determinant": float(np.min(np.linalg.det(vectors))),
        "flipped_faces": int(len(report.flipped_faces)),
        "injectivity_certified_on_sampled_mesh": bool(report.certified),
        "finite": bool(np.all(np.isfinite(mapped)) and np.all(np.isfinite(recovered))),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "alternating positive triangular spatial layers, giving non-triangular cross-direction coupling",
        "limitation": "composition is hard-bijective analytically, but arbitrary-QC expressivity and a mesh-independent sampled-P1 theorem remain open",
    }
    (output_dir / "alternating_triangular_decoder_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.layers), indent=2))


if __name__ == "__main__":
    main()
