"""Independent necessary-error bound for fixed patch-interior conductances."""

from __future__ import annotations

import argparse
import json

import numpy as np

from phase6_target_quad_convexity import _sampled_target
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle


def audit(side: int, patch_cells: int, family: str, seed: int, sample_count: int) -> dict:
    if (side - 1) % patch_cells:
        raise ValueError("patch size must divide the grid cell count")
    _, _, _, coefficients = make_dataset(
        sample_count, 512, seed, return_coefficients=True, target_family=family
    )
    target = _sampled_target(side, coefficients, 8 if family == "base" else 32).numpy().reshape(sample_count, -1, 2)
    mesh = structured_rectangle(side - 1, side - 1)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    a, b = edges[:, 0], edges[:, 1]
    degree = np.bincount(np.concatenate((a, b)), minlength=side**2)
    index = np.arange(side**2)
    x, y = index % side, index // side
    strict = ((x > 0) & (x < side - 1) & (y > 0) & (y < side - 1)
              & (x % patch_cells != 0) & (y % patch_cells != 0))
    sample_metrics = []
    for mapped in target:
        difference = mapped[a] - mapped[b]
        residual = np.zeros_like(mapped)
        np.add.at(residual, a, difference)
        np.add.at(residual, b, -difference)
        size = np.linalg.norm(residual[strict], axis=1)
        lower_bound = size / (2.0 * degree[strict])
        sample_metrics.append({
            "maximum_fixed_interior_harmonic_residual": float(size.max()),
            "rms_fixed_interior_harmonic_residual": float(np.sqrt(np.mean(size**2))),
            "necessary_maximum_vertex_error_lower_bound": float(lower_bound.max()),
        })
    return {
        "control_side": side,
        "control_vertices": side**2,
        "control_faces": len(faces),
        "active_edges": len(edges),
        "patch_cells": patch_cells,
        "strict_patch_interior_vertices": int(strict.sum()),
        "family": family,
        "coefficient_seed": seed,
        "sample_count": sample_count,
        "sample_metrics": sample_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--patch-cells", type=int, required=True)
    parser.add_argument("--family", choices=("base", "high32"), required=True)
    parser.add_argument("--seed", type=int, default=99317)
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(audit(args.side, args.patch_cells, args.family, args.seed, args.sample_count), sort_keys=True))


if __name__ == "__main__":
    main()
