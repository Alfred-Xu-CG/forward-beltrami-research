"""Common high-resolution direct-map barrier comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.forward.benchmarks import smooth_twist_map
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle
from qcopt.optimize import run_direct_optimization


def run(output_dir: Path, n: int = 64, iterations: int = 24) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    initial = mesh.vertices.copy()
    target = smooth_twist_map(mesh.vertices, amplitude=0.18)
    target_t = torch.as_tensor(target, dtype=torch.double)
    records = []
    for method in ("none", "lim_style", "slim_style", "amips_style"):
        def loss(uv: torch.Tensor) -> torch.Tensor:
            return torch.mean((uv - target_t) ** 2)

        t0 = time.perf_counter()
        result = run_direct_optimization(
            mesh,
            initial,
            loss,
            method=method,
            iterations=iterations,
            learning_rate=0.03,
            barrier_weight=0.02,
            rectangle=True,
            max_backtracks=12,
        )
        report = audit_injectivity(mesh, result.uv, rectangle=True)
        determinants = np.linalg.det(
            np.stack(
                (
                    result.uv[mesh.faces[:, 1]] - result.uv[mesh.faces[:, 0]],
                    result.uv[mesh.faces[:, 2]] - result.uv[mesh.faces[:, 0]],
                ),
                axis=-1,
            )
        )
        records.append({
            "method": method,
            "iterations": iterations,
            "elapsed_seconds": result.elapsed_seconds,
            "wall_seconds": time.perf_counter() - t0,
            "final_data_loss": float(np.mean((result.uv - target) ** 2)),
            "min_determinant": float(np.min(determinants)),
            "flipped_faces": int(np.sum(determinants <= 0.0)),
            "certified": bool(report.certified),
            "accepted_fraction": float(np.mean([bool(item["accepted"]) for item in result.history])),
            "max_backtracks": int(max(int(item["backtracks"]) for item in result.history)),
        })
    result = {
        "grid": f"{n}x{n} cells / {mesh.n_faces} faces",
        "records": records,
        "scope": "same target, initialization, optimizer budget, and independent topology audit across direct-map barriers",
        "limitation": "not an implicit KKT backward or a universal optimizer ranking; this is a high-resolution safeguard/compute comparison",
    }
    (output_dir / "barrier_highres_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.iterations), indent=2))


if __name__ == "__main__":
    main()

