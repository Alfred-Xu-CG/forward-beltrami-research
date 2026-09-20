"""Common-target 256² comparison for direct barriers and matrix-free KKT.

The matrix-free KKT route uses a zero-boundary smooth target.  Earlier
high-resolution direct-map controls used a different smooth-twist target,
which was useful for scaling but not a fair solver comparison.  This audit
reuses the KKT target, identity initialization, mesh, and independent
injectivity audit for the direct-map baselines.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.experiments.barrier_kkt_implicit_audit import _target
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle
from qcopt.optimize import run_direct_optimization


def run(output_dir: Path, n: int = 256, iterations: int = 4, amplitude: float = 0.08) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    initial = mesh.vertices.copy()
    target = _target(torch.as_tensor(np.array(mesh.vertices, copy=True), dtype=torch.double), amplitude).numpy()
    target_t = torch.as_tensor(target, dtype=torch.double)
    records: list[dict[str, object]] = []
    for method in ("none", "lim_style", "slim_style", "amips_style"):
        def loss(uv: torch.Tensor) -> torch.Tensor:
            return torch.mean((uv - target_t) ** 2)

        started = time.perf_counter()
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
        tri = result.uv[mesh.faces]
        det = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        records.append(
            {
                "method": method,
                "iterations": iterations,
                "elapsed_seconds": result.elapsed_seconds,
                "wall_seconds": time.perf_counter() - started,
                "target_amplitude": amplitude,
                "final_data_loss": float(np.mean((result.uv - target) ** 2)),
                "min_determinant": float(np.min(det)),
                "flipped_faces": int(np.sum(det <= 0.0)),
                "certified": bool(report.certified),
                "accepted_fraction": float(np.mean([bool(item["accepted"]) for item in result.history])),
                "max_backtracks": int(max(int(item["backtracks"]) for item in result.history)),
            }
        )
    result = {
        "grid": f"{n}x{n} cells / {mesh.n_faces} faces",
        "target_definition": "barrier_kkt_implicit_audit._target(vertices, amplitude)",
        "target_amplitude": amplitude,
        "records": records,
        "scope": "same target, initialization, mesh, budget, and independent topology audit as matrix-free KKT control",
        "limitation": "direct baselines are finite-step Adam controls; KKT backward accuracy and active-set smoothness remain separate",
    }
    (output_dir / "barrier_common_target_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--amplitude", type=float, default=0.08)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.iterations, args.amplitude), indent=2))


if __name__ == "__main__":
    main()
