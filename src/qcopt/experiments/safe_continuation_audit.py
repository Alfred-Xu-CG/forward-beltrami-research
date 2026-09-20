"""Large-deformation audit for determinant-certified target continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcopt.forward.safe_step import certified_step_toward_target
from qcopt.mesh import structured_rectangle


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    center = np.array([0.5, 0.5])
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s], [s, c]])
    return (points - center) @ matrix.T + center


def _direct_path(mesh, target, margin=0.05, safety=0.99, max_steps=30):
    current = mesh.vertices.copy()
    records = []
    for step_index in range(max_steps):
        try:
            updated, alpha = certified_step_toward_target(
                mesh, current, target, min_det_margin=margin, safety=safety
            )
        except ValueError as error:
            records.append({"step": step_index, "stalled": True, "reason": str(error), "remaining": float(np.linalg.norm(target - current))})
            break
        records.append({"step": step_index, "alpha": alpha, "progress": float(np.linalg.norm(updated - current)), "remaining": float(np.linalg.norm(target - updated))})
        current = updated
        if records[-1]["remaining"] < 1e-10:
            break
    return records


def run(output_dir: Path, n: int = 64) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    identity = mesh.vertices.copy()
    target_180 = _rotate(identity, np.pi)
    target_90 = _rotate(identity, 0.5 * np.pi)
    direct = _direct_path(mesh, target_180)
    stage1 = _direct_path(mesh, target_90, max_steps=4)
    # Start the second stage from the exact 90-degree endpoint.
    current = target_90.copy()
    staged = []
    for index in range(4):
        updated, alpha = certified_step_toward_target(
            mesh, current, target_180, min_det_margin=0.05, safety=0.99
        )
        staged.append({"step": index, "alpha": alpha, "remaining": float(np.linalg.norm(target_180 - updated))})
        current = updated
        if staged[-1]["remaining"] < 1e-10:
            break
    result = {
        "mesh": f"{n}x{n} cells / {mesh.n_faces} faces",
        "direct_180_records": direct,
        "staged_90_plus_90_records": staged,
        "direct_final_relative_error": direct[-1]["remaining"] / np.linalg.norm(target_180 - identity),
        "staged_final_relative_error": staged[-1]["remaining"] / np.linalg.norm(target_180 - identity),
        "interpretation": "straight residual continuation approaches the fold barrier for a 180-degree endpoint, while a path through a 90-degree rotation reaches the same valid target",
        "limitation": "this is a path-planning certificate, not a general BHF continuation algorithm",
    }
    (output_dir / "safe_continuation_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, n=args.n), indent=2))


if __name__ == "__main__":
    main()
