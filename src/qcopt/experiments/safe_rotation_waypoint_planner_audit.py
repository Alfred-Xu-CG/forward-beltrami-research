"""Waypoint planner control for determinant-certified PL continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcopt.forward.safe_step import certified_step_toward_target
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    center = np.array([0.5, 0.5])
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s], [s, c]])
    return (points - center) @ matrix.T + center


def _segment(mesh, current: np.ndarray, target: np.ndarray, margin: float, safety: float) -> tuple[np.ndarray, list[dict]]:
    records = []
    for step in range(32):
        remaining_before = float(np.linalg.norm(target - current))
        if remaining_before < 1e-12:
            break
        current, alpha = certified_step_toward_target(
            mesh, current, target, min_det_margin=margin, safety=safety
        )
        report = audit_injectivity(mesh, current)
        remaining = float(np.linalg.norm(target - current))
        records.append({
            "step": step,
            "alpha": alpha,
            "remaining": remaining,
            "min_determinant": report.minimum_signed_area_ratio,
            "certified": report.certified,
        })
        if remaining < 1e-12:
            break
    return current, records


def run(output_dir: Path, n: int = 256, waypoint_degrees: tuple[float, ...] = (60.0, 120.0, 180.0)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    current = mesh.vertices.copy()
    segments = []
    for degree in waypoint_degrees:
        target = _rotate(mesh.vertices, np.deg2rad(degree))
        current, records = _segment(mesh, current, target, margin=0.05, safety=0.99)
        segments.append({
            "target_degrees": degree,
            "records": records,
            "final_error": float(np.linalg.norm(current - target)),
            "final_report": audit_injectivity(mesh, current).__dict__,
        })
    result = {
        "mesh": f"{n}x{n} cells / {mesh.n_faces} faces",
        "waypoint_degrees": list(waypoint_degrees),
        "segments": segments,
        "final_target_error": segments[-1]["final_error"],
        "final_certified": segments[-1]["final_report"]["certified"],
        "scope": "determinant-certified waypoint planning for rigid rotations; not a general nonsmooth continuation theorem",
    }
    (output_dir / "safe_rotation_waypoint_planner_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, n=args.n), indent=2))


if __name__ == "__main__":
    main()
