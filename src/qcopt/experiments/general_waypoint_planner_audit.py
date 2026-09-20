"""Adaptive path-parameter waypoint planner for arbitrary injective targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcopt.forward.safe_step import certified_step_toward_target
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def _target(vertices: np.ndarray, parameter: float) -> np.ndarray:
    """A smooth injective path: rotation composed with an area-preserving shear."""

    center = np.array([0.5, 0.5])
    angle = np.pi * parameter
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    mapped = (vertices - center) @ rotation.T + center
    amplitude = 0.12 * parameter
    mapped[:, 0] += amplitude * np.sin(2.0 * np.pi * mapped[:, 1])
    return np.ascontiguousarray(mapped)


def _advance_segment(mesh, current: np.ndarray, target: np.ndarray, margin: float, safety: float, max_steps: int = 96):
    records = []
    for step in range(max_steps):
        remaining_before = float(np.linalg.norm(target - current))
        if remaining_before < 1e-10:
            break
        try:
            current, alpha = certified_step_toward_target(
                mesh, current, target, min_det_margin=margin, safety=safety
            )
        except ValueError as error:
            records.append(
                {
                    "step": step,
                    "alpha": 0.0,
                    "remaining": remaining_before,
                    "min_determinant": audit_injectivity(mesh, current).minimum_signed_area_ratio,
                    "flipped_faces": len(audit_injectivity(mesh, current).flipped_faces),
                    "certified": False,
                    "stalled": str(error),
                }
            )
            break
        report = audit_injectivity(mesh, current)
        records.append(
            {
                "step": step,
                "alpha": alpha,
                "remaining": float(np.linalg.norm(target - current)),
                "min_determinant": report.minimum_signed_area_ratio,
                "flipped_faces": len(report.flipped_faces),
                "certified": report.certified,
            }
        )
        if records[-1]["remaining"] < 1e-10:
            break
    return current, records


def run(output_dir: Path, n: int = 256, initial_parameter_step: float = 1.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = structured_rectangle(n, n)
    start = mesh.vertices.copy()
    final_target = _target(mesh.vertices, 1.0)
    direct_state, direct_records = _advance_segment(mesh, start, final_target, 0.05, 0.99)
    direct = {
        "steps": len(direct_records),
        "final_error": float(np.linalg.norm(direct_state - final_target)),
        "min_determinant": float(min((r["min_determinant"] for r in direct_records), default=1.0)),
        "certified": audit_injectivity(mesh, direct_state).certified,
        "stalled": bool(np.linalg.norm(direct_state - final_target) >= 1e-10),
    }

    current = start.copy()
    parameter = 0.0
    parameter_step = float(initial_parameter_step)
    segments = []
    while parameter < 1.0 - 1e-12:
        candidate_parameter = min(1.0, parameter + parameter_step)
        candidate_target = _target(mesh.vertices, candidate_parameter)
        candidate_state, records = _advance_segment(mesh, current, candidate_target, 0.05, 0.99)
        candidate_error = float(np.linalg.norm(candidate_state - candidate_target))
        if candidate_error >= 1e-10:
            parameter_step *= 0.5
            if parameter_step < 1e-4:
                raise RuntimeError("adaptive waypoint step fell below 1e-4")
            continue
        current = candidate_state
        parameter = candidate_parameter
        segments.append(
            {
                "parameter_start": parameter - parameter_step,
                "parameter_end": parameter,
                "parameter_step": parameter_step,
                "steps": len(records),
                "records": records,
                "final_error": candidate_error,
                "min_determinant": float(min((r["min_determinant"] for r in records), default=1.0)),
                "flipped_faces": int(max((r["flipped_faces"] for r in records), default=0)),
            }
        )
    final_report = audit_injectivity(mesh, current)
    result = {
        "mesh": f"{n}x{n} cells / {mesh.n_faces} faces",
        "direct_one_segment": direct,
        "adaptive_segments": segments,
        "segment_count": len(segments),
        "final_parameter": parameter,
        "final_target_error": float(np.linalg.norm(current - final_target)),
        "final_min_determinant": final_report.minimum_signed_area_ratio,
        "final_flipped_faces": len(final_report.flipped_faces),
        "final_certified": final_report.certified,
        "scope": "adaptive determinant-certified continuation along a smooth injective rotation-plus-shear target path",
        "limitation": "path-dependent planner control; arbitrary target homotopy selection and nonsmooth active-set theorem remain open",
    }
    (output_dir / "general_waypoint_planner_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--initial-parameter-step", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.initial_parameter_step), indent=2))


if __name__ == "__main__":
    main()
