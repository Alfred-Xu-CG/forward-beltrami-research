"""Independent recomputation of saved experiment metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..adjoint import lbs_mu_vjp
from ..beltrami import face_beltrami, face_jacobians, qc_dilation
from ..constraints import fixed_vertex_constraints
from ..injectivity import audit_injectivity
from ..lbs import solve_lbs
from ..mesh import TriMesh, structured_rectangle
from ..registration import i_field, registration_loss, s_field, soft_dice


def audit_artifacts(root_directory: str | Path) -> dict[str, object]:
    root = Path(root_directory)
    mismatches: list[str] = []
    checks: dict[str, object] = {}
    try:
        checks["solver_recheck"] = _audit_solver(root / "solver_validation", mismatches)
        checks["i_to_s"] = _audit_i_to_s(root / "i_to_s", mismatches)
        checks["density"] = _audit_density(root / "density", mismatches)
        checks["multichart"] = _audit_multichart(root / "multichart", mismatches)
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        mismatches.append(f"audit exception: {type(error).__name__}: {error}")
    report: dict[str, object] = {
        "status": "VERIFIED" if not mismatches else "FAILED",
        "mismatches": mismatches,
        "checks": checks,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _compare(mismatches, label, recomputed, saved, atol=1e-8, rtol=1e-6):
    if isinstance(recomputed, (bool, np.bool_)):
        if bool(recomputed) != bool(saved):
            mismatches.append(f"{label}: recomputed={recomputed}, saved={saved}")
        return
    if np.isinf(recomputed) and np.isinf(saved):
        return
    if not np.isclose(recomputed, saved, atol=atol, rtol=rtol):
        mismatches.append(f"{label}: recomputed={recomputed}, saved={saved}")


def _audit_solver(directory, mismatches):
    saved = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    if not all(saved["acceptance"].values()):
        mismatches.append("solver_validation.acceptance contains false")
    mesh = structured_rectangle(3, 3)
    matrix = np.array([[1.25, 0.17], [-0.08, 0.82]])
    target = mesh.vertices @ matrix.T + np.array([0.1, -0.2])
    mu = face_beltrami(mesh, target)
    boundary = mesh.boundary_loops[0]
    constraints = fixed_vertex_constraints(mesh.n_vertices, boundary, target[boundary])
    result = solve_lbs(mesh, mu, constraints)
    rng = np.random.default_rng(404)
    uv_bar = rng.normal(size=result.uv.shape)
    adjoint = lbs_mu_vjp(mesh, mu, result, uv_bar)
    direction = rng.normal(size=(mesh.n_faces, 2))
    epsilon = 1e-6
    complex_direction = direction[:, 0] + 1j * direction[:, 1]
    finite = (
        np.sum(solve_lbs(mesh, mu + epsilon * complex_direction, constraints).uv * uv_bar)
        - np.sum(solve_lbs(mesh, mu - epsilon * complex_direction, constraints).uv * uv_bar)
    ) / (2.0 * epsilon)
    predicted = np.sum(adjoint.gradient * direction)
    relative_error = abs(finite - predicted) / max(1.0, abs(finite), abs(predicted))
    evidence = {
        "reconstruction_error": float(np.max(np.abs(result.uv - target))),
        "forward_residual": result.primal_residual,
        "adjoint_residual": adjoint.residual,
        "directional_gradient_relative_error": float(relative_error),
    }
    for key, value in evidence.items():
        threshold = 1e-5 if "gradient" in key else 1e-10
        if value >= threshold:
            mismatches.append(f"solver_recheck.{key}={value} exceeds {threshold}")
    return evidence


def _audit_i_to_s(directory, mismatches):
    payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    arrays = np.load(directory / "maps.npz")
    mesh = TriMesh(arrays["vertices"], arrays["faces"])
    source = i_field(torch.tensor(np.array(mesh.vertices, copy=True), dtype=torch.double))
    checked = 0
    for item in payload["methods"]:
        method = item["method"]
        uv = arrays[method]
        report = audit_injectivity(mesh, uv, rectangle=method != "mu_lsqc_free")
        warped = s_field(torch.tensor(uv, dtype=torch.double))
        values = {
            "data_loss": float(registration_loss(source, warped)),
            "soft_dice": float(soft_dice(source, warped)),
            "flipped_faces": len(report.flipped_faces),
            "minimum_signed_area_ratio": report.minimum_signed_area_ratio,
            "certified": report.certified,
        }
        for key, value in values.items():
            _compare(mismatches, f"i_to_s.{method}.{key}", value, item[key])
        checked += 1
    return {"methods_recomputed": checked}


def _audit_density(directory, mismatches):
    payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    arrays = np.load(directory / "maps.npz")
    mesh = TriMesh(arrays["vertices"], arrays["faces"])
    checked = 0
    for item in payload["results"]:
        target, method = item["target"], item["method"]
        uv = arrays[f"{target}__{method}"]
        factors = arrays[f"target_factors__{target}"]
        ratios = np.linalg.det(face_jacobians(mesh, uv))
        report = audit_injectivity(mesh, uv, rectangle=True)
        log_rmse = (
            float(np.sqrt(np.mean((np.log(ratios) - np.log(factors)) ** 2)))
            if np.all(ratios > 0.0)
            else float("inf")
        )
        relative = float(np.sqrt(np.mean(((ratios - factors) / factors) ** 2)))
        values = {
            "log_area_rmse": log_rmse,
            "relative_area_rmse": relative,
            "flipped_faces": len(report.flipped_faces),
            "minimum_signed_area_ratio": report.minimum_signed_area_ratio,
            "certified": report.certified,
        }
        for key, value in values.items():
            _compare(mismatches, f"density.{target}.{method}.{key}", value, item[key])
        checked += 1
    return {"runs_recomputed": checked}


def _audit_multichart(directory, mismatches):
    payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    arrays = np.load(directory / "maps.npz")
    faces = arrays["faces"]
    charts = {
        "c0": TriMesh(arrays["source_c0"], faces),
        "c1": TriMesh(arrays["source_c1"], faces),
    }
    correct_seam = float(
        np.max(np.abs(arrays["correct_c1"] - arrays["correct_c0"] - np.array([1.0, 0.0])))
    )
    wrong_seam = float(
        np.max(np.abs(arrays["wrong_c1"] - arrays["wrong_c0"] - np.array([1.0, 0.0])))
    )
    _compare(
        mismatches,
        "multichart.transition_aware_seam_residual",
        correct_seam,
        payload["transition_aware_seam_residual"],
    )
    _compare(
        mismatches,
        "multichart.raw_coordinate_equality_physical_seam_residual",
        wrong_seam,
        payload["raw_coordinate_equality_physical_seam_residual"],
    )
    for name, mesh in charts.items():
        certificate = audit_injectivity(mesh, arrays[f"correct_{name}"], rectangle=True).certified
        _compare(
            mismatches,
            f"multichart.chart_certificates.{name}",
            certificate,
            payload["chart_certificates"][name],
        )
    return {"correct_seam": correct_seam, "wrong_seam": wrong_seam}
