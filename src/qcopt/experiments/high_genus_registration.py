"""Reproducible genus-2 and official high-genus registration experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..common_surface_map import sample_common_refinement
from ..surface_atlas import SurfacePoint
from ..surface_audit import audit_located_common_map_flow
from ..surface_benchmarks import (
    deterministic_face_samples,
    make_double_torus,
    random_smooth_tangent_field,
    select_farthest_landmarks,
)
from ..surface_flow import (
    FlowHistory,
    FlowStep,
    IntrinsicFaceField,
    required_substeps_for_cfl,
)
from ..surface_history_io import save_flow_history
from ..surface_mesh import (
    CommonRefinement,
    CommonRefinementRepair,
    SurfaceMesh,
    load_common_refinement,
    load_obj,
)
from ..surface_registration import (
    RegistrationConfig,
    RegistrationResult,
    register_surface_map,
)

OFFICIAL_ARCHIVE_URL = (
    "https://www.graphics.rwth-aachen.de/media/papers/309/"
    "s2020-inter-surface-maps-data.zip"
)
OFFICIAL_ARCHIVE_SHA256 = (
    "4d4e7d4c0ecb33c1be334219de1c6cf46faeec15453d8d2db170911bfbf3c9f2"
)


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    source: SurfaceMesh
    target: SurfaceMesh
    common: CommonRefinement
    source_samples: tuple[SurfacePoint, ...]
    ground_truth_target_points: tuple[SurfacePoint, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class TrialConfig:
    perturbation_amplitude: float = 0.8
    perturbation_smoothing: float = 8.0
    perturbation_cfl: float = 0.15
    landmarks: int = 24
    audit_faces: int = 96
    registration: RegistrationConfig = field(
        default_factory=lambda: RegistrationConfig(
            iterations=12,
            landmark_weight=1.0,
            curvature_weight=1.0,
            smoothing=1.5,
            descriptor_smoothing=0.5,
            max_update_fraction=0.06,
            integration_substeps=8,
            cfl_limit=0.15,
        )
    )


def primary_trial_config() -> TrialConfig:
    """Return the immutable budget used by the 2026-08-28 primary matrix."""

    return TrialConfig(
        perturbation_amplitude=0.5,
        landmarks=20,
        audit_faces=64,
        registration=RegistrationConfig(
            iterations=10,
            landmark_weight=1.0,
            curvature_weight=1.0,
            smoothing=1.5,
            descriptor_smoothing=0.5,
            max_update_fraction=0.05,
            integration_substeps=8,
            cfl_limit=0.15,
        ),
    )


def build_synthetic_genus2_case(
    *, resolution: int = 20, maximum_samples: int = 256
) -> ExperimentCase:
    """Create two differently embedded, identically triangulated genus-2 surfaces."""

    source = make_double_torus(resolution=resolution)
    normalized = source.vertices / max(
        float(np.linalg.norm(np.ptp(source.vertices, axis=0))), 1e-12
    )
    phase = 2.1 * normalized[:, 0] - 1.3 * normalized[:, 1] + 0.7 * normalized[:, 2]
    normal_offset = 0.045 * np.sin(2.0 * np.pi * phase)
    affine = np.array(
        [[1.04, 0.035, 0.0], [0.0, 0.96, 0.025], [0.015, 0.0, 1.07]]
    )
    target_vertices = source.vertices @ affine.T + normal_offset[:, None] * source.vertex_normals
    target = SurfaceMesh(target_vertices, source.faces.copy())
    common = CommonRefinement(
        source,
        target,
        CommonRefinementRepair(0, 0, 0, 0, 0.0, 1e-6),
    )
    source_samples = deterministic_face_samples(source, maximum_samples)
    target_samples = tuple(
        SurfacePoint(point.face, point.barycentric.copy()) for point in source_samples
    )
    return ExperimentCase(
        "synthetic_genus2",
        source,
        target,
        common,
        source_samples,
        target_samples,
        {
            "kind": "generated",
            "generator": "smoothed marching-cubes boundary of a thickened figure-eight graph",
            "resolution": resolution,
            "ground_truth": "identical oriented face and barycentric coordinates",
        },
    )


def load_official_case(
    name: str,
    data_root: str | Path,
    *,
    maximum_samples: int = 256,
) -> ExperimentCase:
    """Load one published real-data pair and its held-out common map."""

    data_root = Path(data_root)
    specifications = {
        "official_genus3": (
            Path("Fig11_genus/3"),
            "1_closed_000400_iters_M_on_A.obj",
            "1_closed_000400_iters_M_on_B.obj",
        ),
        "official_genus5": (
            Path("Fig11_genus/5"),
            "1_closed_000600_iters_M_on_A.obj",
            "1_closed_000600_iters_M_on_B.obj",
        ),
        "official_pretzel_genus3": (
            Path("Fig12_texture/pretzel"),
            "M_on_A.obj",
            "M_on_B.obj",
        ),
    }
    if name not in specifications:
        raise ValueError(f"unknown official case: {name}")
    relative, mapped_source, mapped_target = specifications[name]
    directory = data_root / relative
    source_path, target_path = directory / "A.obj", directory / "B.obj"
    common_source_path = directory / mapped_source
    common_target_path = directory / mapped_target
    source = load_obj(source_path)
    target = load_obj(target_path)
    common = load_common_refinement(common_source_path, common_target_path)
    sampled = sample_common_refinement(
        common,
        source,
        target,
        maximum_samples=maximum_samples,
    )
    return ExperimentCase(
        name,
        source,
        target,
        common,
        sampled.source_points,
        sampled.target_points,
        {
            "kind": "published real mesh pair",
            "publication": "Inter-Surface Maps via Constant-Curvature Metrics",
            "archive_url": OFFICIAL_ARCHIVE_URL,
            "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "relative_directory": str(relative).replace("\\", "/"),
            "source_sha256": _sha256(source_path),
            "target_sha256": _sha256(target_path),
            "common_source_sha256": _sha256(common_source_path),
            "common_target_sha256": _sha256(common_target_path),
            "maximum_source_projection_error": sampled.maximum_source_projection_error,
            "maximum_target_projection_error": sampled.maximum_target_projection_error,
            "quantization_repair": asdict(common.repair),
            "landmark_status": "synthetic landmarks sampled from published common map",
        },
    )


def run_trial(
    case: ExperimentCase,
    *,
    seed: int,
    mode: str,
    output_dir: str | Path,
    config: TrialConfig | None = None,
) -> dict[str, Any]:
    """Perturb a certified map, refine it, audit it, and persist all evidence."""

    if mode not in {"landmark", "curvature", "combined"}:
        raise ValueError("mode must be landmark, curvature, or combined")
    config = config or TrialConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.landmarks < 1 or config.landmarks > len(case.source_samples):
        raise ValueError("landmark count is incompatible with sample count")

    trial_start = perf_counter()
    ambient = random_smooth_tangent_field(
        case.target,
        seed=seed,
        relative_amplitude=config.perturbation_amplitude,
        smoothing=config.perturbation_smoothing,
    )
    perturbation_field = IntrinsicFaceField.from_vertex_field(case.target, ambient)
    perturbation_substeps = required_substeps_for_cfl(
        case.target,
        perturbation_field,
        duration=1.0,
        cfl_limit=config.perturbation_cfl,
    )
    perturbation = FlowHistory(
        (FlowStep(perturbation_field, 1.0, perturbation_substeps),)
    )
    initial_advection = perturbation.apply(
        case.target, case.ground_truth_target_points
    )
    initial_points = initial_advection.points
    landmark_indices = select_farthest_landmarks(
        case.target, case.ground_truth_target_points, config.landmarks
    )
    landmark_targets = tuple(
        case.ground_truth_target_points[index] for index in landmark_indices
    )
    weights = {
        "landmark": (1.0, 0.0),
        "curvature": (0.0, 1.0),
        "combined": (1.0, 1.0),
    }[mode]
    registration_config = replace(
        config.registration,
        landmark_weight=weights[0],
        curvature_weight=weights[1],
    )
    optimization_start = perf_counter()
    result = register_surface_map(
        case.source,
        case.target,
        case.source_samples,
        initial_points,
        landmark_indices=landmark_indices,
        landmark_targets=landmark_targets,
        config=registration_config,
    )
    optimization_seconds = perf_counter() - optimization_start
    complete_history = FlowHistory(
        perturbation.steps + result.flow_history.steps
    )
    audit_start = perf_counter()
    audit = audit_located_common_map_flow(
        case.common,
        case.source,
        case.target,
        complete_history,
        max_audit_faces=config.audit_faces,
        cfl_limit=max(config.perturbation_cfl, registration_config.cfl_limit),
    )
    audit_seconds = perf_counter() - audit_start

    initial_dense = _point_rmse(
        case.target, initial_points, case.ground_truth_target_points
    )
    final_dense = _point_rmse(
        case.target, result.points, case.ground_truth_target_points
    )
    normalization = case.target.median_edge_length
    records = [asdict(record) for record in result.iterations]
    monotone = all(
        later["objective"] <= earlier["objective"] + 1e-12
        for earlier, later in zip(records, records[1:])
    )
    expected_residual_improved = bool(
        (mode == "curvature" and result.final_curvature_rmse <= result.initial_curvature_rmse)
        or (mode == "landmark" and result.final_landmark_rmse <= result.initial_landmark_rmse)
        or (
            mode == "combined"
            and result.final_landmark_rmse <= result.initial_landmark_rmse
            and result.final_curvature_rmse <= result.initial_curvature_rmse
        )
    )
    valid_run = bool(audit.certified and monotone and expected_residual_improved)
    registration_success = bool(valid_run and final_dense < initial_dense)
    metrics: dict[str, Any] = {
        "case": case.name,
        "genus": case.source.topology.genus,
        "mode": mode,
        "sample_count": len(case.source_samples),
        "landmark_count": len(landmark_indices),
        "source": _mesh_metrics(case.source),
        "target": _mesh_metrics(case.target),
        "common_refinement": _mesh_metrics(case.common.source),
        "provenance": case.provenance,
        "configuration": {
            "seed": seed,
            "trial": _jsonable(asdict(config)),
            "registration": _jsonable(asdict(registration_config)),
            "perturbation_substeps": perturbation_substeps,
        },
        "initial": {
            "dense_rmse": initial_dense,
            "dense_rmse_median_edges": initial_dense / normalization,
            "landmark_rmse": result.initial_landmark_rmse,
            "landmark_rmse_median_edges": result.initial_landmark_rmse / normalization,
            "curvature_rmse": result.initial_curvature_rmse,
            "chart_transitions": initial_advection.transition_count,
        },
        "final": {
            "dense_rmse": final_dense,
            "dense_rmse_median_edges": final_dense / normalization,
            "dense_improvement_fraction": (initial_dense - final_dense)
            / max(initial_dense, 1e-15),
            "landmark_rmse": result.final_landmark_rmse,
            "landmark_rmse_median_edges": result.final_landmark_rmse / normalization,
            "curvature_rmse": result.final_curvature_rmse,
            "accepted_steps": len(result.flow_history.steps),
            "optimization_chart_transitions": result.total_chart_transitions,
            "convergence_reason": result.convergence_reason,
        },
        "iterations": records,
        "objective_monotone": monotone,
        "expected_residual_improved": expected_residual_improved,
        "certificate": _jsonable(asdict(audit)),
        "valid_run": valid_run,
        "registration_success": registration_success,
        # Backward-compatible alias; it means numerical/optimization validity,
        # not agreement with the held-out correspondence.
        "accepted": valid_run,
        "timing_seconds": {
            "optimization": optimization_seconds,
            "audit": audit_seconds,
            "total": perf_counter() - trial_start,
        },
    }
    save_flow_history(output_dir / "flow_history.npz", complete_history)
    save_flow_history(output_dir / "correction_history.npz", result.flow_history)
    _save_surface_points(
        output_dir / "surface_points.npz",
        case.source_samples,
        case.ground_truth_target_points,
        initial_points,
        result.points,
        landmark_indices,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot_correspondence(
        case,
        initial_points,
        result.points,
        landmark_indices,
        output_dir / "correspondence.png",
    )
    _plot_objective(result, output_dir / "objective.png")
    return metrics


def run_suite(
    *,
    data_root: str | Path,
    output_root: str | Path,
    cases: tuple[str, ...],
    seeds: tuple[int, ...],
    modes: tuple[str, ...],
    maximum_samples: int,
    config: TrialConfig,
) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, ExperimentCase] = {}
    for case_name in cases:
        loaded[case_name] = (
            build_synthetic_genus2_case(maximum_samples=maximum_samples)
            if case_name == "synthetic_genus2"
            else load_official_case(
                case_name, data_root, maximum_samples=maximum_samples
            )
        )
    metrics: list[dict[str, Any]] = []
    for case_name in cases:
        for seed in seeds:
            for mode in modes:
                directory = output_root / case_name / f"seed_{seed:03d}" / mode
                metrics.append(
                    run_trial(
                        loaded[case_name],
                        seed=seed,
                        mode=mode,
                        output_dir=directory,
                        config=config,
                    )
                )
    _write_summary(output_root, metrics)
    return metrics


def _mesh_metrics(mesh: SurfaceMesh) -> dict[str, Any]:
    return {
        "vertices": mesh.n_vertices,
        "faces": mesh.n_faces,
        "edges": mesh.topology.edges,
        "euler_characteristic": mesh.topology.euler_characteristic,
        "genus": mesh.topology.genus,
        "closed": mesh.topology.closed,
        "connected": mesh.topology.connected,
        "oriented": mesh.topology.oriented,
        "minimum_edge_length": mesh.minimum_edge_length,
        "median_edge_length": mesh.median_edge_length,
    }


def _point_rmse(
    mesh: SurfaceMesh,
    first: tuple[SurfacePoint, ...],
    second: tuple[SurfacePoint, ...],
) -> float:
    first_xyz = np.vstack([point.position(mesh) for point in first])
    second_xyz = np.vstack([point.position(mesh) for point in second])
    return float(np.sqrt(np.mean(np.sum((first_xyz - second_xyz) ** 2, axis=1))))


def _save_surface_points(
    path: Path,
    source: tuple[SurfacePoint, ...],
    truth: tuple[SurfacePoint, ...],
    initial: tuple[SurfacePoint, ...],
    final: tuple[SurfacePoint, ...],
    landmarks: np.ndarray,
) -> None:
    payload: dict[str, np.ndarray] = {"landmark_indices": landmarks}
    for name, points in (
        ("source", source),
        ("truth", truth),
        ("initial", initial),
        ("final", final),
    ):
        payload[f"{name}_faces"] = np.asarray([point.face for point in points])
        payload[f"{name}_barycentric"] = np.vstack(
            [point.barycentric for point in points]
        )
    np.savez_compressed(path, **payload)


def _plot_correspondence(
    case: ExperimentCase,
    initial: tuple[SurfacePoint, ...],
    final: tuple[SurfacePoint, ...],
    landmarks: np.ndarray,
    path: Path,
) -> None:
    figure = plt.figure(figsize=(14, 4), constrained_layout=True)
    source_xyz = np.vstack([point.position(case.source) for point in case.source_samples])
    truth_xyz = np.vstack(
        [point.position(case.target) for point in case.ground_truth_target_points]
    )
    initial_xyz = np.vstack([point.position(case.target) for point in initial])
    final_xyz = np.vstack([point.position(case.target) for point in final])
    colors = np.linspace(0.0, 1.0, len(source_xyz))
    panels = (
        (case.source, source_xyz, "source samples"),
        (case.target, truth_xyz, "held-out target truth"),
        (case.target, initial_xyz, "perturbed initialization"),
        (case.target, final_xyz, "refined map"),
    )
    for index, (mesh, points, title) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 4, index, projection="3d")
        _draw_mesh(axis, mesh)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            cmap="turbo",
            s=8,
            depthshade=False,
        )
        if index > 1:
            landmark_xyz = points[landmarks]
            axis.scatter(
                landmark_xyz[:, 0],
                landmark_xyz[:, 1],
                landmark_xyz[:, 2],
                c="black",
                s=18,
                marker="x",
                depthshade=False,
            )
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()
        _equal_3d(axis, mesh.vertices)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _draw_mesh(axis, mesh: SurfaceMesh) -> None:
    maximum_faces = 3500
    step = max(1, int(np.ceil(mesh.n_faces / maximum_faces)))
    triangles = mesh.vertices[mesh.faces[::step]]
    collection = Poly3DCollection(
        triangles,
        facecolor=(0.83, 0.85, 0.88, 0.18),
        edgecolor=(0.25, 0.28, 0.32, 0.16),
        linewidth=0.15,
    )
    axis.add_collection3d(collection)


def _equal_3d(axis, vertices: np.ndarray) -> None:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.52 * max(float(np.max(maximum - minimum)), 1e-12)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _plot_objective(result: RegistrationResult, path: Path) -> None:
    iterations = np.asarray([record.iteration for record in result.iterations])
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    axes[0].semilogy(
        iterations,
        np.maximum([record.objective for record in result.iterations], 1e-16),
        marker="o",
        markersize=3,
    )
    axes[0].set(title="normalized objective", xlabel="accepted iteration")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        iterations,
        [record.landmark_rmse for record in result.iterations],
        label="landmark RMSE",
    )
    axes[1].plot(
        iterations,
        [record.curvature_rmse for record in result.iterations],
        label="curvature RMSE",
    )
    axes[1].set(title="raw residuals", xlabel="accepted iteration")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_summary(output_root: Path, metrics: list[dict[str, Any]]) -> None:
    (output_root / "summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    columns = (
        "case",
        "genus",
        "mode",
        "seed",
        "initial_dense_edges",
        "final_dense_edges",
        "dense_improvement_fraction",
        "initial_landmark_edges",
        "final_landmark_edges",
        "initial_curvature_rmse",
        "final_curvature_rmse",
        "certified",
        "valid_run",
        "registration_success",
        "runtime_seconds",
    )
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "case": item["case"],
                    "genus": item["genus"],
                    "mode": item["mode"],
                    "seed": item["configuration"]["seed"],
                    "initial_dense_edges": item["initial"]["dense_rmse_median_edges"],
                    "final_dense_edges": item["final"]["dense_rmse_median_edges"],
                    "dense_improvement_fraction": item["final"]["dense_improvement_fraction"],
                    "initial_landmark_edges": item["initial"]["landmark_rmse_median_edges"],
                    "final_landmark_edges": item["final"]["landmark_rmse_median_edges"],
                    "initial_curvature_rmse": item["initial"]["curvature_rmse"],
                    "final_curvature_rmse": item["final"]["curvature_rmse"],
                    "certified": item["certificate"]["certified"],
                    "valid_run": item["valid_run"],
                    "registration_success": item["registration_success"],
                    "runtime_seconds": item["timing_seconds"]["total"],
                }
            )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("primary-2026-08-28", "library-default"),
        default="primary-2026-08-28",
        help="fixed primary budget or the generic TrialConfig defaults",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("external_data/s2020-intersurfacemaps-data"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/high_genus_registration")
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "synthetic_genus2",
            "official_genus3",
            "official_genus5",
            "official_pretzel_genus3",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 11, 29])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("landmark", "curvature", "combined"),
        default=["landmark", "curvature", "combined"],
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--audit-faces", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    base = (
        primary_trial_config()
        if arguments.preset == "primary-2026-08-28"
        else TrialConfig()
    )
    config = replace(
        base,
        audit_faces=(
            arguments.audit_faces
            if arguments.audit_faces is not None
            else base.audit_faces
        ),
        registration=replace(
            base.registration,
            iterations=(
                arguments.iterations
                if arguments.iterations is not None
                else base.registration.iterations
            ),
        ),
    )
    maximum_samples = (
        arguments.samples
        if arguments.samples is not None
        else (192 if arguments.preset == "primary-2026-08-28" else 256)
    )
    run_suite(
        data_root=arguments.data_root,
        output_root=arguments.output,
        cases=tuple(arguments.cases),
        seeds=tuple(arguments.seeds),
        modes=tuple(arguments.modes),
        maximum_samples=maximum_samples,
        config=config,
    )


if __name__ == "__main__":
    main()
