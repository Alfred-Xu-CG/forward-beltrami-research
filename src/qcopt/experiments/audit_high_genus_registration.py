"""Independent reload-and-recompute audit for high-genus registration runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..surface_atlas import SurfacePoint
from ..surface_audit import audit_located_common_map_flow
from ..surface_geometry import (
    ScreenedFieldSmoother,
    curvature_descriptors,
    interpolate_vertex_field,
)
from ..surface_history_io import load_flow_history
from .high_genus_registration import (
    ExperimentCase,
    build_synthetic_genus2_case,
    load_official_case,
)


def audit_saved_trial(
    run_directory: str | Path,
    *,
    data_root: str | Path = "external_data/s2020-intersurfacemaps-data",
    case: ExperimentCase | None = None,
) -> dict[str, Any]:
    """Reload one run and recompute map states, residuals, and certificate."""

    run_directory = Path(run_directory)
    metrics_path = run_directory / "metrics.json"
    history_path = run_directory / "flow_history.npz"
    states_path = run_directory / "surface_points.npz"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if case is None:
        samples = int(metrics["sample_count"])
        if metrics["case"] == "synthetic_genus2":
            resolution = int(metrics["provenance"]["resolution"])
            case = build_synthetic_genus2_case(
                resolution=resolution, maximum_samples=samples
            )
        else:
            case = load_official_case(
                metrics["case"], data_root, maximum_samples=samples
            )
    with np.load(states_path, allow_pickle=False) as archive:
        source = _load_points(archive, "source")
        truth = _load_points(archive, "truth")
        initial = _load_points(archive, "initial")
        final = _load_points(archive, "final")
        landmarks = np.asarray(archive["landmark_indices"], dtype=np.int64)
    history = load_flow_history(history_path)
    replay = history.apply(case.target, case.ground_truth_target_points)

    source_match = _point_max_error(
        case.source, source, case.source_samples
    ) <= 1e-12 * max(case.source.median_edge_length, 1.0)
    truth_match = _point_max_error(
        case.target, truth, case.ground_truth_target_points
    ) <= 1e-12 * max(case.target.median_edge_length, 1.0)
    replay_error = _point_max_error(case.target, replay.points, final)
    replay_ok = replay_error <= 1e-9 * max(case.target.median_edge_length, 1.0)

    initial_dense = _point_rmse(case.target, initial, truth)
    final_dense = _point_rmse(case.target, final, truth)
    initial_landmark = _point_rmse(
        case.target,
        tuple(initial[index] for index in landmarks),
        tuple(truth[index] for index in landmarks),
    )
    final_landmark = _point_rmse(
        case.target,
        tuple(final[index] for index in landmarks),
        tuple(truth[index] for index in landmarks),
    )
    descriptor_smoothing = float(
        metrics["configuration"]["registration"]["descriptor_smoothing"]
    )
    initial_curvature = _curvature_rmse(
        case, source, initial, descriptor_smoothing
    )
    final_curvature = _curvature_rmse(
        case, source, final, descriptor_smoothing
    )
    scale = case.target.median_edge_length
    dense_metrics_ok = bool(
        np.isclose(initial_dense, metrics["initial"]["dense_rmse"], rtol=1e-9, atol=1e-12)
        and np.isclose(final_dense, metrics["final"]["dense_rmse"], rtol=1e-9, atol=1e-12)
        and np.isclose(
            initial_dense / scale,
            metrics["initial"]["dense_rmse_median_edges"],
            rtol=1e-9,
            atol=1e-12,
        )
        and np.isclose(
            final_dense / scale,
            metrics["final"]["dense_rmse_median_edges"],
            rtol=1e-9,
            atol=1e-12,
        )
    )
    residual_metrics_ok = bool(
        np.isclose(initial_landmark, metrics["initial"]["landmark_rmse"], rtol=1e-8, atol=1e-11)
        and np.isclose(final_landmark, metrics["final"]["landmark_rmse"], rtol=1e-8, atol=1e-11)
        and np.isclose(initial_curvature, metrics["initial"]["curvature_rmse"], rtol=1e-8, atol=1e-10)
        and np.isclose(final_curvature, metrics["final"]["curvature_rmse"], rtol=1e-8, atol=1e-10)
    )
    trial_configuration = metrics["configuration"]["trial"]
    registration_configuration = metrics["configuration"]["registration"]
    cfl_limit = max(
        float(trial_configuration["perturbation_cfl"]),
        float(registration_configuration["cfl_limit"]),
    )
    certificate = audit_located_common_map_flow(
        case.common,
        case.source,
        case.target,
        history,
        max_audit_faces=int(trial_configuration["audit_faces"]),
        cfl_limit=cfl_limit,
    )
    saved_certificate = metrics["certificate"]
    certificate_ok = set(saved_certificate) == set(certificate.__dict__) and all(
        _saved_scalar_matches(saved_certificate[key], value)
        for key, value in certificate.__dict__.items()
    )
    objective_values = [float(record["objective"]) for record in metrics["iterations"]]
    objective_monotone = all(
        later <= earlier + 1e-12
        for earlier, later in zip(objective_values, objective_values[1:])
    )
    mode = str(metrics["mode"])
    expected_residual_improved = bool(
        (mode == "curvature" and final_curvature <= initial_curvature)
        or (mode == "landmark" and final_landmark <= initial_landmark)
        or (
            mode == "combined"
            and final_landmark <= initial_landmark
            and final_curvature <= initial_curvature
        )
    )
    valid_run = bool(
        certificate.certified and objective_monotone and expected_residual_improved
    )
    registration_success = bool(valid_run and final_dense < initial_dense)
    recomputed_status = {
        "expected_residual_improved": expected_residual_improved,
        "valid_run": valid_run,
        "accepted": valid_run,
        "registration_success": registration_success,
    }
    status_ok = all(
        key not in metrics or bool(metrics[key]) == expected
        for key, expected in recomputed_status.items()
    )
    result = {
        "run_directory": str(run_directory.resolve()),
        "case": metrics["case"],
        "genus": metrics["genus"],
        "mode": metrics["mode"],
        "seed": metrics["configuration"]["seed"],
        "source_state_matches_case": source_match,
        "truth_state_matches_case": truth_match,
        "flow_replay_matches_final": replay_ok,
        "flow_replay_max_error": replay_error,
        "dense_metrics_recomputed": dense_metrics_ok,
        "residual_metrics_recomputed": residual_metrics_ok,
        "certificate_recomputed": certificate_ok,
        "objective_monotone_recomputed": objective_monotone,
        "saved_status_matches_recomputed": status_ok,
        "initial_dense_rmse_median_edges": initial_dense / scale,
        "final_dense_rmse_median_edges": final_dense / scale,
        "dense_improvement_fraction": (initial_dense - final_dense)
        / max(initial_dense, 1e-15),
        "initial_landmark_rmse_median_edges": initial_landmark / scale,
        "final_landmark_rmse_median_edges": final_landmark / scale,
        "initial_curvature_rmse": initial_curvature,
        "final_curvature_rmse": final_curvature,
        "certified": certificate.certified,
        "registration_success": registration_success,
        "certificate": {
            key: _scalar(value)
            for key, value in certificate.__dict__.items()
        },
        "artifact_sha256": {
            "metrics": _sha256(metrics_path),
            "history": _sha256(history_path),
            "states": _sha256(states_path),
            "correspondence": _sha256(run_directory / "correspondence.png"),
            "objective": _sha256(run_directory / "objective.png"),
        },
    }
    result["passed"] = bool(
        source_match
        and truth_match
        and replay_ok
        and dense_metrics_ok
        and residual_metrics_ok
        and certificate_ok
        and objective_monotone
        and status_ok
    )
    return result


def audit_suite(
    run_directories: list[str | Path],
    *,
    output_directory: str | Path,
    data_root: str | Path = "external_data/s2020-intersurfacemaps-data",
    require_complete_matrix: bool = True,
) -> dict[str, Any]:
    """Audit explicit primary runs and create aggregate tables and plots."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    case_cache: dict[tuple[str, int, int | None], ExperimentCase] = {}
    audits: list[dict[str, Any]] = []
    triplets: set[tuple[str, int, str]] = set()
    for directory in sorted(Path(item) for item in run_directories):
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        name = str(metrics["case"])
        samples = int(metrics["sample_count"])
        resolution = (
            int(metrics["provenance"]["resolution"])
            if name == "synthetic_genus2"
            else None
        )
        key = (name, samples, resolution)
        if key not in case_cache:
            case_cache[key] = (
                build_synthetic_genus2_case(
                    resolution=resolution or 20, maximum_samples=samples
                )
                if name == "synthetic_genus2"
                else load_official_case(name, data_root, maximum_samples=samples)
            )
        audit = audit_saved_trial(directory, data_root=data_root, case=case_cache[key])
        triplet = (audit["case"], int(audit["seed"]), audit["mode"])
        if triplet in triplets:
            raise ValueError(f"duplicate primary trial: {triplet}")
        triplets.add(triplet)
        audits.append(audit)
    return _finalize_audits(
        audits,
        output_directory=output_directory,
        require_complete_matrix=require_complete_matrix,
    )


def combine_audit_files(
    audit_files: list[str | Path],
    *,
    output_directory: str | Path,
    require_complete_matrix: bool = True,
) -> dict[str, Any]:
    """Combine already independently audited partial payloads without reruns."""

    audits: list[dict[str, Any]] = []
    for path in audit_files:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        trials = payload.get("trials")
        if not isinstance(trials, list):
            raise ValueError(f"audit payload contains no trial list: {path}")
        audits.extend(trials)
    return _finalize_audits(
        audits,
        output_directory=Path(output_directory),
        require_complete_matrix=require_complete_matrix,
    )


def _finalize_audits(
    audits: list[dict[str, Any]],
    *,
    output_directory: Path,
    require_complete_matrix: bool,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    triplets: set[tuple[str, int, str]] = set()
    for audit in audits:
        triplet = (audit["case"], int(audit["seed"]), audit["mode"])
        if triplet in triplets:
            raise ValueError(f"duplicate primary trial: {triplet}")
        triplets.add(triplet)
    expected_cases = {
        "synthetic_genus2",
        "official_genus3",
        "official_genus5",
        "official_pretzel_genus3",
    }
    expected_seeds = {3, 11, 29}
    expected_modes = {"landmark", "curvature", "combined"}
    expected_triplets = {
        (case, seed, mode)
        for case in expected_cases
        for seed in expected_seeds
        for mode in expected_modes
    }
    matrix_complete = triplets == expected_triplets
    if require_complete_matrix and not matrix_complete:
        missing = sorted(expected_triplets - triplets)
        extra = sorted(triplets - expected_triplets)
        raise ValueError(f"primary matrix incomplete; missing={missing}, extra={extra}")
    grouped = _group_summary(audits)
    official = [item for item in audits if item["case"].startswith("official_")]
    synthetic = [item for item in audits if item["case"] == "synthetic_genus2"]
    overall = {
        "trial_count": len(audits),
        "matrix_complete": matrix_complete,
        "independent_audit_passes": sum(item["passed"] for item in audits),
        "certified": sum(item["certified"] for item in audits),
        "registration_successes": sum(item["registration_success"] for item in audits),
        "all_independent_audits_pass": all(item["passed"] for item in audits),
        "all_certified": all(item["certified"] for item in audits),
        "official_trial_count": len(official),
        "official_registration_successes": sum(
            item["registration_success"] for item in official
        ),
        "synthetic_trial_count": len(synthetic),
        "synthetic_registration_successes": sum(
            item["registration_success"] for item in synthetic
        ),
        "minimum_orientation_ratio": min(
            float(item["certificate"]["minimum_orientation_ratio"])
            for item in audits
        ),
        "maximum_round_trip_error_minimum_edges": max(
            float(item["certificate"]["round_trip_error"])
            for item in audits
        ),
        "maximum_cfl_ratio": max(
            float(item["certificate"]["maximum_cfl_ratio"])
            for item in audits
        ),
        "maximum_projection_error": max(
            float(item["certificate"]["maximum_projection_error"])
            for item in audits
        ),
        "maximum_flow_replay_error": max(
            float(item["flow_replay_max_error"]) for item in audits
        ),
    }
    payload = {"overall": overall, "groups": grouped, "trials": audits}
    (output_directory / "independent_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output_directory / "independent_trials.csv", audits)
    _plot_aggregate(output_directory / "dense_improvement.png", grouped)
    (output_directory / "research_report.md").write_text(
        _render_report(overall, grouped), encoding="utf-8"
    )
    return payload


def _load_points(archive, prefix: str) -> tuple[SurfacePoint, ...]:
    faces = np.asarray(archive[f"{prefix}_faces"], dtype=np.int64)
    barycentric = np.asarray(archive[f"{prefix}_barycentric"], dtype=np.float64)
    return tuple(
        SurfacePoint(int(face), barycentric[index])
        for index, face in enumerate(faces)
    )


def _point_positions(mesh, points: tuple[SurfacePoint, ...]) -> np.ndarray:
    return np.vstack([point.position(mesh) for point in points])


def _point_rmse(mesh, first, second) -> float:
    delta = _point_positions(mesh, first) - _point_positions(mesh, second)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _point_max_error(mesh, first, second) -> float:
    delta = _point_positions(mesh, first) - _point_positions(mesh, second)
    return float(np.max(np.linalg.norm(delta, axis=1), initial=0.0))


def _curvature_rmse(
    case: ExperimentCase,
    source_points: tuple[SurfacePoint, ...],
    target_points: tuple[SurfacePoint, ...],
    smoothing: float,
) -> float:
    source = curvature_descriptors(case.source)
    target = curvature_descriptors(case.target)
    source_smoother = ScreenedFieldSmoother(case.source, smoothing)
    target_smoother = ScreenedFieldSmoother(case.target, smoothing)
    source_fields = np.column_stack(
        (
            source_smoother.smooth(source.gaussian),
            source_smoother.smooth(source.mean),
        )
    )
    target_fields = np.column_stack(
        (
            target_smoother.smooth(target.gaussian),
            target_smoother.smooth(target.mean),
        )
    )
    first = interpolate_vertex_field(case.source, source_fields, list(source_points))
    second = interpolate_vertex_field(case.target, target_fields, list(target_points))
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _group_summary(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in audits:
        groups[(item["case"], item["mode"])].append(item)
    result: list[dict[str, Any]] = []
    for (case, mode), items in sorted(groups.items()):
        improvements = np.asarray([item["dense_improvement_fraction"] for item in items])
        initial = np.asarray([item["initial_dense_rmse_median_edges"] for item in items])
        final = np.asarray([item["final_dense_rmse_median_edges"] for item in items])
        result.append(
            {
                "case": case,
                "mode": mode,
                "trials": len(items),
                "independent_audit_passes": int(sum(item["passed"] for item in items)),
                "certified": int(sum(item["certified"] for item in items)),
                "registration_successes": int(
                    sum(item["registration_success"] for item in items)
                ),
                "mean_initial_dense_rmse_median_edges": float(initial.mean()),
                "mean_final_dense_rmse_median_edges": float(final.mean()),
                "mean_dense_improvement_fraction": float(improvements.mean()),
                "std_dense_improvement_fraction": float(improvements.std(ddof=0)),
                "minimum_dense_improvement_fraction": float(improvements.min()),
                "mean_landmark_improvement_fraction": float(
                    np.mean(
                        [
                            (
                                item["initial_landmark_rmse_median_edges"]
                                - item["final_landmark_rmse_median_edges"]
                            )
                            / max(item["initial_landmark_rmse_median_edges"], 1e-15)
                            for item in items
                        ]
                    )
                ),
                "mean_curvature_improvement_fraction": float(
                    np.mean(
                        [
                            (item["initial_curvature_rmse"] - item["final_curvature_rmse"])
                            / max(item["initial_curvature_rmse"], 1e-15)
                            for item in items
                        ]
                    )
                ),
            }
        )
    return result


def _write_csv(path: Path, audits: list[dict[str, Any]]) -> None:
    fields = (
        "case",
        "genus",
        "seed",
        "mode",
        "passed",
        "certified",
        "registration_success",
        "initial_dense_rmse_median_edges",
        "final_dense_rmse_median_edges",
        "dense_improvement_fraction",
        "initial_landmark_rmse_median_edges",
        "final_landmark_rmse_median_edges",
        "initial_curvature_rmse",
        "final_curvature_rmse",
        "flow_replay_max_error",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in audits:
            writer.writerow({field: item[field] for field in fields})


def _plot_aggregate(path: Path, groups: list[dict[str, Any]]) -> None:
    cases = sorted({group["case"] for group in groups})
    modes = ("landmark", "curvature", "combined")
    figure, axis = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(cases))
    width = 0.24
    colors = ("#3b82f6", "#f59e0b", "#10b981")
    for offset, (mode, color) in enumerate(zip(modes, colors)):
        selected = {
            group["case"]: group for group in groups if group["mode"] == mode
        }
        # Partial audits intentionally contain only a subset of the primary
        # case/mode matrix.  Matplotlib skips NaN bars, which lets the same
        # plotting code serve both incremental and complete audits.
        means = [
            selected[case]["mean_dense_improvement_fraction"]
            if case in selected
            else np.nan
            for case in cases
        ]
        errors = [
            selected[case]["std_dense_improvement_fraction"]
            if case in selected
            else np.nan
            for case in cases
        ]
        axis.bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=errors,
            label=mode,
            color=color,
            alpha=0.9,
            capsize=3,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [case.replace("official_", "") for case in cases], rotation=12)
    axis.set_ylabel("held-out dense error improvement fraction")
    axis.set_title("Three-seed registration recovery (mean +/- population std)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _render_report(overall: dict[str, Any], groups: list[dict[str, Any]]) -> str:
    lines = [
        "# Native multi-chart high-genus registration: independent audit",
        "",
        "## Outcome",
        "",
        f"- Primary matrix: {overall['trial_count']} trials; complete={overall['matrix_complete']}.",
        f"- Reload-and-recompute audits passed: {overall['independent_audit_passes']}/{overall['trial_count']}.",
        f"- Numerical homeomorphism certificates: {overall['certified']}/{overall['trial_count']}.",
        f"- Held-out dense registration improvements: {overall['registration_successes']}/{overall['trial_count']}.",
        f"- Published real-pair improvements: {overall['official_registration_successes']}/{overall['official_trial_count']}; generated genus-2 improvements: {overall['synthetic_registration_successes']}/{overall['synthetic_trial_count']}.",
        "- Worst audited values: minimum orientation ratio "
        f"{overall['minimum_orientation_ratio']:.6g}; maximum forward/inverse error "
        f"{overall['maximum_round_trip_error_minimum_edges']:.6g} minimum target edges; "
        f"maximum CFL ratio {overall['maximum_cfl_ratio']:.6g}; maximum overlay projection error "
        f"{overall['maximum_projection_error']:.6g}; maximum saved-flow replay error "
        f"{overall['maximum_flow_replay_error']:.6g}.",
        "",
        "## Per-case, per-objective aggregate",
        "",
        "| case | mode | certified | dense successes | mean initial | mean final | mean improvement | landmark improvement | curvature improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            "| {case} | {mode} | {certified}/{trials} | {success}/{trials} | {initial:.4f} | {final:.4f} | {improvement:+.1%} | {landmark:+.1%} | {curvature:+.1%} |".format(
                case=group["case"],
                mode=group["mode"],
                certified=group["certified"],
                success=group["registration_successes"],
                trials=group["trials"],
                initial=group["mean_initial_dense_rmse_median_edges"],
                final=group["mean_final_dense_rmse_median_edges"],
                improvement=group["mean_dense_improvement_fraction"],
                landmark=group["mean_landmark_improvement_fraction"],
                curvature=group["mean_curvature_improvement_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## What is and is not established",
            "",
            "The state is a global target-surface point `(face id, barycentric coordinates)`; face ids change only through deterministic edge walking. The optimized generators are edge-compatible quadratic fields in the native PL atlas, so there is no fixed source-chart to target-chart assignment and no cut seam.",
            "",
            "The numerical certificate combines a topology-valid common-refinement base map, projection checks, a shared Lipschitz-CFL policy, target-chart orientation probes, and forward/inverse replay. It is floating-point evidence, not an exact-predicate proof for arbitrary smooth maps.",
            "",
            "Official mesh landmarks are synthetic samples from the published common map; they are not manual semantic annotations. The published map is held out for error measurement and initialization perturbation. Consequently these experiments validate robust refinement from a known topology class, not automatic initialization from unrelated raw meshes.",
            "",
            "Curvature-only optimization is accepted only when its own residual decreases, but correspondence success is reported separately. On geometrically dissimilar official pairs, curvature residual reduction can move away from the held-out map; this negative result is retained.",
            "",
            "The official archive contains genus-3 and genus-5 cases, not genus-2. Exact genus-2 coverage is supplied by the generated smoothed double torus and is not presented as published real data.",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _saved_scalar_matches(saved: Any, recomputed: Any) -> bool:
    recomputed = _scalar(recomputed)
    if isinstance(recomputed, bool):
        return isinstance(saved, bool) and saved == recomputed
    if isinstance(recomputed, int):
        return isinstance(saved, int) and not isinstance(saved, bool) and saved == recomputed
    if isinstance(recomputed, float):
        if not isinstance(saved, (int, float)) or isinstance(saved, bool):
            return False
        return bool(
            np.isclose(float(saved), recomputed, rtol=1e-8, atol=1e-10)
        )
    return saved == recomputed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("external_data/s2020-intersurfacemaps-data"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    audit_suite(
        arguments.run_directories,
        output_directory=arguments.output,
        data_root=arguments.data_root,
        require_complete_matrix=not arguments.allow_incomplete,
    )


if __name__ == "__main__":
    main()
