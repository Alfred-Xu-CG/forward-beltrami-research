"""Dense texture-pullback visualization for saved surface registrations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.subplots import make_subplots

from ..surface_flow import (
    FlowHistory,
    FlowStep,
    IntrinsicFaceField,
    points_at_face_centroids,
)
from ..surface_history_io import load_flow_history
from ..surface_locator import SurfaceLocator
from ..surface_mesh import CommonRefinement, SurfaceMesh


@dataclass(frozen=True)
class TexturePullback:
    """Source positions seen through a target-surface registration."""

    source_positions: np.ndarray
    query_positions: np.ndarray
    maximum_locator_distance: float
    flow_transitions: int
    flow_substeps: int


def split_perturbation_history(
    full: FlowHistory, correction: FlowHistory
) -> FlowHistory:
    """Return the perturbation prefix of a full perturbation+correction flow."""

    correction_count = len(correction.steps)
    if correction_count > len(full.steps):
        raise ValueError("correction history is longer than the full history")
    prefix_count = len(full.steps) - correction_count
    suffix = full.steps[prefix_count:]
    if any(
        not _flow_steps_equal(full_step, correction_step)
        for full_step, correction_step in zip(suffix, correction.steps)
    ):
        raise ValueError("correction history does not match the full-history suffix")
    return FlowHistory(full.steps[:prefix_count])


def _flow_steps_equal(first: FlowStep, second: FlowStep) -> bool:
    if first.duration != second.duration or first.substeps != second.substeps:
        return False
    if isinstance(first.vertex_field, IntrinsicFaceField) != isinstance(
        second.vertex_field, IntrinsicFaceField
    ):
        return False
    if isinstance(first.vertex_field, IntrinsicFaceField):
        assert isinstance(second.vertex_field, IntrinsicFaceField)
        return bool(
            np.array_equal(first.vertex_field.edge_values, second.vertex_field.edge_values)
            and np.array_equal(
                first.vertex_field.bubble_values, second.vertex_field.bubble_values
            )
        )
    return bool(np.array_equal(first.vertex_field, second.vertex_field))


def pullback_source_positions(
    target: SurfaceMesh,
    common: CommonRefinement,
    history: FlowHistory,
) -> TexturePullback:
    """Evaluate source preimages at target face centroids for ``Phi o F0``.

    ``F0`` is represented by the paired embeddings of ``common`` and ``Phi``
    is the saved target-surface flow.  The inverse flow is evaluated first;
    the resulting target point is then transferred through the common mesh by
    retaining its overlay face and barycentric coordinates.
    """

    # Intrinsic P2 velocity fields vanish at PL cone vertices by construction.
    # Sampling vertices would therefore make every saved flow appear to be the
    # identity. Face interiors reveal the actual registration motion.
    target_points = points_at_face_centroids(target)
    inverse = history.inverse(target, target_points)
    queries = np.vstack([point.position(target) for point in inverse.points])
    located = SurfaceLocator(common.target).locate_many(queries)
    source_positions = np.vstack(
        [
            common.source.point_from_barycentric(point.face, point.barycentric)
            for point in located.points
        ]
    )
    return TexturePullback(
        source_positions=source_positions,
        query_positions=queries,
        maximum_locator_distance=located.maximum_distance,
        flow_transitions=inverse.transition_count,
        flow_substeps=inverse.substeps,
    )


def source_texture_rgb(
    positions: np.ndarray,
    reference_vertices: np.ndarray,
    *,
    frequency: float = 7.0,
) -> np.ndarray:
    """Evaluate a seam-free embedded-coordinate texture on source positions."""

    positions = np.asarray(positions, dtype=np.float64)
    reference = np.asarray(reference_vertices, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 3:
        raise ValueError("reference_vertices must have shape (m, 3), m >= 3")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(reference)):
        raise ValueError("texture coordinates must be finite")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be positive and finite")

    center = reference.mean(axis=0)
    _, _, axes = np.linalg.svd(reference - center, full_matrices=False)
    axes = np.array(axes, copy=True)
    for index in range(3):
        pivot = int(np.argmax(np.abs(axes[index])))
        if axes[index, pivot] < 0.0:
            axes[index] *= -1.0
    reference_local = (reference - center) @ axes.T
    local = (positions - center) @ axes.T
    minimum = reference_local.min(axis=0)
    span = np.maximum(np.ptp(reference_local, axis=0), 1e-12)
    normalized = (local - minimum) / span

    cells = np.floor(frequency * normalized).astype(np.int64)
    palette = _texture_palette()
    palette_index = np.mod(
        3 * cells[:, 0]
        + 5 * cells[:, 1]
        + 7 * cells[:, 2]
        + cells[:, 0] * cells[:, 1],
        len(palette),
    )
    colors = palette[palette_index]
    shade = np.where(np.mod(cells.sum(axis=1), 2) == 0, 1.0, 0.82)
    return np.clip(colors * shade[:, None], 0.0, 1.0)


def load_obj_face_uv(path: str | Path, source: SurfaceMesh) -> np.ndarray:
    """Load validated per-corner UV coordinates using the source face order."""

    path = Path(path)
    texture_vertices: list[tuple[float, float]] = []
    geometry_vertex_count = 0
    geometry_faces: list[tuple[int, int, int]] = []
    texture_faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line_number, raw in enumerate(stream, start=1):
            fields = raw.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                geometry_vertex_count += 1
            elif fields[0] == "vt":
                if len(fields) < 3:
                    raise ValueError(f"invalid texture coordinate at OBJ line {line_number}")
                coordinate = (float(fields[1]), float(fields[2]))
                if not np.all(np.isfinite(coordinate)):
                    raise ValueError(f"non-finite texture coordinate at OBJ line {line_number}")
                texture_vertices.append(coordinate)
            elif fields[0] == "f":
                if len(fields) < 4:
                    raise ValueError(f"invalid face at OBJ line {line_number}")
                geometry_polygon: list[int] = []
                texture_polygon: list[int] = []
                for token in fields[1:]:
                    indices = token.split("/")
                    if len(indices) < 2 or not indices[0] or not indices[1]:
                        raise ValueError(f"face has no UV index at OBJ line {line_number}")
                    geometry_polygon.append(
                        _obj_index(indices[0], geometry_vertex_count, line_number)
                    )
                    texture_polygon.append(
                        _obj_index(indices[1], len(texture_vertices), line_number)
                    )
                for offset in range(1, len(geometry_polygon) - 1):
                    geometry_faces.append(
                        (
                            geometry_polygon[0],
                            geometry_polygon[offset],
                            geometry_polygon[offset + 1],
                        )
                    )
                    texture_faces.append(
                        (
                            texture_polygon[0],
                            texture_polygon[offset],
                            texture_polygon[offset + 1],
                        )
                    )
    geometry = np.asarray(geometry_faces, dtype=np.int64)
    if geometry.shape != source.faces.shape or not np.array_equal(geometry, source.faces):
        raise ValueError("OBJ geometry faces do not match the loaded source mesh")
    if not texture_vertices or len(texture_faces) != source.n_faces:
        raise ValueError("OBJ contains no complete per-corner UV map")
    coordinates = np.asarray(texture_vertices, dtype=np.float64)
    result = coordinates[np.asarray(texture_faces, dtype=np.int64)]
    result.setflags(write=False)
    return result


def evaluate_source_texture_rgb(
    source: SurfaceMesh,
    positions: np.ndarray,
    *,
    face_uv: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the source texture at arbitrary points on the source mesh."""

    positions = np.asarray(positions, dtype=np.float64)
    if face_uv is None:
        return source_texture_rgb(positions, source.vertices)
    face_uv = _validate_face_uv(face_uv, source)
    located = SurfaceLocator(source).locate_many(positions)
    uv = np.vstack(
        [
            point.barycentric @ face_uv[point.face]
            for point in located.points
        ]
    )
    return _uv_cell_texture_rgb(uv)


def source_face_texture_rgb(
    source: SurfaceMesh, *, face_uv: np.ndarray | None = None
) -> np.ndarray:
    """Return one procedural source-texture color for every source face."""

    if face_uv is not None:
        face_uv = _validate_face_uv(face_uv, source)
        return _uv_cell_texture_rgb(face_uv.mean(axis=1))
    centroids = source.vertices[source.faces].mean(axis=1)
    return source_texture_rgb(centroids, source.vertices)


def render_trial_texture(
    *,
    source: SurfaceMesh,
    target: SurfaceMesh,
    common: CommonRefinement,
    full_history: FlowHistory,
    correction_history: FlowHistory,
    output_directory: str | Path,
    label: str,
    source_face_uv: np.ndarray | None = None,
) -> dict[str, object]:
    """Render dense texture and inverse-map error evidence for one trial."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    initial_history = split_perturbation_history(full_history, correction_history)
    truth = pullback_source_positions(target, common, FlowHistory())
    initial = pullback_source_positions(target, common, initial_history)
    final = pullback_source_positions(target, common, full_history)
    locator_tolerance = max(
        20.0 * float(common.repair.quantization_step),
        5e-4 * target.median_edge_length,
    )
    maximum_locator_distance = max(
        truth.maximum_locator_distance,
        initial.maximum_locator_distance,
        final.maximum_locator_distance,
    )
    if maximum_locator_distance > locator_tolerance:
        raise ValueError(
            "common-refinement localization residual exceeds tolerance: "
            f"{maximum_locator_distance:.6e} > {locator_tolerance:.6e}"
        )

    normalization = source.median_edge_length
    initial_error = np.linalg.norm(
        initial.source_positions - truth.source_positions, axis=1
    ) / normalization
    final_error = np.linalg.norm(
        final.source_positions - truth.source_positions, axis=1
    ) / normalization
    initial_rmse = _vector_rmse(
        initial.source_positions - truth.source_positions
    ) / normalization
    final_rmse = _vector_rmse(
        final.source_positions - truth.source_positions
    ) / normalization
    metrics: dict[str, object] = {
        "label": str(label),
        "sampling": "one target face centroid per triangle chart",
        "texture": (
            "source OBJ UV color cells evaluated through inverse registration"
            if source_face_uv is not None
            else "source PCA-coordinate cells evaluated through inverse registration"
        ),
        "normalization": "source median edge length",
        "source": {
            "vertices": source.n_vertices,
            "faces": source.n_faces,
            "genus": source.topology.genus,
            "median_edge_length": normalization,
        },
        "target": {
            "vertices": target.n_vertices,
            "faces": target.n_faces,
            "genus": target.topology.genus,
            "median_edge_length": target.median_edge_length,
        },
        "histories": {
            "perturbation_steps": len(initial_history.steps),
            "correction_steps": len(correction_history.steps),
            "full_steps": len(full_history.steps),
        },
        "locator": {
            "truth_maximum_distance": truth.maximum_locator_distance,
            "initial_maximum_distance": initial.maximum_locator_distance,
            "final_maximum_distance": final.maximum_locator_distance,
            "tolerance": locator_tolerance,
        },
        "initial": {
            "source_preimage_rmse_median_edges": initial_rmse,
            "source_preimage_max_median_edges": float(
                np.max(initial_error, initial=0.0)
            ),
            "inverse_flow_chart_transitions": initial.flow_transitions,
            "inverse_flow_substeps": initial.flow_substeps,
        },
        "final": {
            "source_preimage_rmse_median_edges": final_rmse,
            "source_preimage_max_median_edges": float(
                np.max(final_error, initial=0.0)
            ),
            "inverse_flow_chart_transitions": final.flow_transitions,
            "inverse_flow_substeps": final.flow_substeps,
        },
    }

    _render_texture_panels(
        source,
        target,
        truth.source_positions,
        initial.source_positions,
        final.source_positions,
        initial_rmse,
        final_rmse,
        label,
        output_directory / "texture_registration.png",
        source_face_uv=source_face_uv,
    )
    _render_interactive_texture_panels(
        source,
        target,
        truth.source_positions,
        initial.source_positions,
        final.source_positions,
        initial_rmse,
        final_rmse,
        label,
        output_directory / "texture_registration.html",
        source_face_uv=source_face_uv,
    )
    _render_error_panels(
        target,
        initial_error,
        final_error,
        initial_rmse,
        final_rmse,
        label,
        output_directory / "inverse_map_error.png",
    )
    (output_directory / "texture_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def render_saved_trial(
    *,
    source: SurfaceMesh,
    target: SurfaceMesh,
    common: CommonRefinement,
    run_directory: str | Path,
    output_directory: str | Path | None = None,
    source_face_uv: np.ndarray | None = None,
) -> dict[str, object]:
    """Reload one persisted experiment trial and render its dense map."""

    run_directory = Path(run_directory)
    required = (
        run_directory / "flow_history.npz",
        run_directory / "correction_history.npz",
        run_directory / "metrics.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing saved trial files: " + ", ".join(missing))
    saved_metrics = json.loads(required[2].read_text(encoding="utf-8"))
    try:
        case_name = str(saved_metrics["case"])
        mode = str(saved_metrics["mode"])
        seed = int(saved_metrics["configuration"]["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("saved metrics do not identify case, mode, and seed") from error
    return render_trial_texture(
        source=source,
        target=target,
        common=common,
        full_history=load_flow_history(required[0]),
        correction_history=load_flow_history(required[1]),
        output_directory=output_directory or run_directory,
        label=f"{case_name} | {mode} | seed {seed}",
        source_face_uv=source_face_uv,
    )


def _render_texture_panels(
    source: SurfaceMesh,
    target: SurfaceMesh,
    truth_source_positions: np.ndarray,
    initial_source_positions: np.ndarray,
    final_source_positions: np.ndarray,
    initial_rmse: float,
    final_rmse: float,
    label: str,
    path: Path,
    *,
    source_face_uv: np.ndarray | None = None,
) -> None:
    source_vertices = _display_vertices(source.vertices)
    target_vertices = _display_vertices(target.vertices)
    source_colors = source_face_texture_rgb(source, face_uv=source_face_uv)
    truth_colors = evaluate_source_texture_rgb(
        source, truth_source_positions, face_uv=source_face_uv
    )
    initial_colors = evaluate_source_texture_rgb(
        source, initial_source_positions, face_uv=source_face_uv
    )
    final_colors = evaluate_source_texture_rgb(
        source, final_source_positions, face_uv=source_face_uv
    )
    panels = (
        (source_vertices, source.faces, source_colors, "source texture"),
        (target_vertices, target.faces, truth_colors, "published/base map $F_0$"),
        (
            target_vertices,
            target.faces,
            initial_colors,
            f"perturbed initialization\npreimage RMSE {initial_rmse:.3f} edges",
        ),
        (
            target_vertices,
            target.faces,
            final_colors,
            f"refined registration\npreimage RMSE {final_rmse:.3f} edges",
        ),
    )
    figure = plt.figure(figsize=(15.5, 4.4), constrained_layout=True)
    for index, (vertices, faces, vertex_colors, title) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 4, index, projection="3d")
        _draw_colored_surface(
            axis,
            vertices,
            faces,
            vertex_colors,
            colors_are_per_face=True,
        )
        _style_surface_axis(axis, vertices)
        axis.set_title(title, fontsize=10, pad=2)
    figure.suptitle(
        f"{label}: one source texture transported by each registration map",
        fontsize=12,
    )
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def _render_error_panels(
    target: SurfaceMesh,
    initial_vertex_error: np.ndarray,
    final_vertex_error: np.ndarray,
    initial_rmse: float,
    final_rmse: float,
    label: str,
    path: Path,
) -> None:
    vertices = _display_vertices(target.vertices)
    maximum = max(
        float(np.max(initial_vertex_error, initial=0.0)),
        float(np.max(final_vertex_error, initial=0.0)),
        1e-12,
    )
    normalization = Normalize(vmin=0.0, vmax=maximum)
    colormap = plt.get_cmap("magma")
    figure = plt.figure(figsize=(9.2, 4.4), constrained_layout=True)
    axes = []
    for index, (error, rmse, title) in enumerate(
        (
            (initial_vertex_error, initial_rmse, "perturbed initialization"),
            (final_vertex_error, final_rmse, "refined registration"),
        ),
        start=1,
    ):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        face_colors = colormap(normalization(error))
        _draw_colored_surface(
            axis,
            vertices,
            target.faces,
            face_colors,
            colors_are_per_face=True,
        )
        _style_surface_axis(axis, vertices)
        axis.set_title(f"{title}\nRMSE {rmse:.3f} source edges", fontsize=10, pad=2)
        axes.append(axis)
    figure.colorbar(
        ScalarMappable(norm=normalization, cmap=colormap),
        ax=axes,
        shrink=0.72,
        pad=0.015,
        label="source-preimage error / source median edge",
    )
    figure.suptitle(f"{label}: correspondence error relative to $F_0$", fontsize=12)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def _render_interactive_texture_panels(
    source: SurfaceMesh,
    target: SurfaceMesh,
    truth_source_positions: np.ndarray,
    initial_source_positions: np.ndarray,
    final_source_positions: np.ndarray,
    initial_rmse: float,
    final_rmse: float,
    label: str,
    path: Path,
    *,
    source_face_uv: np.ndarray | None = None,
) -> None:
    """Write a rotatable four-panel registration texture comparison."""

    source_vertices = _display_vertices(source.vertices)
    target_vertices = _display_vertices(target.vertices)
    panels = (
        (
            source_vertices,
            source.faces,
            source_face_texture_rgb(source, face_uv=source_face_uv),
            "source texture",
        ),
        (
            target_vertices,
            target.faces,
            evaluate_source_texture_rgb(
                source, truth_source_positions, face_uv=source_face_uv
            ),
            "published / base map F0",
        ),
        (
            target_vertices,
            target.faces,
            evaluate_source_texture_rgb(
                source, initial_source_positions, face_uv=source_face_uv
            ),
            f"perturbed initialization | RMSE {initial_rmse:.3f} edges",
        ),
        (
            target_vertices,
            target.faces,
            evaluate_source_texture_rgb(
                source, final_source_positions, face_uv=source_face_uv
            ),
            f"refined registration | RMSE {final_rmse:.3f} edges",
        ),
    )
    figure = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scene"}] * 4],
        subplot_titles=[panel[3] for panel in panels],
        horizontal_spacing=0.008,
    )
    for column, (vertices, faces, colors, title) in enumerate(panels, start=1):
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                facecolor=_rgb_strings(colors),
                flatshading=True,
                lighting={
                    "ambient": 0.68,
                    "diffuse": 0.72,
                    "specular": 0.12,
                    "roughness": 0.82,
                    "fresnel": 0.04,
                },
                lightposition={"x": 120, "y": 80, "z": 180},
                hovertemplate=title + "<extra></extra>",
                showscale=False,
                name=title,
            ),
            row=1,
            col=column,
        )
    scene_style = {
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "zaxis": {"visible": False},
        "aspectmode": "data",
        "camera": {"projection": {"type": "orthographic"}},
        "bgcolor": "rgba(0,0,0,0)",
    }
    figure.update_layout(
        title={
            "text": (
                f"{label}<br><sup>Same source texture; drag any target panel "
                "to rotate the three target views together</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        scene=scene_style,
        scene2=scene_style,
        scene3=scene_style,
        scene4=scene_style,
        margin={"l": 0, "r": 0, "t": 88, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        height=570,
        showlegend=False,
    )
    synchronize_target_cameras = r"""
(function () {
  const graph = document.getElementById('{plot_id}');
  const targetScenes = ['scene2', 'scene3', 'scene4'];
  let synchronizing = false;
  graph.on('plotly_relayout', function (event) {
    if (synchronizing) return;
    const changed = Object.entries(event).filter(([key]) =>
      targetScenes.some((scene) => key === scene + '.camera' || key.startsWith(scene + '.camera.'))
    );
    if (!changed.length) return;
    const sourceScene = targetScenes.find((scene) =>
      changed.some(([key]) => key === scene + '.camera' || key.startsWith(scene + '.camera.'))
    );
    const update = {};
    for (const [key, value] of changed) {
      const suffix = key.slice(sourceScene.length);
      for (const scene of targetScenes) {
        if (scene !== sourceScene) update[scene + suffix] = value;
      }
    }
    synchronizing = true;
    Plotly.relayout(graph, update).finally(() => { synchronizing = false; });
  });
})();
"""
    figure.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
        post_script=synchronize_target_cameras,
        auto_open=False,
    )


def _draw_colored_surface(
    axis,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    colors_are_per_face: bool = False,
) -> None:
    face_colors = colors if colors_are_per_face else colors[faces].mean(axis=1)
    collection = Poly3DCollection(
        vertices[faces],
        facecolors=face_colors,
        edgecolors=(0.02, 0.025, 0.035, 0.10),
        linewidths=0.08,
    )
    axis.add_collection3d(collection)


def _display_vertices(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    axes = np.array(axes, copy=True)
    for index in range(3):
        pivot = int(np.argmax(np.abs(axes[index])))
        if axes[index, pivot] < 0.0:
            axes[index] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[2] *= -1.0
    return centered @ axes.T


def _style_surface_axis(axis, vertices: np.ndarray) -> None:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.53 * max(float(np.max(maximum - minimum)), 1e-12)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_proj_type("ortho")
    axis.view_init(elev=23.0, azim=-62.0)
    axis.set_axis_off()


def _vector_rmse(vectors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(vectors) ** 2, axis=1))))


def _rgb_strings(colors: np.ndarray) -> list[str]:
    integers = np.rint(255.0 * np.clip(colors, 0.0, 1.0)).astype(np.int64)
    return [f"rgb({red},{green},{blue})" for red, green, blue in integers]


def _uv_cell_texture_rgb(uv: np.ndarray, *, frequency: float = 14.0) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.all(np.isfinite(uv)):
        raise ValueError("UV samples must be a finite (n, 2) array")
    cells = np.floor(float(frequency) * uv).astype(np.int64)
    palette = _texture_palette()
    palette_index = np.mod(
        3 * cells[:, 0] + 5 * cells[:, 1] + cells[:, 0] * cells[:, 1],
        len(palette),
    )
    colors = palette[palette_index]
    shade = np.where(np.mod(cells[:, 0] + cells[:, 1], 2) == 0, 1.0, 0.82)
    return np.clip(colors * shade[:, None], 0.0, 1.0)


def _texture_palette() -> np.ndarray:
    return np.asarray(
        [
            [0.06, 0.48, 0.56],
            [0.06, 0.68, 0.62],
            [0.18, 0.72, 0.30],
            [0.72, 0.78, 0.12],
            [0.98, 0.67, 0.10],
            [0.89, 0.28, 0.18],
            [0.56, 0.18, 0.48],
            [0.18, 0.28, 0.62],
        ],
        dtype=np.float64,
    )


def _validate_face_uv(face_uv: np.ndarray, source: SurfaceMesh) -> np.ndarray:
    result = np.asarray(face_uv, dtype=np.float64)
    if result.shape != (source.n_faces, 3, 2) or not np.all(np.isfinite(result)):
        raise ValueError("face_uv must be a finite (source.n_faces, 3, 2) array")
    return result


def _obj_index(raw: str, current_count: int, line_number: int) -> int:
    value = int(raw)
    index = value - 1 if value > 0 else current_count + value
    if index < 0 or index >= current_count:
        raise ValueError(f"OBJ index out of range at line {line_number}")
    return index


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "synthetic_genus2",
            "official_genus3",
            "official_genus5",
            "official_pretzel_genus3",
        ),
    )
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("external_data/s2020-intersurfacemaps-data"),
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parse_args(arguments)
    from .high_genus_registration import (
        build_synthetic_genus2_case,
        load_official_case,
    )

    case = (
        build_synthetic_genus2_case(maximum_samples=16)
        if parsed.case == "synthetic_genus2"
        else load_official_case(parsed.case, parsed.data_root, maximum_samples=16)
    )
    source_face_uv = (
        None
        if parsed.case == "synthetic_genus2"
        else load_obj_face_uv(
            _official_source_obj_path(parsed.case, parsed.data_root), case.source
        )
    )
    metrics = render_saved_trial(
        source=case.source,
        target=case.target,
        common=case.common,
        run_directory=parsed.run_directory,
        output_directory=parsed.output_directory,
        source_face_uv=source_face_uv,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _official_source_obj_path(case_name: str, data_root: str | Path) -> Path:
    relative = {
        "official_genus3": Path("Fig11_genus/3/A.obj"),
        "official_genus5": Path("Fig11_genus/5/A.obj"),
        "official_pretzel_genus3": Path("Fig12_texture/pretzel/A.obj"),
    }
    if case_name not in relative:
        raise ValueError(f"no published source OBJ for case: {case_name}")
    return Path(data_root) / relative[case_name]


if __name__ == "__main__":
    main()
