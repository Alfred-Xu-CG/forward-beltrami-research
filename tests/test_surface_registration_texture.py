import json

import numpy as np

from qcopt.experiments.surface_registration_texture import (
    _parse_args,
    _official_source_obj_path,
    evaluate_source_texture_rgb,
    load_obj_face_uv,
    pullback_source_positions,
    render_saved_trial,
    render_trial_texture,
    source_face_texture_rgb,
    source_texture_rgb,
    split_perturbation_history,
)
from qcopt.surface_flow import FlowHistory, FlowStep, IntrinsicFaceField
from qcopt.surface_history_io import save_flow_history
from qcopt.surface_mesh import CommonRefinement, CommonRefinementRepair, SurfaceMesh


def _tetrahedron(scale: float) -> SurfaceMesh:
    vertices = scale * np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return SurfaceMesh(vertices, faces)


def _common_tetrahedra() -> CommonRefinement:
    return CommonRefinement(
        _tetrahedron(1.0),
        _tetrahedron(2.0),
        CommonRefinementRepair(0, 0, 0, 0, 0.0, 1e-6),
    )


def test_identity_pullback_transfers_common_map_barycentric_coordinates():
    common = _common_tetrahedra()

    result = pullback_source_positions(
        common.target,
        common,
        FlowHistory(),
    )

    assert result.source_positions.shape == (common.target.n_faces, 3)
    assert np.allclose(result.source_positions, 0.5 * result.query_positions, atol=1e-12)
    assert result.maximum_locator_distance < 1e-12
    assert result.flow_transitions == 0


def test_pullback_samples_face_interiors_where_intrinsic_flow_is_observable():
    common = _common_tetrahedra()
    ambient = np.tile(np.array([0.03, -0.02, 0.01]), (common.target.n_vertices, 1))
    field = IntrinsicFaceField.from_vertex_field(common.target, ambient)
    history = FlowHistory((FlowStep(field, 0.4, 8),))

    truth = pullback_source_positions(common.target, common, FlowHistory())
    moved = pullback_source_positions(common.target, common, history)
    displacement = np.linalg.norm(moved.source_positions - truth.source_positions, axis=1)

    assert displacement.max() > 1e-3


def test_source_texture_is_finite_bounded_and_spatially_varying():
    source = _tetrahedron(1.0)

    colors = source_texture_rgb(source.vertices, source.vertices)

    assert colors.shape == (source.n_vertices, 3)
    assert np.all(np.isfinite(colors))
    assert np.all((0.0 <= colors) & (colors <= 1.0))
    assert np.ptp(colors, axis=0).max() > 0.25
    assert colors.min() > 0.03


def test_source_face_texture_has_exactly_one_color_per_triangle():
    source = _tetrahedron(1.0)

    colors = source_face_texture_rgb(source)

    assert colors.shape == (source.n_faces, 3)
    assert np.all((0.0 <= colors) & (colors <= 1.0))


def test_procedural_fallback_uses_coherent_discrete_color_cells():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(80, 3))
    positions = rng.normal(size=(120, 3))

    colors = source_texture_rgb(positions, reference)
    distinct = np.unique(np.round(colors, 6), axis=0)

    assert len(distinct) <= 16


def test_obj_corner_uv_drives_the_transferred_color_texture(tmp_path):
    source = _tetrahedron(1.0)
    obj = tmp_path / "textured.obj"
    obj.write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "vt 0.05 0.05",
                "vt 0.35 0.05",
                "vt 0.05 0.35",
                "vt 0.65 0.65",
                "f 1/1 3/3 2/2",
                "f 1/1 2/2 4/4",
                "f 2/2 3/3 4/4",
                "f 3/3 1/1 4/4",
            )
        ),
        encoding="utf-8",
    )
    face_uv = load_obj_face_uv(obj, source)
    centroids = source.vertices[source.faces].mean(axis=1)

    colors = evaluate_source_texture_rgb(source, centroids, face_uv=face_uv)

    assert face_uv.shape == (source.n_faces, 3, 2)
    assert colors.shape == (source.n_faces, 3)
    assert np.ptp(colors, axis=0).max() > 0.1


def test_perturbation_history_is_the_prefix_before_correction_steps():
    field = np.zeros((4, 3))
    perturbation = FlowStep(field, 1.0, 3)
    first_correction = FlowStep(field, 0.5, 2)
    second_correction = FlowStep(field, 0.25, 1)
    full = FlowHistory((perturbation, first_correction, second_correction))
    correction = FlowHistory((first_correction, second_correction))

    prefix = split_perturbation_history(full, correction)

    assert len(prefix.steps) == 1
    assert prefix.steps[0] is perturbation


def test_history_split_rejects_a_longer_correction_history():
    field = np.zeros((4, 3))
    step = FlowStep(field, 1.0, 1)

    try:
        split_perturbation_history(FlowHistory(), FlowHistory((step,)))
    except ValueError as error:
        assert "correction" in str(error)
    else:
        raise AssertionError("a correction history cannot exceed the full history")


def test_history_split_rejects_a_nonmatching_correction_suffix():
    field = np.zeros((4, 3))
    full = FlowHistory(
        (FlowStep(field, 1.0, 1), FlowStep(field, 0.5, 2))
    )
    inconsistent = FlowHistory((FlowStep(field, 0.25, 2),))

    try:
        split_perturbation_history(full, inconsistent)
    except ValueError as error:
        assert "suffix" in str(error)
    else:
        raise AssertionError("correction steps must match the full-history suffix")


def test_trial_renderer_writes_texture_error_and_machine_readable_metrics(tmp_path):
    common = _common_tetrahedra()

    metrics = render_trial_texture(
        source=common.source,
        target=common.target,
        common=common,
        full_history=FlowHistory(),
        correction_history=FlowHistory(),
        output_directory=tmp_path,
        label="identity tetrahedron",
    )

    texture_path = tmp_path / "texture_registration.png"
    interactive_path = tmp_path / "texture_registration.html"
    error_path = tmp_path / "inverse_map_error.png"
    metrics_path = tmp_path / "texture_metrics.json"
    assert texture_path.stat().st_size > 10_000
    assert interactive_path.stat().st_size > 10_000
    interactive = interactive_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in interactive
    assert all(f'"scene{suffix}"' in interactive for suffix in ("", "2", "3", "4"))
    assert "source texture" in interactive
    assert "perturbed initialization" in interactive
    assert "refined registration" in interactive
    assert error_path.stat().st_size > 10_000
    assert metrics_path.stat().st_size > 100
    reloaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert reloaded == metrics
    assert metrics["initial"]["source_preimage_rmse_median_edges"] < 1e-12
    assert metrics["final"]["source_preimage_rmse_median_edges"] < 1e-12


def test_saved_trial_wrapper_loads_histories_and_uses_saved_trial_label(tmp_path):
    common = _common_tetrahedra()
    save_flow_history(tmp_path / "flow_history.npz", FlowHistory())
    save_flow_history(tmp_path / "correction_history.npz", FlowHistory())
    (tmp_path / "metrics.json").write_text(
        json.dumps({"case": "tetra", "mode": "identity", "configuration": {"seed": 7}}),
        encoding="utf-8",
    )

    metrics = render_saved_trial(
        source=common.source,
        target=common.target,
        common=common,
        run_directory=tmp_path,
    )

    assert metrics["label"] == "tetra | identity | seed 7"


def test_texture_cli_parses_case_run_and_optional_output_paths(tmp_path):
    arguments = _parse_args(
        [
            "--case",
            "synthetic_genus2",
            "--run-directory",
            str(tmp_path / "run"),
            "--output-directory",
            str(tmp_path / "figures"),
        ]
    )

    assert arguments.case == "synthetic_genus2"
    assert arguments.run_directory == tmp_path / "run"
    assert arguments.output_directory == tmp_path / "figures"


def test_official_source_obj_path_uses_the_published_case_directory(tmp_path):
    assert _official_source_obj_path("official_genus3", tmp_path) == (
        tmp_path / "Fig11_genus" / "3" / "A.obj"
    )
    assert _official_source_obj_path("official_pretzel_genus3", tmp_path) == (
        tmp_path / "Fig12_texture" / "pretzel" / "A.obj"
    )
