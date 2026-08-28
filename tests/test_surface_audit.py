import numpy as np
from pathlib import Path
import pytest

from qcopt.surface_audit import (
    audit_common_refinement_flow,
    audit_located_common_map_flow,
)
from qcopt.surface_benchmarks import random_smooth_tangent_field
from qcopt.surface_flow import FlowHistory, FlowStep, IntrinsicFaceField
from qcopt.surface_mesh import (
    CommonRefinement,
    CommonRefinementRepair,
    SurfaceMesh,
    load_common_refinement,
    load_obj,
)


def _torus(nu: int = 10, nv: int = 7) -> SurfaceMesh:
    vertices = []
    for i in range(nu):
        u = 2.0 * np.pi * i / nu
        for j in range(nv):
            v = 2.0 * np.pi * j / nv
            vertices.append(
                [
                    (2.0 + 0.55 * np.cos(v)) * np.cos(u),
                    (2.0 + 0.55 * np.cos(v)) * np.sin(u),
                    0.55 * np.sin(v),
                ]
            )
    index = lambda i, j: (i % nu) * nv + (j % nv)
    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i + 1, j + 1), index(i, j + 1)
            faces.extend(((a, b, c), (a, c, d)))
    return SurfaceMesh(np.asarray(vertices), np.asarray(faces))


def _swirl(mesh: SurfaceMesh, scale: float) -> np.ndarray:
    field = scale * np.column_stack(
        (-mesh.vertices[:, 1], mesh.vertices[:, 0], np.zeros(mesh.n_vertices))
    )
    field -= np.sum(field * mesh.vertex_normals, axis=1)[:, None] * mesh.vertex_normals
    return field


def test_identity_common_refinement_has_full_numerical_certificate():
    mesh = _torus()
    report = audit_common_refinement_flow(mesh, mesh, FlowHistory(), max_audit_faces=64)
    assert report.certified
    assert report.topology_ok
    assert report.orientation_ok
    assert report.inverse_ok
    assert report.cfl_ok
    assert np.isclose(report.minimum_orientation_ratio, 1.0, atol=2e-14)
    assert report.round_trip_error == 0.0


def test_small_chart_crossing_flow_retains_certificate():
    mesh = _torus()
    history = FlowHistory((FlowStep(_swirl(mesh, 0.4), 0.8, 32),))
    report = audit_common_refinement_flow(mesh, mesh, history, max_audit_faces=72)
    assert report.certified
    assert report.chart_transitions > 0
    assert report.minimum_orientation_ratio > 0.0
    assert report.round_trip_error < report.round_trip_tolerance


def test_unsafe_single_step_fails_cfl_even_if_topology_matches():
    mesh = _torus()
    history = FlowHistory((FlowStep(_swirl(mesh, 3.0), 1.0, 1),))
    report = audit_common_refinement_flow(mesh, mesh, history, max_audit_faces=24)
    assert not report.certified
    assert not report.cfl_ok


def test_mismatched_common_refinement_is_rejected_not_certified():
    mesh = _torus()
    faces = mesh.faces.copy()
    faces[0] = faces[0, [0, 2, 1]]
    mismatched = SurfaceMesh(mesh.vertices, faces)
    report = audit_common_refinement_flow(mesh, mismatched, FlowHistory(), max_audit_faces=24)
    assert not report.certified
    assert not report.topology_ok


def test_overlay_map_can_be_audited_while_flow_lives_on_original_mesh():
    mesh = _torus()
    common = CommonRefinement(
        mesh,
        mesh,
        CommonRefinementRepair(0, 0, 0, 0, 0.0, 1e-6),
    )
    history = FlowHistory((FlowStep(_swirl(mesh, 0.4), 0.8, 32),))
    report = audit_located_common_map_flow(
        common,
        mesh,
        mesh,
        history,
        max_audit_faces=48,
    )
    assert report.certified
    assert report.projection_ok
    assert report.maximum_projection_error < 1e-14
    assert report.chart_transitions > 0


def test_official_overlay_does_not_create_false_orientation_flip_across_original_charts():
    root = Path(
        "external_data/s2020-intersurfacemaps-data/Fig11_genus/3"
    )
    if not root.exists():
        pytest.skip("optional official data package is not installed")
    source = load_obj(root / "A.obj")
    target = load_obj(root / "B.obj")
    common = load_common_refinement(
        root / "1_closed_000400_iters_M_on_A.obj",
        root / "1_closed_000400_iters_M_on_B.obj",
    )
    raw = random_smooth_tangent_field(
        target, seed=11, relative_amplitude=0.8, smoothing=8.0
    )
    intrinsic = IntrinsicFaceField.from_vertex_field(target, raw)
    history = FlowHistory((FlowStep(intrinsic, 1.0, 96),))

    direct = audit_common_refinement_flow(
        target, target, history, max_audit_faces=96
    )
    located = audit_located_common_map_flow(
        common, source, target, history, max_audit_faces=96
    )

    assert direct.certified
    assert located.certified
    assert located.orientation_ok
