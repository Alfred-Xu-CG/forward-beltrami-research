import numpy as np

from qcopt.constraints import fixed_vertex_constraints
from qcopt.mesh import structured_rectangle, TriMesh
from qcopt.multichart import (
    AffineTransition,
    Atlas,
    Chart,
    Overlap,
    build_affine_compatibility,
    conformal_beltrami_transition,
    nonlinear_compatibility_residual_jacobian,
    raw_coordinate_equality_constraints,
)
from qcopt.multichart_solver import solve_coupled_lsqc


def _two_shifted_charts():
    base = structured_rectangle(1, 1)
    shifted = TriMesh(base.vertices + np.array([1.0, 0.0]), base.faces)
    chart0 = Chart("c0", base)
    chart1 = Chart("c1", shifted)
    source_transition = AffineTransition("c0", "c1", np.eye(2), np.array([1.0, 0.0]))
    target_transition = AffineTransition("c0", "c1", np.eye(2), np.array([1.0, 0.0]))
    overlap = Overlap(
        "c0",
        "c1",
        np.arange(base.n_vertices),
        np.arange(base.n_vertices),
        source_transition,
        target_transition,
    )
    return Atlas((chart0, chart1), (overlap,))


def test_atlas_validates_source_transition_and_round_trip():
    atlas = _two_shifted_charts()
    overlap = atlas.overlaps[0]
    source = atlas.chart("c0").mesh.vertices
    mapped = overlap.source_transition.apply(source)
    assert np.allclose(mapped, atlas.chart("c1").mesh.vertices)
    assert np.allclose(overlap.source_transition.inverse().apply(mapped), source)


def test_transition_compatibility_accepts_known_maps_and_raw_equality_does_not():
    atlas = _two_shifted_charts()
    correct = build_affine_compatibility(atlas)
    wrong = raw_coordinate_equality_constraints(atlas)
    x = atlas.pack_maps(
        {
            "c0": atlas.chart("c0").mesh.vertices,
            "c1": atlas.chart("c1").mesh.vertices,
        }
    )
    assert np.linalg.norm(correct.C @ x - correct.d, ord=np.inf) < 1e-12
    assert np.linalg.norm(wrong.C @ x - wrong.d, ord=np.inf) > 0.9


def test_coupled_lsqc_reconstructs_two_chart_identity_with_hard_seams():
    atlas = _two_shifted_charts()
    seam = build_affine_compatibility(atlas)
    pins0 = fixed_vertex_constraints(
        atlas.chart("c0").mesh.n_vertices,
        np.array([0, 3]),
        atlas.chart("c0").mesh.vertices[[0, 3]],
    )
    constraints = atlas.embed_chart_constraints("c0", pins0).stack(seam)
    result = solve_coupled_lsqc(
        atlas,
        {"c0": np.zeros(2, complex), "c1": np.zeros(2, complex)},
        constraints,
    )
    assert result.algebraic_residual < 1e-10
    assert result.seam_residual < 1e-10
    assert np.max(np.abs(result.maps["c0"] - atlas.chart("c0").mesh.vertices)) < 1e-9
    assert np.max(np.abs(result.maps["c1"] - atlas.chart("c1").mesh.vertices)) < 1e-9


def test_nonlinear_transition_jacobian_matches_finite_difference_and_gn_reduces_residual():
    source = np.array([[0.2, -0.1], [0.5, 0.3]])
    target = np.array([[0.4, 0.0], [0.8, 0.4]])

    def transition(points):
        return np.column_stack((points[:, 0] + 0.2 * points[:, 1] ** 2, points[:, 1]))

    def jacobian(points):
        result = np.tile(np.eye(2), (len(points), 1, 1))
        result[:, 0, 1] = 0.4 * points[:, 1]
        return result

    residual, matrix = nonlinear_compatibility_residual_jacobian(
        source, target, transition, jacobian
    )
    direction = np.array([[0.1, -0.2], [-0.3, 0.25]])
    epsilon = 1e-6
    fd = (
        (target - transition(source + epsilon * direction)).reshape(-1)
        - (target - transition(source - epsilon * direction)).reshape(-1)
    ) / (2 * epsilon)
    predicted = matrix[:, : source.size] @ direction.reshape(-1)
    assert np.allclose(fd, predicted, atol=1e-7)
    step = np.linalg.lstsq(matrix, -residual, rcond=None)[0]
    updated_source = source + step[: source.size].reshape(source.shape)
    updated_target = target + step[source.size :].reshape(target.shape)
    assert np.linalg.norm(updated_target - transition(updated_source)) < np.linalg.norm(residual)


def test_conformal_source_chart_rotation_obeys_beltrami_covariance():
    mu = np.array([0.2 + 0.1j])
    angle = 0.37
    derivative = np.exp(1j * angle)
    transformed = conformal_beltrami_transition(mu, derivative)
    assert np.allclose(transformed, mu * np.exp(2j * angle))
