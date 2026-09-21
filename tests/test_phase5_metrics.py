from __future__ import annotations

import numpy as np

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.metrics import compute_p1_map_metrics


def test_identity_metrics_have_exact_accuracy_and_positive_topology() -> None:
    mesh = structured_rectangle(4, 3)

    metrics = compute_p1_map_metrics(mesh, mesh.vertices, target=mesh.vertices)

    assert metrics.flip_count == 0
    assert metrics.minimum_signed_area > 0.0
    np.testing.assert_allclose(metrics.minimum_area_ratio, 1.0, atol=2e-15)
    assert metrics.boundary_order_min_gap > 0.0
    assert metrics.global_injectivity_certificate
    assert metrics.map_rmse == 0.0
    assert metrics.maximum_map_error == 0.0
    assert metrics.mu_rmse == 0.0
    assert metrics.maximum_mu_error == 0.0


def test_map_and_beltrami_errors_use_euclidean_and_complex_magnitudes() -> None:
    mesh = structured_rectangle(3, 3)
    target = np.column_stack((1.2 * mesh.vertices[:, 0], 0.8 * mesh.vertices[:, 1]))
    prediction = target + np.asarray((0.03, -0.04))

    metrics = compute_p1_map_metrics(mesh, prediction, target=target)

    np.testing.assert_allclose(metrics.map_rmse, 0.05, atol=2e-15)
    np.testing.assert_allclose(metrics.maximum_map_error, 0.05, atol=2e-15)
    np.testing.assert_allclose(metrics.mu_rmse, 0.0, atol=2e-15)
    np.testing.assert_allclose(metrics.maximum_mu_error, 0.0, atol=2e-15)


def test_orientation_reversal_counts_every_face_and_fails_certificate() -> None:
    mesh = structured_rectangle(3, 2)
    reflected = np.column_stack((1.0 - mesh.vertices[:, 0], mesh.vertices[:, 1]))

    metrics = compute_p1_map_metrics(mesh, reflected)

    assert metrics.flip_count == mesh.n_faces
    assert metrics.minimum_signed_area < 0.0
    assert metrics.minimum_area_ratio < 0.0
    assert not metrics.global_injectivity_certificate
