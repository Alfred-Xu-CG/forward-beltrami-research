import numpy as np

from phase6_frequency_gate_threshold_transfer import _fit_threshold, _rates
from phase6_posthoc_frequency_gate_routing import _route


def test_scalar_threshold_uses_both_classes() -> None:
    low = np.array([0.1, 0.2])
    high = np.array([0.4, 0.5])
    threshold, train = _fit_threshold(low, high)
    assert 0.2 < threshold < 0.4
    assert train["balanced_accuracy"] == 1.0
    assert _rates(low, high, threshold)["false_positive_rate_high32"] == 0.0


def test_posthoc_route_selects_entire_sample_not_metric_average() -> None:
    neutral = [
        {"image_mse": 1.0, "query_map_mse": 1.0, "face_beltrami_mse": 1.0,
         "minimum_face_determinant": 0.4, "maximum_predicted_beltrami_modulus": 0.5},
        {"image_mse": 2.0, "query_map_mse": 4.0, "face_beltrami_mse": 9.0,
         "minimum_face_determinant": 0.3, "maximum_predicted_beltrami_modulus": 0.6},
    ]
    trained = [
        {"image_mse": 3.0, "query_map_mse": 9.0, "face_beltrami_mse": 16.0,
         "minimum_face_determinant": 0.2, "maximum_predicted_beltrami_modulus": 0.7},
        {"image_mse": 4.0, "query_map_mse": 16.0, "face_beltrami_mse": 25.0,
         "minimum_face_determinant": 0.1, "maximum_predicted_beltrami_modulus": 0.8},
    ]
    result = _route(np.array([0.1, 0.9]), 0.5, neutral, trained)
    assert result["trained64_selected"] == 1
    assert result["routed"]["image_mse"] == 2.5
    assert np.isclose(result["routed"]["face_beltrami_rmse"], np.sqrt(13))
    assert result["routed"]["minimum_face_determinant"] == 0.1
