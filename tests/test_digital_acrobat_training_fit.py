"""Train-set diagnostics use the same Euclidean vertex metric as held-out evaluation."""

import torch

from tools.digital_acrobat_training_fit import split_vertex_rmse


def test_split_vertex_rmse_separates_interior_and_boundary():
    target = torch.zeros(1, 3, 3, 2)
    predicted = target.clone()
    predicted[0, 1, 1, 0] = 3
    predicted[0, 0, 0, 1] = 4
    result = split_vertex_rmse(predicted, target)
    assert abs(result["interior"] - 3.0) < 1e-7
    assert abs(result["boundary"] - (4 / 8**.5)) < 1e-7
    assert abs(result["all"] - (25 / 9)**.5) < 1e-7
