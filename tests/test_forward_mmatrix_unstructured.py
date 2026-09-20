import numpy as np

from qcopt.forward.mmatrix import beltrami_conductivity, edge_direction_conductances


def test_unstructured_edge_direction_fit_keeps_nonnegative_weights():
    tensor = beltrami_conductivity(0.42 + 0.27j)
    directions = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    fit = edge_direction_conductances(tensor, directions)
    assert np.all(fit.values >= -1e-12)
    assert np.isfinite(fit.residual)


def test_unstructured_edge_direction_fit_rejects_degenerate_directions():
    tensor = beltrami_conductivity(0.2 + 0.1j)
    try:
        edge_direction_conductances(tensor, np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("degenerate direction dictionary should be rejected")
