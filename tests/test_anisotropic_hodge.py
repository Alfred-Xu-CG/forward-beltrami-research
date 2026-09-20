import numpy as np

from qcopt.forward.anisotropic_hodge import diamond_stencil_decomposition


def test_diamond_stencil_reconstructs_the_tensor_when_positive():
    tensor = np.array([[2.0, -0.5], [-0.5, 3.0]])
    result = diamond_stencil_decomposition(tensor)
    assert result.feasible
    assert np.all(result.conductances >= 0.0)
    assert np.allclose(result.reconstructed, tensor)


def test_diamond_stencil_reports_negative_axis_conductance_outside_positive_region():
    tensor = np.array([[1.0, 1.2], [1.2, 2.0]])
    result = diamond_stencil_decomposition(tensor)
    assert not result.feasible
    assert result.conductances[0] < 0.0
    assert np.allclose(result.reconstructed, tensor)


def test_diamond_stencil_rejects_non_symmetric_or_non_spd_tensor():
    with np.testing.assert_raises(ValueError):
        diamond_stencil_decomposition(np.array([[1.0, 0.1], [0.2, 1.0]]))
    with np.testing.assert_raises(ValueError):
        diamond_stencil_decomposition(np.array([[1.0, 2.0], [2.0, 1.0]]))
