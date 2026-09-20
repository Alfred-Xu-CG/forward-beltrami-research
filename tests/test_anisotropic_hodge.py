import numpy as np

from qcopt.forward.anisotropic_hodge import diamond_stencil_decomposition


def test_diamond_stencil_reconstructs_the_tensor_when_positive():
    tensor = np.array([[2.0, -0.5], [-0.5, 3.0]])
    minimal = diamond_stencil_decomposition(tensor)
    alternative = diamond_stencil_decomposition(tensor, diagonal_mass=1.0)
    assert minimal.feasible and alternative.feasible
    assert np.all(minimal.conductances >= 0.0)
    assert np.all(alternative.conductances >= 0.0)
    assert np.allclose(minimal.reconstructed, tensor)
    assert np.allclose(alternative.reconstructed, tensor)
    assert not np.allclose(minimal.conductances, alternative.conductances)
    assert minimal.positive_feasible_interval == (0.5, 2.0)


def test_diamond_stencil_reports_negative_axis_conductance_outside_positive_region():
    tensor = np.array([[1.0, 1.2], [1.2, 2.0]])
    result = diamond_stencil_decomposition(tensor)
    assert not result.feasible
    assert result.conductances[0] < 0.0
    assert np.allclose(result.reconstructed, tensor)


def test_diamond_stencil_distinguishes_algebraic_family_from_positive_region():
    result = diamond_stencil_decomposition(np.eye(2), diagonal_mass=1.1)
    assert not result.feasible
    assert result.positive_feasible_interval == (0.0, 1.0)
    assert np.allclose(result.reconstructed, np.eye(2))


def test_diamond_stencil_rejects_non_symmetric_or_non_spd_tensor():
    with np.testing.assert_raises(ValueError):
        diamond_stencil_decomposition(np.array([[1.0, 0.1], [0.2, 1.0]]))
    with np.testing.assert_raises(ValueError):
        diamond_stencil_decomposition(np.array([[1.0, 2.0], [2.0, 1.0]]))
