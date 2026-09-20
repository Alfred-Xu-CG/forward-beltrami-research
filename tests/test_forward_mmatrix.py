import numpy as np

from qcopt.forward.mmatrix import (
    beltrami_conductivity,
    conditioning_safe_dictionary_conductances,
    directional_conductances,
    dictionary_conductances,
    integer_wide_stencil_conductances,
    integer_wide_stencil_directions,
    spectral_conductances,
)


def test_moderate_beltrami_tensor_has_nonnegative_directional_split():
    tensor = beltrami_conductivity(0.2 + 0.1j)
    split = directional_conductances(tensor)

    assert np.linalg.eigvalsh(tensor).min() > 0.0
    assert split.monotone_possible
    assert min(split.axis_x, split.axis_y, split.diagonal_plus, split.diagonal_minus) >= 0.0


def test_near_unit_diagonal_anisotropy_can_break_the_simple_mmatrix_split():
    tensor = beltrami_conductivity(0.9 * np.exp(1j * np.pi / 4.0))
    split = directional_conductances(tensor)

    assert np.linalg.eigvalsh(tensor).min() > 0.0
    assert not split.monotone_possible
    assert split.axis_x < 0.0 or split.axis_y < 0.0


def test_spectral_conductances_are_positive_but_not_mesh_direction_constraints():
    tensor = beltrami_conductivity(0.9 * np.exp(1j * np.pi / 4.0))
    spectral = spectral_conductances(tensor)
    reconstructed = sum(
        value * np.outer(direction, direction)
        for value, direction in zip(spectral.values, spectral.directions)
    )
    assert np.all(spectral.values > 0.0)
    assert np.allclose(reconstructed, tensor, atol=1e-12)


def test_finer_direction_dictionary_reduces_adaptive_stencil_error():
    tensor = beltrami_conductivity(0.88 * np.exp(0.37j))
    coarse = dictionary_conductances(tensor, n_directions=8)
    fine = dictionary_conductances(tensor, n_directions=128)
    assert np.all(coarse.values >= 0.0)
    assert np.all(fine.values >= 0.0)
    assert fine.residual < coarse.residual
    assert fine.residual < 1e-2


def test_conditioning_safe_dictionary_stops_at_accurate_small_dictionary():
    tensor = beltrami_conductivity(0.73 * np.exp(0.4j))
    fit = conditioning_safe_dictionary_conductances(tensor, residual_tolerance=1e-8)
    assert fit.directions.shape[0] <= 32
    assert fit.residual < 1e-8
    assert np.all(fit.values >= -1e-12)


def test_integer_wide_stencil_uses_positive_regular_grid_directions():
    directions = integer_wide_stencil_directions(3)
    assert directions.shape[1] == 2
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert len(directions) == 16
    fit = integer_wide_stencil_conductances(
        beltrami_conductivity(0.9 * np.exp(0.37j)), max_step=3
    )
    assert np.all(fit.values >= -1e-12)
    assert np.isfinite(fit.residual)

    # A tensor assembled from available grid directions is exactly
    # representable by the same positive wide-stencil dictionary.
    directions = integer_wide_stencil_directions(3)
    tensor = 2.0 * np.outer(directions[0], directions[0]) + 3.0 * np.outer(
        directions[1], directions[1]
    )
    exact = integer_wide_stencil_conductances(tensor, max_step=3)
    assert exact.residual < 1e-5


def test_batch_positive_directional_fit_recovers_variable_cone_tensors():
    from qcopt.forward.mmatrix import batch_positive_directional_conductances

    directions = integer_wide_stencil_directions(2)
    weights = np.array(
        [[1.0 + 0.2 * i, 0.7 + 0.1 * i, 0.4, 0.3, 0.2, 0.1, 0.05, 0.08]
         for i in range(3)],
        dtype=np.float64,
    )
    tensors = np.einsum("nm,mi,mj->nij", weights, directions, directions)
    fit = batch_positive_directional_conductances(tensors, directions)
    assert fit.values.shape == weights.shape
    assert np.all(fit.values >= -1e-12)
    assert np.max(fit.residual) < 1e-10


def test_spatial_smooth_positive_fit_preserves_exact_cone_and_nonnegativity():
    from qcopt.forward.mmatrix import integer_wide_stencil_directions
    from qcopt.forward.mmatrix_smooth import smooth_positive_directional_conductances

    directions = integer_wide_stencil_directions(2)
    weights = np.zeros((6, 7, directions.shape[0]), dtype=np.float64)
    weights[..., 0] = 1.0 + 0.1 * np.sin(np.arange(6)[:, None])
    weights[..., 3] = 0.8 + 0.1 * np.cos(np.arange(7)[None, :])
    tensors = np.einsum("...m,mi,mj->...ij", weights, directions, directions)
    selected, residual = smooth_positive_directional_conductances(tensors, directions)
    assert np.all(selected >= -1e-12)
    assert np.max(residual) < 1e-10
    reconstructed = np.einsum("...m,mi,mj->...ij", selected, directions, directions)
    assert np.allclose(reconstructed, tensors, atol=1e-10)
