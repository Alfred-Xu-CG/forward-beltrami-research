import numpy as np

from qcopt.forward.electrical_rectangle import (
    solve_isotropic_electrical_rectangle,
    solve_weighted_electrical_rectangle,
)


def test_isotropic_electrical_rectangle_has_exact_tiling_bookkeeping() -> None:
    result = solve_isotropic_electrical_rectangle(9, 7)
    assert abs(result.modulus - 0.875) < 1e-12
    assert abs(result.graph_energy - 0.875) < 1e-12
    assert abs(result.right_flux - 0.875) < 1e-12
    assert result.minimum_cell_width > 0.0
    assert result.minimum_cell_height > 0.0
    assert abs(result.tiling_area - result.right_flux) < 1e-12


def test_small_grid_effective_conductance_and_area_are_independent_checks() -> None:
    for nx, ny in ((2, 2), (3, 2), (3, 3)):
        result = solve_isotropic_electrical_rectangle(nx, ny)
        expected = ny / (nx - 1)
        assert abs(result.modulus - expected) < 1e-12
        assert abs(result.graph_energy - expected) < 1e-12
        assert abs(result.right_flux - expected) < 1e-12
        assert abs(result.tiling_area - expected) < 1e-12
        assert result.dual_potential.shape == (ny + 1, nx - 1)


def test_nonuniform_positive_rows_have_independent_energy_flux_and_tiling_checks() -> None:
    nx, ny = 5, 4
    row_conductance = np.array([0.5, 1.0, 2.0, 3.0])
    horizontal = np.repeat(row_conductance[:, None], nx - 1, axis=1)
    vertical = np.array(
        [[0.7, 1.1, 1.3, 2.0, 0.9], [1.4, 0.8, 2.2, 1.7, 1.0], [0.6, 1.8, 0.9, 1.2, 2.5]],
        dtype=float,
    )
    result = solve_weighted_electrical_rectangle(horizontal, vertical)
    expected = float(np.sum(row_conductance) / (nx - 1))
    assert abs(result.modulus - expected) < 1e-12
    assert abs(result.graph_energy - expected) < 1e-12
    assert abs(result.right_flux - expected) < 1e-12
    assert abs(result.tiling_area - expected) < 1e-12
    assert result.minimum_cell_width > 0.0
    assert result.minimum_cell_height > 0.0
