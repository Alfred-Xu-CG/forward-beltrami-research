from qcopt.forward.electrical_rectangle import solve_isotropic_electrical_rectangle


def test_isotropic_electrical_rectangle_has_exact_tiling_bookkeeping() -> None:
    result = solve_isotropic_electrical_rectangle(9, 7)
    assert abs(result.modulus - 0.75) < 1e-12
    assert result.minimum_cell_width > 0.0
    assert result.minimum_cell_height > 0.0
    assert abs(result.tiling_area - result.target_area) < 1e-12
