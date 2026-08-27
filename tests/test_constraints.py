import numpy as np

from qcopt.constraints import (
    fixed_vertex_constraints,
    rectangle_sliding_constraints,
    two_pin_constraints,
)
from qcopt.mesh import structured_rectangle


def test_two_complex_pins_fix_both_coordinates_in_uv_block_order():
    constraints = two_pin_constraints(5, [1, 4], np.array([[0.2, 0.3], [1.1, -0.7]]))
    x = np.zeros(10)
    x[[1, 4, 6, 9]] = [0.2, 1.1, 0.3, -0.7]

    assert constraints.C.shape == (4, 10)
    assert np.allclose(constraints.C @ x, constraints.d)
    assert constraints.rank == 4


def test_fixed_vertex_constraints_preserve_given_order():
    constraints = fixed_vertex_constraints(
        4, np.array([3, 0]), np.array([[0.9, 0.8], [0.1, 0.2]])
    )
    x = np.array([0.1, 7.0, 8.0, 0.9, 0.2, 9.0, 10.0, 0.8])
    assert np.allclose(constraints.C @ x, constraints.d)


def test_rectangle_sliding_fixes_only_side_normal_coordinates():
    mesh = structured_rectangle(2, 2)
    constraints = rectangle_sliding_constraints(mesh)
    n = mesh.n_vertices

    assert constraints.C.shape == (12, 2 * n)
    assert constraints.rank == 12
    assert np.allclose(constraints.C @ mesh.vertices.T.reshape(-1), constraints.d)

    middle_bottom = 1
    middle_left = 3
    constrained_columns = set(constraints.C.indices.tolist())
    assert middle_bottom not in constrained_columns
    assert n + middle_bottom in constrained_columns
    assert middle_left in constrained_columns
    assert n + middle_left not in constrained_columns
