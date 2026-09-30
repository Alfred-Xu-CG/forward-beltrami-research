import torch

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_matchopt_distill257 import (
    square_symmetry_coords, square_symmetry_image, square_symmetry_map,
)
from tools.digital_mind_sparse_match_finetune import p1_at_points


def test_four_symmetries_exactly_conjugate_fixed_diagonal_p1():
    axis = torch.linspace(0, 1, 5, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    interior = torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
    vertices = identity + torch.stack((.025 * interior, -.015 * interior), dim=-1)[None]
    query = torch.tensor([[.13, .28], [.38, .17], [.62, .71], [.87, .36]],
                         dtype=torch.float64)
    image = torch.arange(25, dtype=torch.float64).reshape(1, 1, 5, 5)
    assert validate_q1_map(vertices, identity)["valid"]
    for symmetry in range(4):
        transformed = square_symmetry_map(vertices, symmetry)
        transformed_query = square_symmetry_coords(query, symmetry)
        lhs = p1_at_points(transformed, transformed_query)
        rhs = square_symmetry_coords(p1_at_points(vertices, query), symmetry)
        torch.testing.assert_close(lhs, rhs, atol=1e-14, rtol=0)
        assert validate_q1_map(transformed, identity)["valid"]
        torch.testing.assert_close(
            square_symmetry_map(transformed, symmetry), vertices,
            atol=1e-14, rtol=0)
        pixels = square_symmetry_image(image, symmetry)
        torch.testing.assert_close(
            square_symmetry_image(pixels, symmetry), image, atol=0, rtol=0)
