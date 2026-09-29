"""A small exact-formula check for an uncapped, one-vertex F1 latent."""

import math

import torch

from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants


def test_current_edge_uncapped_radial_inverse_on_irregular_grid() -> None:
    side = 17
    h = 1 / (side - 1)
    axis = torch.linspace(0, 1, side, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    vertices = torch.stack((xx + .04 * torch.sin(2 * math.pi * xx)
                            * torch.sin(math.pi * yy), yy), dim=-1)[None]
    center = (side // 2, side // 2)
    q0 = q1_corner_determinants(vertices).reshape(-1)
    assert bool((q0 > 0).all())

    coefficients = []
    for direction in ((1., 0.), (0., 1.)):
        trial = vertices.clone()
        trial[0, center[0], center[1]] += h * torch.tensor(direction, dtype=trial.dtype)
        coefficients.append((q1_corner_determinants(trial).reshape(-1) - q0) / h)
    a = torch.stack(coefficients, dim=-1)
    assert int((a.abs().amax(dim=-1) > 1e-12).sum()) == 12
    margins = torch.minimum(.4 * q0, q0 - .05 * h * h)
    assert bool((margins > 0).all())

    row, column = center
    e = torch.stack(((vertices[0, row, column + 1] - vertices[0, row, column - 1]) / 2,
                     (vertices[0, row + 1, column] - vertices[0, row - 1, column]) / 2), dim=-1)
    assert torch.linalg.det(e) > 0

    def gauge(vector: torch.Tensor) -> torch.Tensor:
        return torch.clamp((-a @ vector / margins).amax(), min=0)

    def decode(latent: torch.Tensor) -> torch.Tensor:
        raw = e @ latent
        return raw / (1 + gauge(raw))

    generator = torch.Generator().manual_seed(290930)
    for latent in torch.randn(100, 2, dtype=torch.float64, generator=generator) * 8:
        displacement = decode(latent)
        assert bool((margins + a @ displacement > 0).all())
        moved = vertices.clone()
        moved[0, center[0], center[1]] += displacement
        actual_q = q1_corner_determinants(moved).reshape(-1)
        torch.testing.assert_close(actual_q, q0 + a @ displacement,
                                   rtol=2e-12, atol=2e-12)
        assert bool((actual_q > q0 - margins).all())
        recovered_raw = displacement / (1 - gauge(displacement))
        recovered_latent = torch.linalg.solve(e, recovered_raw)
        torch.testing.assert_close(recovered_latent, latent, rtol=2e-12, atol=2e-12)

    independent_feasible = 0
    for displacement in torch.randn(100, 2, dtype=torch.float64,
                                    generator=generator) * .01:
        if gauge(displacement) >= 1:
            continue
        independent_feasible += 1
        latent = torch.linalg.solve(e, displacement / (1 - gauge(displacement)))
        torch.testing.assert_close(decode(latent), displacement,
                                   rtol=2e-12, atol=2e-12)
    assert independent_feasible >= 25

    latent = torch.tensor((1.3, -2.1), dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(decode, latent)
    expected = torch.linalg.det(e) / (1 + gauge(e @ latent.detach())) ** 3
    torch.testing.assert_close(torch.linalg.det(jacobian), expected,
                               rtol=2e-12, atol=2e-12)
