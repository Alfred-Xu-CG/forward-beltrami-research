"""Independent comparison of the MVC derivative and covariance right inverse."""

from __future__ import annotations

import numpy as np
import torch

from qcopt.mesh import TriMesh
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer
from qcopt.neural_bijection.tutte.mvc import MeanValueCoordinateEncoder
from qcopt.neural_bijection.tutte.mvc_retraction import CovarianceLogitLift


def _variable_degree_disk() -> TriMesh:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.15],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 5], [1, 4, 5], [4, 0, 5], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=np.int64,
    )
    return TriMesh(vertices, faces)


def _decoded_state():
    mesh = _variable_degree_disk()
    decoder = DirectTutteLayer(mesh)
    encoder = MeanValueCoordinateEncoder(mesh)
    logits = torch.tensor(
        [[0.31, -0.17, 0.24, -0.09, 0.11], [-0.28, 0.19, 0.37, 0.0, 0.0]],
        dtype=torch.float64,
    )
    count = len(mesh.boundary_loops[0])
    angle = torch.arange(count, dtype=torch.float64) * (2.0 * torch.pi / count)
    boundary = torch.stack((1.2 * torch.cos(angle), 0.83 * torch.sin(angle)), dim=-1)
    control = decoder(logits, boundary)
    return mesh, decoder, encoder, boundary, control


def test_mvc_derivative_and_covariance_lift_are_distinct_right_inverses() -> None:
    _, decoder, encoder, boundary, control = _decoded_state()
    canonical = encoder(control)
    direction = torch.zeros_like(control)
    direction[torch.from_numpy(decoder.system.interior)] = torch.tensor(
        [[0.027, -0.018], [-0.016, 0.031]], dtype=torch.float64
    )

    _, mvc_derivative = torch.autograd.functional.jvp(
        lambda vertices: encoder(vertices).logits,
        (control,),
        (direction,),
    )
    covariance = CovarianceLogitLift(decoder.system)(
        control, canonical.probabilities, direction
    ).delta_logits

    mask = torch.from_numpy(decoder.system.valid_mask)
    probabilities = canonical.probabilities
    mvc_dp = probabilities * (
        mvc_derivative - torch.sum(probabilities * mvc_derivative, dim=-1, keepdim=True)
    )
    covariance_dp = probabilities * (
        covariance - torch.sum(probabilities * covariance, dim=-1, keepdim=True)
    )
    # The degree-five row has a genuine barycentric fiber, so the two lifts
    # differ even after eliminating the softmax row-shift gauge.
    assert float(torch.linalg.vector_norm((mvc_dp - covariance_dp)[0, mask[0]])) > 1.0e-4

    epsilon = 2.0e-6

    def decoded_tangent(latent_direction: torch.Tensor) -> torch.Tensor:
        plus = decoder(canonical.logits + epsilon * latent_direction, boundary)
        minus = decoder(canonical.logits - epsilon * latent_direction, boundary)
        return (plus - minus) / (2.0 * epsilon)

    mvc_tangent = decoded_tangent(mvc_derivative)
    covariance_tangent = decoded_tangent(covariance)
    torch.testing.assert_close(mvc_tangent, direction, atol=2.0e-9, rtol=2.0e-8)
    torch.testing.assert_close(covariance_tangent, direction, atol=2.0e-9, rtol=2.0e-8)
    torch.testing.assert_close(
        decoded_tangent(mvc_derivative - covariance),
        torch.zeros_like(direction),
        atol=2.0e-9,
        rtol=0.0,
    )


def test_mvc_projector_is_idempotent_on_its_safe_image() -> None:
    _, decoder, encoder, boundary, control = _decoded_state()
    first = encoder(control)
    canonical_control = decoder(first.logits, boundary)
    second = encoder(canonical_control)
    torch.testing.assert_close(canonical_control, control, atol=2.0e-14, rtol=2.0e-14)
    torch.testing.assert_close(second.logits, first.logits, atol=3.0e-14, rtol=3.0e-14)
    torch.testing.assert_close(
        second.probabilities, first.probabilities, atol=3.0e-14, rtol=3.0e-14
    )

