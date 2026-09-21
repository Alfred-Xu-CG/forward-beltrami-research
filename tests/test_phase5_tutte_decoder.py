from __future__ import annotations

import numpy as np
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.decoder import TutteRectangleDecoder
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer


def _decoder(nx: int = 4, ny: int = 3, height: int = 31, width: int = 37):
    mesh = structured_rectangle(nx, ny)
    solver = DirectTutteLayer(mesh)
    decoder = TutteRectangleDecoder(
        mesh,
        solver,
        image_height=height,
        image_width=width,
        minimum_height=0.05,
    )
    return mesh, solver, decoder


def test_neutral_parameters_decode_identity_control_and_dense_maps() -> None:
    mesh, solver, decoder = _decoder()
    interior_logits = torch.zeros(
        solver.system.n_rows, solver.system.max_degree, dtype=torch.float64
    )
    boundary_logits = torch.zeros(decoder.boundary.n_segments, dtype=torch.float64)
    raw_modulus = torch.tensor(
        decoder.boundary.raw_modulus_for_height(1.0), dtype=torch.float64
    )

    result = decoder(interior_logits, boundary_logits, raw_modulus)

    torch.testing.assert_close(
        result.control,
        torch.tensor(mesh.vertices, dtype=torch.float64),
        atol=2e-13,
        rtol=0.0,
    )
    y = torch.linspace(0.0, 1.0, decoder.image_height, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, decoder.image_width, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    torch.testing.assert_close(
        result.dense,
        torch.stack((xx, yy), dim=-1),
        atol=2e-13,
        rtol=0.0,
    )


def test_decoder_backpropagates_to_interior_boundary_and_modulus() -> None:
    _, solver, decoder = _decoder(3, 3, 19, 23)
    generator = torch.Generator().manual_seed(4401)
    interior_logits = (
        0.15
        * torch.randn(
            (solver.system.n_rows, solver.system.max_degree),
            generator=generator,
            dtype=torch.float64,
        )
    ).requires_grad_()
    boundary_logits = (
        0.12
        * torch.randn(decoder.boundary.n_segments, generator=generator, dtype=torch.float64)
    ).requires_grad_()
    raw_modulus = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)

    result = decoder(interior_logits, boundary_logits, raw_modulus)
    weights = torch.linspace(0.2, 1.1, result.dense.numel(), dtype=torch.float64).reshape(
        result.dense.shape
    )
    gradients = torch.autograd.grad((result.dense.square() * weights).mean(), (
        interior_logits,
        boundary_logits,
        raw_modulus,
    ))

    for gradient in gradients:
        assert bool(torch.isfinite(gradient).all())
        assert float(torch.linalg.vector_norm(gradient)) > 0.0


def test_decoder_broadcasts_shared_boundary_across_latent_batch() -> None:
    _, solver, decoder = _decoder(3, 2, 13, 17)
    generator = torch.Generator().manual_seed(72)
    interior_logits = torch.randn(
        (2, solver.system.n_rows, solver.system.max_degree),
        generator=generator,
        dtype=torch.float64,
    ) * 0.1
    boundary_logits = torch.zeros(decoder.boundary.n_segments, dtype=torch.float64)
    raw_modulus = torch.tensor(
        decoder.boundary.raw_modulus_for_height(1.0), dtype=torch.float64
    )

    batched = decoder(interior_logits, boundary_logits, raw_modulus)
    separate = [decoder(interior_logits[index], boundary_logits, raw_modulus) for index in range(2)]

    torch.testing.assert_close(batched.control, torch.stack([item.control for item in separate]))
    torch.testing.assert_close(batched.dense, torch.stack([item.dense for item in separate]))


def test_float32_requires_and_accepts_explicit_query_preparation() -> None:
    _, solver, decoder = _decoder(3, 3, 11, 15)
    interior_logits = torch.zeros(
        solver.system.n_rows, solver.system.max_degree, dtype=torch.float32
    )
    boundary_logits = torch.zeros(decoder.boundary.n_segments, dtype=torch.float32)
    raw_modulus = torch.tensor(0.0, dtype=torch.float32)

    with np.testing.assert_raises_regex(ValueError, "prepare"):
        decoder(interior_logits, boundary_logits, raw_modulus)

    decoder.prepare(device="cpu", dtype=torch.float32)
    result = decoder(interior_logits, boundary_logits, raw_modulus)
    assert result.control.dtype == torch.float32
    assert result.dense.dtype == torch.float32
