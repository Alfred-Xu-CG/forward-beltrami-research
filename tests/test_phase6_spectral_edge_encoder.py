"""Shape and gradient tests for the fixed-frequency edge-logit adapter."""

from __future__ import annotations

import numpy as np
import torch

from phase6_spectral_edge_encoder import SpectralEdgeImageEncoder
from phase6_train_conductance_image_to_latent import MultiscaleEdgeImageEncoder
from qcopt.mesh import structured_rectangle


def test_spectral_edge_encoder_zero_head_matches_base_and_has_vjp() -> None:
    torch.manual_seed(2408)
    side = 17
    mesh = structured_rectangle(side - 1, side - 1)
    grid = np.arange(side**2).reshape(side, side)
    edges = np.concatenate((
        np.stack((grid[:, :-1].ravel(), grid[:, 1:].ravel()), axis=1),
        np.stack((grid[:-1].ravel(), grid[1:].ravel()), axis=1),
        np.stack((grid[:-1, :-1].ravel(), grid[1:, 1:].ravel()), axis=1),
    ))
    encoder = SpectralEdgeImageEncoder(
        side, mesh.vertices[edges].mean(axis=1),
        width=4, body_mode="local", frequencies=(1, 2, 4))
    pair = torch.randn(2, 2, 33, 33)
    output = encoder(pair)
    baseline = MultiscaleEdgeImageEncoder.forward(encoder, pair)
    assert output.shape == (2, 2 * side * (side - 1) + (side - 1)**2)
    assert torch.allclose(output, baseline, rtol=0, atol=1e-6)
    loss = (output * torch.randn_like(output)).sum()
    gradient = torch.autograd.grad(loss, encoder.spectral_head.weight)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().max()) > 0
