"""Image-conditioned multiscale Fourier edge-logit adapter for Route C."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from phase6_train_conductance_image_to_latent import MultiscaleEdgeImageEncoder


class SpectralEdgeImageEncoder(MultiscaleEdgeImageEncoder):
    """Add image-predicted coefficients of fixed edge-frequency templates.

    This is a deliberately restricted Fourier dictionary, not a generic
    image-to-Beltrami inverse. Positivity remains the downstream sigmoid
    conductance transform and not a property of these unconstrained logits.
    """

    def __init__(
        self, side: int, edge_midpoints: np.ndarray, *,
        width: int = 8, body_mode: str = "context",
        frequencies: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    ) -> None:
        super().__init__(side, edge_midpoints, width=width, body_mode=body_mode)
        if not frequencies or any(value < 1 or value >= (side - 1) // 2 for value in frequencies):
            raise ValueError("frequencies must be positive and below grid Nyquist")
        self.frequencies = frequencies
        count_h = side * (side - 1)
        count_v = count_h
        count_d = (side - 1)**2
        if len(edge_midpoints) != count_h + count_v + count_d:
            raise ValueError("edge midpoint order must be horizontal, vertical, diagonal")
        line = torch.linspace(0, 1, side)
        yy, xx = torch.meshgrid(line, line, indexing="ij")
        image_basis = torch.stack([
            torch.sin(2 * math.pi * mode * xx) * torch.sin(2 * math.pi * mode * yy)
            for mode in frequencies
        ])
        self.register_buffer("image_basis", image_basis, persistent=False)
        midpoint = torch.tensor(np.array(edge_midpoints, copy=True), dtype=torch.float32)
        groups = (midpoint[:count_h], midpoint[count_h:count_h + count_v],
                  midpoint[count_h + count_v:])
        for name, points in zip(("horizontal", "vertical", "diagonal"), groups):
            patterns = []
            for mode in frequencies:
                angle_x = 2 * math.pi * mode * points[:, 0]
                angle_y = 2 * math.pi * mode * points[:, 1]
                horizontal = torch.cos(angle_x) * torch.sin(angle_y)
                vertical = torch.sin(angle_x) * torch.cos(angle_y)
                patterns.append({
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "diagonal": (horizontal + vertical) / math.sqrt(2),
                }[name])
            self.register_buffer(f"{name}_basis", torch.stack(patterns), persistent=False)
        self.spectral_head = torch.nn.Linear((len(frequencies) + 1) * width,
                                             3 * len(frequencies))
        torch.nn.init.zeros_(self.spectral_head.weight)
        torch.nn.init.zeros_(self.spectral_head.bias)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        feature = self.body(F.interpolate(
            pair, size=(self.side, self.side), mode="bilinear", align_corners=True))
        coarse_feature = F.interpolate(
            feature, size=(self.coarse_side, self.coarse_side),
            mode="bilinear", align_corners=True)
        grid = self.edge_grid.expand(pair.shape[0], -1, -1, -1)
        coarse = F.grid_sample(
            self.coarse_head(coarse_feature), grid,
            mode="bilinear", align_corners=True)[:, 0, 0]
        fine = F.grid_sample(
            self.fine_head(feature), grid,
            mode="bilinear", align_corners=True)[:, 0, 0]
        correlated = torch.einsum(
            "bchw,khw->bkc", feature, self.image_basis,
        ) / (self.side**2)
        global_feature = feature.mean(dim=(2, 3))[:, None]
        summary = torch.cat((global_feature, correlated), dim=1).flatten(1)
        coefficients = self.spectral_head(summary).reshape(
            pair.shape[0], 3, len(self.frequencies))
        additions = torch.cat((
            coefficients[:, 0] @ self.horizontal_basis,
            coefficients[:, 1] @ self.vertical_basis,
            coefficients[:, 2] @ self.diagonal_basis,
        ), dim=1)
        return -0.8 + coarse + fine + additions
