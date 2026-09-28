"""Image-pair proposals for a safe fixed-grid Q1 residual and positive affine.

The CNN predicts latents, not unconstrained final vertices. The Q1 decoder
provides the residual map U with ordered identity boundary. The affine head
represents A(x)=Mx+b by a bounded LDU factorization with det(M)>0 in exact
arithmetic. The effective map A∘U is Q1 and a homeomorphism onto A([0,1]^2)
when U is valid. A stored floating-point output needs its own sign audit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .forward_p1_encoder import ForwardP1ImageEncoder
from .forward_q1_pyramid import HybridPatchSeedVertexQ1Pyramid


class PositiveAffineHead(nn.Module):
    """Bounded LDU affine: M=L diag(sx,sy) U, sx,sy>0.

    Six raw outputs control two log scales, two shears, and translation. At
    zero weights the map is exactly identity. Bounded parameters prevent
    overflow but do not, by themselves, certify a saved rounded vertex table.
    """

    def __init__(self, width: int, *, max_log_scale: float = 0.3,
                 max_shear: float = 0.3, max_translation: float = 0.2) -> None:
        super().__init__()
        bounds = (max_log_scale, max_shear, max_translation)
        if width < 2 or any(not math.isfinite(value) or not 0 < value <= 2
                            for value in bounds):
            raise ValueError("affine head bounds must be finite and in (0,2]")
        self.max_log_scale = max_log_scale
        self.max_shear = max_shear
        self.max_translation = max_translation
        self.hidden = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(width, width), nn.GELU())
        self.output = nn.Linear(width, 6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.output(self.hidden(features))
        sx = torch.exp(self.max_log_scale * torch.tanh(raw[:, 0]))
        sy = torch.exp(self.max_log_scale * torch.tanh(raw[:, 1]))
        upper = self.max_shear * torch.tanh(raw[:, 2])
        lower = self.max_shear * torch.tanh(raw[:, 3])
        first_row = torch.stack((sx, sx * upper), dim=-1)
        second_row = torch.stack((lower * sx, lower * sx * upper + sy), dim=-1)
        matrix = torch.stack((first_row, second_row), dim=-2)
        offset = self.max_translation * torch.tanh(raw[:, 4:6])
        return matrix, offset


class Q1ImageRegistrationNetwork(nn.Module):
    """Two grayscale images -> safe Q1 residual vertices plus affine factors.

    The image intensity grid and Q1 control grid are distinct. Only the
    decoder's 17/33/...-style grid defines the output Q1 function.
    """

    def __init__(self, *, seed_side: int = 17, final_side: int = 257,
                 width: int = 16, feature_side: int = 257,
                 flow_hint: bool = True) -> None:
        super().__init__()
        self.decoder = HybridPatchSeedVertexQ1Pyramid(seed_side, final_side)
        self.encoder = ForwardP1ImageEncoder(
            seed_side, self.decoder.level_sides,
            seed_passes=self.decoder.seed_passes, feature_side=feature_side,
            width=width, flow_hint=flow_hint,
        )
        self.affine_head = PositiveAffineHead(width)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder.features(fixed, moving)
        seed_features = F.interpolate(
            features, size=(self.decoder.seed_side, self.decoder.seed_side),
            mode="bilinear", align_corners=True,
        )
        seed = tuple(
            head(seed_features)[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            for head in self.encoder.seed_heads
        )
        levels = tuple(
            head(F.interpolate(features, size=(side, side), mode="bilinear",
                               align_corners=True))[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            for side, head in zip(self.decoder.level_sides, self.encoder.level_heads)
        )
        residual = self.decoder(seed, levels)
        matrix, offset = self.affine_head(features)
        return residual, matrix, offset

    @staticmethod
    def apply_affine(residual: torch.Tensor, matrix: torch.Tensor,
                     offset: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 4 or residual.shape[-1] != 2:
            raise ValueError("residual must be BHWC with two coordinates")
        if matrix.shape != (residual.shape[0], 2, 2) or offset.shape != (residual.shape[0], 2):
            raise ValueError("affine factors have wrong shape")
        return torch.einsum("bhwi,bji->bhwj", residual, matrix) + offset[:, None, None, :]
