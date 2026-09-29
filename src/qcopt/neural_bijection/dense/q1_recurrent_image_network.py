"""Image-conditioned repeated safe updates on one nested Q1 vertex table.

The fixed source grid is regular at each dyadic level; its current image is
not assumed regular. Proposal vectors use its *current* centered edge vectors
and its current twelve incident corner-area budgets, not a fixed alpha*h.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .digital_q1 import AdaptiveSoftRadialQ1Relaxation, q1_dyadic_refine
from .q1_image_network import Q1ImageRegistrationNetwork


class CurrentImageProposalHead(nn.Module):
    """Shared within-level head conditioned on image evidence and current map."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Conv2d(width + 5, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.output = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, static_features: torch.Tensor, fixed_nodes: torch.Tensor,
                warped_moving: torch.Tensor, displacement: torch.Tensor,
                round_fraction: float) -> torch.Tensor:
        batch, _, height, width = static_features.shape
        marker = static_features.new_full((batch, 1, height, width), round_fraction)
        evidence = torch.cat((
            static_features, fixed_nodes, warped_moving,
            displacement.permute(0, 3, 1, 2), marker,
        ), dim=1)
        return self.output(self.hidden(evidence))[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)


class RecurrentQ1ImageRegistrationNetwork(nn.Module):
    """Frozen one-pass base plus trainable repeated full-vertex corrections.

    Repeated passes occur at selected levels, then exact Q1 dyadic refinement
    passes the *same function* to the next level. There is no composition of
    independent maps and no post-composition resampling. All new heads start
    at zero, so the initial output equals the frozen base map up to ordinary
    repeated zero-step floating-point operations.
    """

    def __init__(self, base: Q1ImageRegistrationNetwork, *,
                 rounds_by_side: dict[int, int] | None = None,
                 raw_span: float = 8.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        schedule = rounds_by_side or {base.decoder.seed_side: 4, 33: 2, 65: 1}
        valid_sides = {base.decoder.seed_side, *base.decoder.level_sides}
        if not schedule or any(side not in valid_sides or rounds < 1
                               for side, rounds in schedule.items()):
            raise ValueError("nonempty positive rounds on existing levels required")
        self.rounds_by_side = dict(sorted(schedule.items()))
        width = base.encoder.stem[0].out_channels
        self.heads = nn.ModuleDict({str(side): CurrentImageProposalHead(width)
                                    for side in self.rounds_by_side})
        self.updates = nn.ModuleDict({
            str(side): AdaptiveSoftRadialQ1Relaxation(
                side, raw_span=raw_span,
                minimum_jacobian=base.decoder.seed_update.passes[0].minimum_jacobian,
            ) for side in self.rounds_by_side
        })

    def _repeat(self, mapped: torch.Tensor, fixed: torch.Tensor,
                moving: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        side = mapped.shape[1]
        if side not in self.rounds_by_side:
            return mapped
        batch = mapped.shape[0]
        axis = torch.arange(side, dtype=mapped.dtype, device=mapped.device) / (side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        identity = torch.stack((xx, yy), dim=-1)[None].expand(batch, -1, -1, -1)
        static_features = F.interpolate(features, size=(side, side),
                                        mode="bilinear", align_corners=True)
        fixed_nodes = F.grid_sample(
            fixed, 2 * identity - 1, padding_mode="border", align_corners=False,
        )
        head, update = self.heads[str(side)], self.updates[str(side)]
        for round_index in range(self.rounds_by_side[side]):
            warped = F.grid_sample(
                moving, 2 * mapped - 1, padding_mode="border", align_corners=False,
            )
            latent = head(static_features, fixed_nodes, warped,
                          mapped - identity,
                          (round_index + 1) / self.rounds_by_side[side])
            mapped = update(mapped, latent)
        return mapped

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder, decoder = self.base.encoder, self.base.decoder
        features = encoder.features(fixed, moving)
        batch = fixed.shape[0]
        seed_side = decoder.seed_side
        seed_features = F.interpolate(features, size=(seed_side, seed_side),
                                      mode="bilinear", align_corners=True)
        seed_logits = tuple(
            head(seed_features)[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            for head in encoder.seed_heads
        )
        axis = torch.arange(seed_side, device=fixed.device, dtype=fixed.dtype) / (seed_side - 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        identity = torch.stack((xx, yy), dim=-1)[None].expand(batch, -1, -1, -1)
        mapped = decoder.seed_update(identity, seed_logits)
        mapped = self._repeat(mapped, fixed, moving, features)
        for side, level_head, level_update in zip(
            decoder.level_sides, encoder.level_heads, decoder.level_updates,
            strict=True,
        ):
            latent = level_head(F.interpolate(
                features, size=(side, side), mode="bilinear", align_corners=True,
            ))[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1)
            mapped = q1_dyadic_refine(mapped)
            new_vertices = getattr(decoder, f"_new_mask_{side}")
            mapped = level_update(mapped, torch.where(
                new_vertices, latent, torch.zeros_like(latent),
            ))
            mapped = self._repeat(mapped, fixed, moving, features)
        matrix, offset = self.base.affine_head(features)
        return mapped, matrix, offset
