"""Image-conditioned repeated safe updates on one nested Q1 vertex table.

The fixed source grid is regular at each dyadic level; its current image is
not assumed regular. Proposal vectors use its *current* centered edge vectors
and its current twelve incident corner-area budgets, not a fixed alpha*h.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .digital_q1 import (AdaptiveSoftRadialQ1Relaxation, FixedSpanSoftRadialQ1Relaxation,
                         SafeColoredQ1Relaxation, StaggeredPatchQ1Layer,
                         q1_dyadic_refine)
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
                 raw_span: float = 8.0,
                 refresh_moving_evidence: bool = True,
                 proposal_mode: str = "current_edge",
                 proposal_modes_by_side: dict[int, str] | None = None,
                 update_families_by_side: dict[int, str] | None = None,
                 checkpoint_rounds: bool = False) -> None:
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
        self.refresh_moving_evidence = refresh_moving_evidence
        if proposal_mode not in ("current_edge", "fixed_h", "fixed_h_soft"):
            raise ValueError("proposal_mode must be current_edge, fixed_h or fixed_h_soft")
        self.proposal_mode = proposal_mode
        self.checkpoint_rounds = checkpoint_rounds
        modes = ({side: proposal_mode for side in self.rounds_by_side}
                 if proposal_modes_by_side is None else dict(proposal_modes_by_side))
        if set(modes) != set(self.rounds_by_side) or any(
            mode not in ("current_edge", "fixed_h", "fixed_h_soft")
            for mode in modes.values()
        ):
            raise ValueError("one valid proposal mode per recurrent side required")
        self.proposal_modes_by_side = modes
        families = ({side: "f1" for side in self.rounds_by_side}
                    if update_families_by_side is None else dict(update_families_by_side))
        if set(families) != set(self.rounds_by_side) or any(
            family not in ("f1", "f2") for family in families.values()
        ):
            raise ValueError("one f1/f2 update family per recurrent side required")
        if any(families[side] == "f2" and modes[side] == "fixed_h_soft"
               for side in families):
            raise ValueError("F2 does not implement fixed_h_soft proposal")
        self.update_families_by_side = families
        width = base.encoder.stem[0].out_channels
        self.heads = nn.ModuleDict({str(side): CurrentImageProposalHead(width)
                                    for side in self.rounds_by_side})
        update_types = {
            "current_edge": AdaptiveSoftRadialQ1Relaxation,
            "fixed_h": SafeColoredQ1Relaxation,
            "fixed_h_soft": FixedSpanSoftRadialQ1Relaxation,
        }
        self.updates = nn.ModuleDict({
            str(side): (
                update_types[modes[side]](
                    side, raw_span=raw_span,
                    minimum_jacobian=base.decoder.seed_update.passes[0].minimum_jacobian,
                ) if families[side] == "f1" else
                StaggeredPatchQ1Layer(
                    side, patch_cells=8, proposal_mode=modes[side], raw_span=.5,
                    minimum_jacobian=base.decoder.seed_update.passes[0].minimum_jacobian,
                )
            ) for side in self.rounds_by_side
        })

    def _repeat(self, mapped: torch.Tensor, fixed: torch.Tensor,
                moving: torch.Tensor, features: torch.Tensor,
                matrix: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
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
        level_entry_query = Q1ImageRegistrationNetwork.apply_affine(
            mapped, matrix, offset,
        ) if not self.refresh_moving_evidence else None
        def one_pass(state: torch.Tensor, fraction: float
                     ) -> tuple[torch.Tensor, torch.Tensor]:
            # The returned map is A_internal composed with this residual Q1
            # table. Sample the input moving canvas at that actual current
            # position, not at the un-affined residual coordinate.
            current_query = (Q1ImageRegistrationNetwork.apply_affine(
                state, matrix, offset,
            ) if self.refresh_moving_evidence else level_entry_query)
            warped = F.grid_sample(
                moving, 2 * current_query - 1,
                padding_mode="border", align_corners=False,
            )
            latent = head(static_features, fixed_nodes, warped,
                          state - identity, fraction)
            return state, latent

        def one_round(state: torch.Tensor, round_index: int) -> torch.Tensor:
            if self.update_families_by_side[side] == "f1":
                fraction = (round_index + 1) / self.rounds_by_side[side]
                current, latent = one_pass(state, fraction)
                return update(current, latent)
            current = state
            total = self.rounds_by_side[side] * len(update.passes)
            for offset_index, patch_pass in enumerate(update.passes):
                fraction = (round_index * len(update.passes) + offset_index + 1) / total
                current, latent = one_pass(current, fraction)
                current = patch_pass(current, latent)
            return current

        for round_index in range(self.rounds_by_side[side]):
            if self.checkpoint_rounds and self.training:
                mapped = checkpoint(
                    lambda state, round_index=round_index: one_round(state, round_index),
                    mapped, use_reentrant=False,
                )
            else:
                mapped = one_round(mapped, round_index)
        return mapped

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder, decoder = self.base.encoder, self.base.decoder
        features = encoder.features(fixed, moving)
        matrix, offset = self.base.affine_head(features)
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
        mapped = self._repeat(mapped, fixed, moving, features, matrix, offset)
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
            mapped = self._repeat(mapped, fixed, moving, features, matrix, offset)
        return mapped, matrix, offset
