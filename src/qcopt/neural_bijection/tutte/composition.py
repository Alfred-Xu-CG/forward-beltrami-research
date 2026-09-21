"""Coordinate composition of unit-square Tutte maps, without image resampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .decoder import TutteRectangleDecoder
from .dense_warp import StructuredDenseQueryTable


def compose_control_maps(
    tables: Sequence[StructuredDenseQueryTable],
    controls: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Evaluate ``f_last(...f_1(f_0(q)))`` at the first table's fixed queries.

    This is an interpolation primitive, NOT a topology certifier: arbitrary
    controls (including affine contractions into, but not onto, the square)
    are permitted. Each control must be finite, float32/64, and inside the
    unit square up to 64 dtype eps. Dynamic query interpolation may clamp
    these roundoff excursions and has zero outward-coordinate VJP there.

    Controls may have shape (V,2) or (B,V,2), with singleton/unbatched layer
    broadcasting. A batched layer makes the output batched. Query working
    storage is O(BQ) per layer, not O(BQV); autograd retains layer states.
    """
    if not tables or len(tables) != len(controls):
        raise ValueError("one table and control tensor are required per layer")
    for table, control in zip(tables, controls):
        if control.dtype not in (torch.float32, torch.float64):
            raise ValueError("layer controls must have float32 or float64 dtype")
        if control.ndim not in (2, 3) or control.shape[-2:] != (table.control_vertices, 2):
            raise ValueError("layer controls must have shape (V,2) or (B,V,2)")
        if not bool(torch.isfinite(control).all()):
            raise ValueError("layer controls must be finite")
        tolerance = 64.0 * torch.finfo(control.dtype).eps
        if bool(torch.any(control < -tolerance)) or bool(torch.any(control > 1 + tolerance)):
            raise ValueError("every layer must remain in the unit square")
        if control.device != controls[0].device or control.dtype != controls[0].dtype:
            raise ValueError("layer controls must share dtype and device")

    dense = tables[0].interpolate(controls[0])
    height, width = tables[0].height, tables[0].width
    for table, control in zip(tables[1:], controls[1:]):
        query = dense.reshape((-1, height * width, 2)) if dense.ndim == 4 else dense.reshape(-1, 2)
        points = table.interpolate_points(control, query)
        dense = points.reshape((-1, height, width, 2)) if points.ndim == 3 else points.reshape(height, width, 2)
    return dense


@dataclass(frozen=True)
class TutteCompositionResult:
    """Per-layer solved maps and samples of their composition.

    There is intentionally no original-mesh P1 certificate for ``dense``.
    A composition of certified square homeomorphisms is a homeomorphism,
    but its affine partition is a refinement, not the original mesh or the
    sampled image grid. Reinterpolating samples on either can alter topology.
    """

    boundaries: tuple[torch.Tensor, ...]
    controls: tuple[torch.Tensor, ...]
    dense: torch.Tensor


class SquareTutteComposition(torch.nn.Module):
    """Solve 1/2/4 unit-square layers and propagate coordinates through them.

    Every decoder supplies a certified solver and ordered rectangle boundary.
    This wrapper relies on that solver contract; it does not turn an arbitrary
    user-supplied solver into a certified one. All layers, including the final
    layer, have fixed realized H=1. Modulus is deliberately not learnable here.
    Each layer is solved once at its own parameter batch size; broadcasting
    a shared layer across another layer's batch does not duplicate its solve.

    The first dense table is precomputed; later layers perform dynamic
    coordinate location, never image resampling. Derivatives exist almost
    everywhere; at P1 edges the chosen triangle supplies a branch derivative.
    No nontrivial affine square homeomorphism fixes all four corners: affine
    contractions are useful interpolation fixtures, not theorem examples.
    """

    def __init__(self, decoders: Sequence[TutteRectangleDecoder]):
        super().__init__()
        if len(decoders) not in (1, 2, 4):
            raise ValueError("composition requires 1, 2, or 4 layers")
        if not all(isinstance(decoder, TutteRectangleDecoder) for decoder in decoders):
            raise TypeError("layers must be TutteRectangleDecoder instances")
        self.decoders = torch.nn.ModuleList(decoders)

    def prepare(self, *, device: torch.device | str, dtype: torch.dtype) -> None:
        """Prepare only the first fixed query table; move modules via ``to``."""
        self.decoders[0].prepare(device=device, dtype=dtype)

    def forward(
        self,
        latents: Sequence[torch.Tensor],
        boundary_logits: Sequence[torch.Tensor],
    ) -> TutteCompositionResult:
        if len(latents) != len(self.decoders) or len(boundary_logits) != len(self.decoders):
            raise ValueError("one latent and boundary-logit tensor are required per layer")
        boundaries = []
        controls = []
        for decoder, latent, logits in zip(self.decoders, latents, boundary_logits):
            boundary = decoder.boundary.at_height(logits, 1.0)
            control = decoder.solver(latent, boundary)
            boundaries.append(boundary)
            controls.append(control)
        dense = compose_control_maps([d.query_table for d in self.decoders], controls)
        return TutteCompositionResult(tuple(boundaries), tuple(controls), dense)
