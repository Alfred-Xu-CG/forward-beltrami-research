"""First-order multilevel latent decoder on ONE fixed material vertex grid.

Raw source-basis proposals are prolonged, not accepted maps. Each x/y update
protects the actual rounded output. A globally injective initial map and its
unchanged simple boundary remain assumptions; no encoder or universality claim
is made. Nested coarse spaces lie in the finest coefficient space, so repeating
updates does not create higher-frequency source basis functions.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .coordinated_explicit_vjp import explicit_coordinated_candidate
from .coordinated_update import CoordinatedQ1Update, interpolate_proposal


class _DirectionalStage(torch.nn.Module):
    def __init__(self, rows, columns, level, component, stage_index, backend,
                 minimum_jacobian, theta, amplitude):
        super().__init__()
        self.shape = (rows, columns)
        self.level, self.component, self.stage_index = level, component, stage_index
        self.backend, self.amplitude = backend, amplitude
        self.layer = CoordinatedQ1Update(
            (1., 0.) if component == 0 else (0., 1.), mode="analytic",
            minimum_jacobian=minimum_jacobian, theta=theta)

    def forward(self, vertices, latent, reference):
        coefficients = F.pad(self.amplitude * 16 / (self.level - 1) *
                             torch.tanh(latent[:, self.component]), (1, 1, 1, 1))
        proposal = interpolate_proposal(coefficients, self.shape)
        try:
            if self.backend == "existing":
                return self.layer(vertices, proposal, reference=reference,
                                  validate=True).vertices
            return explicit_coordinated_candidate(
                vertices, proposal, reference=reference, direction=self.layer.direction,
                mode="analytic", minimum_jacobian=self.layer.minimum_jacobian,
                theta=self.layer.theta, validate=True, backward_backend="torch_manual")
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"coordinated stage {self.stage_index}, level {self.level}, "
                               f"component {self.component}: {error}") from error


class _LevelBlock(torch.nn.Module):
    """Immutable bound x/y block; changing tensors are explicit inputs."""
    def __init__(self, rows, columns, level, index, backend, minimum_jacobian,
                 theta, amplitude):
        super().__init__()
        self.x = _DirectionalStage(rows, columns, level, 0, 2 * index, backend,
                                   minimum_jacobian, theta, amplitude)
        self.y = _DirectionalStage(rows, columns, level, 1, 2 * index + 1, backend,
                                   minimum_jacobian, theta, amplitude)

    def forward(self, vertices, latent, reference):
        return self.y(self.x(vertices, latent, reference), latent, reference)


class CoordinatedMultilevelDecoder(torch.nn.Module):
    """Sequential analytic coordinated candidates, with ALL latent/Y gradients.

    Latents are a sequence of (B,2,L-2,L-2) tensors, one per coefficient level.
    Geometry/reference and latents must share dtype/device. ``initial_vertices``
    may be None to start at the constant source reference. Reference is copied
    and must not require gradients. Manual backends support FIRST derivatives
    only; checkpointed_manual recomputes each bound x/y block during backward.
    ``.to`` casts the stored reference like an ordinary module buffer; every
    stage still checks the actual float64 corner determinants of that buffer.
    """
    def __init__(self, rows: int, columns: int, levels: Sequence[int], *,
                 reference: torch.Tensor | None = None, backend: str = "existing",
                 minimum_jacobian: float = .001, theta: float = .95,
                 amplitude: float = .005, diagonal: str = "ac"):
        super().__init__()
        if (not isinstance(rows, int) or not isinstance(columns, int) or
                min(rows, columns) < 3):
            raise ValueError("rows/columns must be integers >=3")
        levels = tuple(levels)
        if (not levels or any(not isinstance(l, int) or isinstance(l, bool) or
                              l < 3 or l > min(rows, columns) for l in levels) or
                any(b <= a or (b - 1) % (a - 1) for a, b in zip(levels, levels[1:]))):
            raise ValueError("strictly increasing nested coefficient levels required")
        if backend not in ("existing", "manual", "checkpointed_manual"):
            raise ValueError("backend existing/manual/checkpointed_manual required")
        if diagonal not in ("ac", "bd"):
            raise ValueError("fixed diagonal ac/bd required")
        if not math.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("amplitude must be positive and finite")
        if reference is None:
            yy, xx = torch.meshgrid(torch.linspace(0, 1, rows, dtype=torch.float64),
                                    torch.linspace(0, 1, columns, dtype=torch.float64),
                                    indexing="ij")
            reference = torch.stack((xx, yy), -1)[None]
        if (reference.requires_grad or reference.ndim != 4 or reference.shape[0] < 1 or
                reference.shape[1:] != (rows, columns, 2) or
                reference.dtype not in (torch.float32, torch.float64)):
            raise ValueError("constant float32/64 reference (B,R,C,2) required")
        self.register_buffer("reference", reference.detach().clone())
        self.rows, self.columns, self.levels = rows, columns, levels
        self.backend, self.diagonal = backend, diagonal
        self.stage_count = 2 * len(levels)
        self.full_control_latent = rows == columns == levels[-1]
        self.blocks = torch.nn.ModuleList([
            _LevelBlock(rows, columns, l, i, backend, minimum_jacobian, theta, amplitude)
            for i, l in enumerate(levels)])

    def forward(self, initial_vertices: torch.Tensor | None,
                latents: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(latents) != len(self.levels):
            raise ValueError("one latent tensor per coefficient level required")
        batch = latents[0].shape[0] if latents[0].ndim == 4 else 0
        if batch < 1 or self.reference.shape[0] not in (1, batch):
            raise ValueError("positive common latent batch and reference batch1/B required")
        for latent, level in zip(latents, self.levels):
            if (latent.shape != (batch, 2, level - 2, level - 2) or
                    latent.dtype != self.reference.dtype or
                    latent.device != self.reference.device):
                raise ValueError("latent (B,2,L-2,L-2) and reference dtype/device must match")
        if initial_vertices is None:
            vertices = self.reference.expand(batch, -1, -1, -1)
        else:
            if (initial_vertices.shape != (batch, self.rows, self.columns, 2) or
                    initial_vertices.dtype != self.reference.dtype or
                    initial_vertices.device != self.reference.device):
                raise ValueError("initial vertices must match grid/batch/reference dtype/device")
            vertices = initial_vertices
        for block, latent in zip(self.blocks, latents):
            if self.backend == "checkpointed_manual":
                vertices = checkpoint(block, vertices, latent, self.reference,
                                      use_reentrant=False)
            else:
                vertices = block(vertices, latent, self.reference)
        return vertices
