"""CPU-only, sequential adapter for the legacy two-factor directed solver."""

from __future__ import annotations

import numpy as np
import torch

from ...forward import tutte_directed_implicit as legacy
from ...mesh import TriMesh


class LegacyReferenceTutteLayer(torch.nn.Module):
    """Legacy/reference baseline, deliberately not the factor-reusing direct layer.

    Accepts CPU float32/float64 logits ``(I,D)`` and ordered boundary ``(B,2)``,
    each optionally with a leading batch dimension. Singleton batches broadcast;
    output is unbatched only when both inputs are unbatched. Samples execute
    sequentially through ``directed_tutte_embedding_torch_implicit``.

    Legacy weights, sparse assembly and solves are float64 even for float32
    public inputs. Every nonempty sample forward calls SciPy ``factorized``
    separately for A and A.T; backward uses the saved transpose factor. There
    is no cross-forward factor cache, GPU solve, or official KLU replication.
    Only first-order derivatives are supported by the inherited NumPy VJP.

    The adapter additionally checks both the legacy float64 normalized
    probabilities and strictly positive faces in the returned dtype. These are
    fail-closed numerical screens, not repair or an exact proof. Boundary
    copies protect the legacy NumPy saved boundary from caller aliases.
    """

    def __init__(self, mesh: TriMesh) -> None:
        super().__init__()
        self.mesh = mesh
        self.system = legacy.DirectedTutteSystem.from_mesh(mesh)

    def forward(self, latent: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        for name, value, shape in (
            ('latent', latent, (self.system.n_rows, self.system.max_degree)),
            ('boundary', boundary, (len(self.system.loop), 2)),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'{name} must be a torch tensor')
            if value.device.type != 'cpu':
                raise ValueError('legacy reference backend is CPU-only; batches are sequential')
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError(f'{name} must be float32 or float64')
            if value.ndim not in (2, 3) or tuple(value.shape[-2:]) != shape:
                raise ValueError(f'{name} must have trailing shape {shape} and optional batch')
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite')
        if latent.dtype != boundary.dtype:
            raise TypeError('latent and boundary must share dtype')
        unbatched = latent.ndim == boundary.ndim == 2
        z = latent.unsqueeze(0) if latent.ndim == 2 else latent
        b = boundary.unsqueeze(0) if boundary.ndim == 2 else boundary
        batch = max(z.shape[0], b.shape[0])
        if batch == 0 or z.shape[0] not in (1, batch) or b.shape[0] not in (1, batch):
            raise ValueError('batch dimensions must match or be singleton and nonempty')
        z, b = z.expand(batch, -1, -1), b.expand(batch, -1, -1)
        outputs = []
        for index in range(batch):
            # Legacy checks exp(logit)>0 before normalization, but an extreme
            # finite value can still round to zero after division by the row
            # sum.  Tutte's theorem needs the realized supported weights to be
            # strictly positive, so the adapter fails closed on that case.
            if self.system.n_rows:
                probabilities = self.system._probabilities(
                    z[index].detach().double().numpy()
                )
                supported = probabilities[self.system.valid_mask]
                if not np.isfinite(supported).all() or np.any(supported <= 0.0):
                    raise ValueError(
                        'legacy normalized supported probabilities must be finite and strictly positive'
                    )
            output = legacy.directed_tutte_embedding_torch_implicit(
                self.mesh, b[index].clone(), z[index], self.system
            )
            legacy._validate_strictly_positive_faces(
                self.system.faces, output.detach().double().numpy()
            )
            outputs.append(output)
        result = torch.stack(outputs)
        return result[0] if unbatched else result
