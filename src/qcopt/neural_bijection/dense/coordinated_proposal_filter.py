"""Linear zero-ghost smoothing of RAW interior proposal coefficients only."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dirichlet_lazy_proposal_filter(raw: torch.Tensor, steps: int = 4) -> torch.Tensor:
    """Apply a lazy five-point Dirichlet stencil ``steps`` times to (B,C,M,N).

    Every pass is 0.5*center + 0.125*each of four neighbors, with zero ghosts
    outside the INTERIOR coefficient rectangle. M,N>=1; float32/64, finite raw
    values and a constant nonnegative Python integer pass count are required.
    Steps zero returns the input itself. Output shape/dtype/device are unchanged.
    The caller subsequently zero-pads the actual fixed coefficient boundary and
    interpolates RAW proposals before a safe update; this never repairs or
    smooths an accepted deformation and has no geometry-dependent state.

    On the separable sine basis, eigenvalues are
    .5 + .25*cos(pi*k/(M+1)) + .25*cos(pi*l/(N+1)), strictly between zero and one.
    Thus the finite-grid operator and its integer powers are symmetric positive
    definite and invertible in EXACT arithmetic: unrestricted raw latent freedom
    is not rank-reduced. Their VJP is the same stencil power. Strong high-frequency
    attenuation can nevertheless be numerically ill-conditioned; bounded latent
    ranges, finite precision, Adam/momentum and finite optimization budgets can
    change attainable updates. No inverse or optimizer-quality claim is made.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a constant nonnegative Python integer")
    if (not isinstance(raw, torch.Tensor) or raw.ndim != 4 or min(raw.shape) < 1 or
            raw.dtype not in (torch.float32, torch.float64)):
        raise ValueError("raw must have nonempty float32/64 shape (B,C,M,N)")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("raw proposals must be finite")
    result = raw
    for _ in range(steps):
        padded = F.pad(result, (1, 1, 1, 1), mode="constant", value=0.)
        result = (.5 * result + .125 * padded[..., :-2, 1:-1]
                  + .125 * padded[..., 2:, 1:-1]
                  + .125 * padded[..., 1:-1, :-2]
                  + .125 * padded[..., 1:-1, 2:])
    return result
