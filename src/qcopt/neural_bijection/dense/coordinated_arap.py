"""Conventional P1 distance-to-SO(2) prior on the fixed uniform source mesh."""
from __future__ import annotations

import torch


def p1_arap_energy(vertices: torch.Tensor, diagonal: str = "ac", validate: bool = True) -> torch.Tensor:
    """Return .5*mean_actual_faces ||J-R*(J)||_F^2, averaged over batch.

    ``vertices`` is a float32/64 (B,R>=2,C>=2,2) table on the uniform UNIT
    source rectangle. The declared fixed-source P1 diagonal is ac or bd with
    a,b,c,d=(00,10,11,01). AC uses Q1 corner Jacobians J2/J4; BD uses J1/J3.
    Horizontal and vertical source spacings are independent. Uniform source
    triangles have equal area, so the face mean is the source-area average.

    For positive determinant, trace t=J00+J11 and skew s=J10-J01 give the
    unique closest proper rotation [[t,-s],[s,t]]/hypot(t,s). The squared
    residual is evaluated directly, avoiding subtraction of near-equal norm
    energies. Ordinary AD stays connected through this rotation; there is no
    epsilon, clamp, detached rotation, SVD, inverse or global iteration.

    Shape/dtype/diagonal/bool guards always apply. ``validate=True`` also
    requires finite coordinates and finite STRICTLY positive actual P1 face
    determinants. The trusted ``False`` path skips these checks and can yield
    nonfinite results for bad maps; the caller must reject them. This energy
    is NOT a determinant barrier or a four-corner/global topology certificate.
    """
    if diagonal not in ("ac", "bd"):
        raise ValueError("diagonal must be ac or bd")
    if not isinstance(validate, bool):
        raise ValueError("validate must be a Python bool")
    if (not isinstance(vertices, torch.Tensor) or vertices.ndim != 4 or
            vertices.shape[0] < 1 or vertices.shape[-1] != 2 or
            min(vertices.shape[1:3]) < 2 or vertices.dtype not in (torch.float32, torch.float64)):
        raise ValueError("vertices must have float32/64 shape (B>=1,R>=2,C>=2,2)")
    if validate and not bool(torch.isfinite(vertices).all()):
        raise ValueError("vertices must be finite")
    a, b = vertices[:, :-1, :-1], vertices[:, :-1, 1:]
    d, c = vertices[:, 1:, :-1], vertices[:, 1:, 1:]
    if diagonal == "ac":
        dx = torch.stack((b-a, c-d), -2) * (vertices.shape[2]-1)
        dy = torch.stack((c-b, d-a), -2) * (vertices.shape[1]-1)
    else:
        dx = torch.stack((b-a, c-d), -2) * (vertices.shape[2]-1)
        dy = torch.stack((d-a, c-b), -2) * (vertices.shape[1]-1)
    if validate:
        determinant = dx[..., 0]*dy[..., 1] - dx[..., 1]*dy[..., 0]
        if not bool(torch.isfinite(determinant).all() and (determinant > 0).all()):
            raise ValueError("actual P1 face determinants must be finite and strictly positive")
    trace = dx[..., 0] + dy[..., 1]
    skew = dx[..., 1] - dy[..., 0]
    norm = torch.hypot(trace, skew)
    cosine, sine = trace/norm, skew/norm
    residual_squared = ((dx[..., 0]-cosine).square() + (dy[..., 0]+sine).square()
                        + (dx[..., 1]-sine).square() + (dy[..., 1]-cosine).square())
    return .5 * residual_squared.mean()
