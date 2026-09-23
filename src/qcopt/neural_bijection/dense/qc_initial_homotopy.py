"""Optional exact-P1 identity homotopy into a facewise Beltrami cap."""

from __future__ import annotations

import torch


def cap_initial_map(
    mapped: torch.Tensor, *, qc_cap: float = 0.8, safety_fraction: float = 0.99,
) -> torch.Tensor:
    """Scale a fixed-boundary square map from identity before its first QC-root.

    This controls a QC *metric* of an initial map; it is not the primary
    topology mechanism. The guarantee is in exact arithmetic and assumes
    unit-square identity boundary. For a face, the derivative at homotopy
    parameter t has alpha=1+t*(alpha_F-1), beta=t*beta_F. We bound t by the
    first positive root of cap**2*|alpha|**2-|beta|**2, if one lies in (0,1).
    """
    if mapped.ndim != 4 or mapped.shape[-1] != 2 or mapped.shape[1] != mapped.shape[2]:
        raise ValueError("mapped must have shape (batch,side,side,2)")
    if not 0 < qc_cap < 1 or not 0 < safety_fraction < 1:
        raise ValueError("invalid QC cap or safety fraction")
    side = mapped.shape[1]
    if side < 2:
        raise ValueError("side must be at least two")
    h = 1.0 / (side - 1)
    a = mapped[:, :-1, :-1]
    b = mapped[:, :-1, 1:]
    c = mapped[:, 1:, 1:]
    d = mapped[:, 1:, :-1]
    jac_x = torch.stack(((b - a) / h, (c - d) / h), dim=1)
    jac_y = torch.stack(((c - b) / h, (d - a) / h), dim=1)
    # Face axis: lower a-b-c, upper a-c-d. Spatial axes remain the last two
    # before output components, and the reductions below cover both faces.
    alpha_re = 0.5 * (jac_x[..., 0] + jac_y[..., 1]) - 1.0
    alpha_im = 0.5 * (jac_x[..., 1] - jac_y[..., 0])
    beta_re = 0.5 * (jac_x[..., 0] - jac_y[..., 1])
    beta_im = 0.5 * (jac_x[..., 1] + jac_y[..., 0])
    # q(t)=A*t*t+B*t+C, C>0. The stable expression below is the first
    # positive root whenever D>0 and -B+sqrt(D)>0. No root in (0,1) => 1.
    k2 = qc_cap * qc_cap
    coeff_a = k2 * (alpha_re.square() + alpha_im.square()) - (
        beta_re.square() + beta_im.square())
    coeff_b = 2 * k2 * alpha_re
    coeff_c = k2
    discriminant = coeff_b.square() - 4 * coeff_a * coeff_c
    # Inactive faces must not send a sqrt(0) or division-by-zero derivative
    # into the VJP; mask *before* those operations, not merely after them.
    safe_discriminant = torch.where(discriminant > 0, discriminant,
                                    torch.ones_like(discriminant))
    denominator = -coeff_b + torch.sqrt(safe_discriminant)
    safe_denominator = torch.where(denominator > 0, denominator,
                                   torch.ones_like(denominator))
    root = 2 * coeff_c / safe_denominator
    valid = (discriminant > 0) & (denominator > 0) & (root > 0) & (root < 1)
    bound = torch.where(valid, root, torch.ones_like(root))
    scale = torch.minimum(
        bound.amin(dim=(1, 2, 3)) * safety_fraction,
        mapped.new_ones((mapped.shape[0],)),
    )
    # If there is no root, no shrink is needed. This avoids shrinking every
    # perfectly eligible input by the safety fraction.
    any_root = valid.any(dim=(1, 2, 3))
    scale = torch.where(any_root, scale, torch.ones_like(scale))
    line = torch.linspace(0, 1, side, device=mapped.device, dtype=mapped.dtype)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)
    return identity + scale[:, None, None, None] * (mapped - identity)
