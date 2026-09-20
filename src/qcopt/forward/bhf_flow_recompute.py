"""Memory-bounded multi-step BHF flow in a real-pair representation.

The custom autograd function stores only the initial state and static mesh
data. Backward reconstructs the trajectory and differentiates the replayed
flow, so the layer does not retain one dense BHF interaction graph per step.
This is checkpoint/recompute differentiation, not an implicit adjoint theorem.
"""

from __future__ import annotations

import torch

from .bhf_torch import bhf_near_far_apply_torch_differentiable_real_pair


def _face_fz_pair(source: torch.Tensor, image: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    source_tri = source[faces]
    image_tri = image[faces]
    dx1 = source_tri[:, 1, 0] - source_tri[:, 0, 0]
    dy1 = source_tri[:, 1, 1] - source_tri[:, 0, 1]
    dx2 = source_tri[:, 2, 0] - source_tri[:, 0, 0]
    dy2 = source_tri[:, 2, 1] - source_tri[:, 0, 1]
    determinant = dx1 * dy2 - dx2 * dy1
    du1 = image_tri[:, 1, 0] - image_tri[:, 0, 0]
    du2 = image_tri[:, 2, 0] - image_tri[:, 0, 0]
    dv1 = image_tri[:, 1, 1] - image_tri[:, 0, 1]
    dv2 = image_tri[:, 2, 1] - image_tri[:, 0, 1]
    ux = (du1 * dy2 - du2 * dy1) / determinant
    uy = (-du1 * dx2 + du2 * dx1) / determinant
    vx = (dv1 * dy2 - dv2 * dy1) / determinant
    vy = (-dv1 * dx2 + dv2 * dx1) / determinant
    return torch.stack((0.5 * (ux + vy), 0.5 * (vx - uy)), dim=1)


def _safe_step_active_torch(
    current: torch.Tensor,
    delta: torch.Tensor,
    faces: torch.Tensor,
    requested: float,
    min_det_margin: float,
    safety: float,
) -> tuple[float, int]:
    """Return a conservative first determinant-root step and active face."""

    with torch.no_grad():
        base = current[faces]
        direction = delta[faces]
        a0, a1, a2 = base[:, 0], base[:, 1], base[:, 2]
        d0, d1, d2 = direction[:, 0], direction[:, 1], direction[:, 2]
        e1, e2 = a1 - a0, a2 - a0
        de1, de2 = d1 - d0, d2 - d0
        cross = lambda left, right: left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
        c = cross(e1, e2)
        if bool(torch.any(c <= min_det_margin)):
            raise ValueError("current map is at or below the determinant margin")
        a = cross(de1, de2)
        b = cross(e1, de2) + cross(de1, e2)
        constant = c - min_det_margin
        inf = torch.full_like(c, float("inf"))
        linear = torch.where((torch.abs(a) <= 1e-12) & (b < 0.0), -constant / b, inf)
        discriminant = b * b - 4.0 * a * constant
        valid = (torch.abs(a) > 1e-12) & (discriminant >= 0.0)
        root = torch.sqrt(torch.clamp(discriminant, min=0.0))
        first = torch.where(valid, (-b - root) / (2.0 * a), inf)
        second = torch.where(valid, (-b + root) / (2.0 * a), inf)
        first = torch.where(first > 0.0, first, inf)
        second = torch.where(second > 0.0, second, inf)
        bound = torch.minimum(linear, torch.minimum(first, second))
        value_tensor, active_tensor = torch.min(bound, dim=0)
        value = float(value_tensor.item())
        active = int(active_tensor.item())
        if not torch.isfinite(torch.as_tensor(value)):
            return float(requested), -1
        return float(min(requested, safety * value)), active


def _maximum_safe_step_torch(
    current: torch.Tensor,
    delta: torch.Tensor,
    faces: torch.Tensor,
    requested: float,
    min_det_margin: float,
    safety: float,
) -> float:
    """Return a conservative first determinant-root step over all faces."""

    return _safe_step_active_torch(
        current, delta, faces, requested, min_det_margin, safety
    )[0]


def _smooth_min_conservative_torch(
    values: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """Differentiate a conservative smooth lower envelope.

    The pairwise map ``(a+b-sqrt((a-b)^2+eps^2))/2`` is no larger than
    ``min(a,b)``.  A reduction therefore remains a lower bound on every input
    while avoiding the hard active-face selection used by the exact root step.
    This is intentionally conservative: the small amount of extra shrinkage
    buys a globally smooth active-set surrogate.
    """

    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty one-dimensional tensor")
    if smoothing <= 0.0 or not torch.isfinite(torch.as_tensor(smoothing)):
        raise ValueError("smoothing must be positive and finite")
    current = values
    eps = torch.as_tensor(float(smoothing), dtype=values.dtype, device=values.device)
    while current.numel() > 1:
        if current.numel() % 2:
            current = torch.cat((current, current[-1:]))
        left = current[0::2]
        right = current[1::2]
        difference = left - right
        current = 0.5 * (left + right - torch.sqrt(difference * difference + eps * eps))
    return current[0]


def _smooth_safe_step_torch(
    current: torch.Tensor,
    delta: torch.Tensor,
    faces: torch.Tensor,
    requested: float,
    min_det_margin: float,
    safety: float,
    smoothing: float = 1e-7,
) -> torch.Tensor:
    """Return a smooth, conservative determinant-safe step.

    For one face, ``det(A+tB)=c+bt+at^2``.  Instead of selecting the first
    exact root, this routine solves the conservative inequality
    ``|a|t^2+|b|t <= c-margin``.  The resulting bound is valid for every face.
    A pairwise smooth lower envelope is then taken over all faces and the
    requested step.  The output is therefore differentiable with respect to
    ``current``/``delta`` and cannot exceed the certified determinant bound.
    """

    if current.ndim != 2 or current.shape[1] != 2 or delta.shape != current.shape:
        raise ValueError("current and delta must have shape (n_vertices,2)")
    if requested <= 0.0 or min_det_margin <= 0.0 or not 0.0 < safety < 1.0:
        raise ValueError("requested, min_det_margin, and safety are invalid")
    if smoothing <= 0.0 or not torch.isfinite(torch.as_tensor(smoothing)):
        raise ValueError("smoothing must be positive and finite")
    base = current[faces]
    direction = delta[faces]
    e1, e2 = base[:, 1] - base[:, 0], base[:, 2] - base[:, 0]
    de1, de2 = direction[:, 1] - direction[:, 0], direction[:, 2] - direction[:, 0]
    cross = lambda left, right: left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
    c = cross(e1, e2)
    if bool(torch.any(c <= min_det_margin)):
        raise ValueError("current map is at or below the determinant margin")
    a = cross(de1, de2)
    b = cross(e1, de2) + cross(de1, e2)
    constant = c - min_det_margin
    # Smooth absolute values are upper bounds on |a| and |b|, preserving the
    # safety proof while keeping the expression differentiable at zero.
    abs_eps = torch.as_tensor(1e-12, dtype=current.dtype, device=current.device)
    abs_a = torch.sqrt(a * a + abs_eps * abs_eps)
    abs_b = torch.sqrt(b * b + abs_eps * abs_eps)
    discriminant = abs_b * abs_b + 4.0 * abs_a * constant
    roots = 2.0 * constant / (abs_b + torch.sqrt(discriminant))
    requested_tensor = torch.as_tensor(float(requested), dtype=current.dtype, device=current.device)
    values = torch.cat((requested_tensor.reshape(1), safety * roots))
    return _smooth_min_conservative_torch(values, smoothing)


def _fixed_active_safe_step_torch(
    current: torch.Tensor,
    delta: torch.Tensor,
    faces: torch.Tensor,
    requested: float,
    min_det_margin: float,
    safety: float,
    active: int,
) -> torch.Tensor:
    """Differentiate the selected determinant root with the active face fixed."""

    if active < 0:
        return torch.as_tensor(requested, dtype=current.dtype, device=current.device)
    base = current[faces[active]]
    direction = delta[faces[active]]
    e1, e2 = base[1] - base[0], base[2] - base[0]
    de1, de2 = direction[1] - direction[0], direction[2] - direction[0]
    cross = lambda left, right: left[0] * right[1] - left[1] * right[0]
    c = cross(e1, e2)
    a = cross(de1, de2)
    b = cross(e1, de2) + cross(de1, e2)
    constant = c - min_det_margin
    discriminant = torch.clamp(b * b - 4.0 * a * constant, min=0.0)
    root = torch.sqrt(discriminant)
    first = (-b - root) / (2.0 * a)
    second = (-b + root) / (2.0 * a)
    # The active branch was selected in forward. These masks only choose the
    # positive algebraic root and do not differentiate the active-set choice.
    positive_first = first > 0.0
    positive_second = second > 0.0
    root_value = torch.where(positive_first, first, second)
    root_value = torch.where(positive_first & positive_second, torch.minimum(first, second), root_value)
    root_value = torch.where(torch.abs(a) <= 1e-12, -constant / b, root_value)
    return torch.minimum(
        torch.as_tensor(requested, dtype=current.dtype, device=current.device),
        safety * root_value,
    )


def _flow_forward(
    initial: torch.Tensor,
    source: torch.Tensor,
    faces: torch.Tensor,
    variation: torch.Tensor,
    step_size: float,
    steps: int,
    near_order: int,
    target_block_size: int,
    pair_block_size: int,
    adaptive_safe: bool = False,
    min_det_margin: float = 1e-8,
    safety: float = 0.95,
    step_history: list[float] | None = None,
    differentiate_safe_step: bool = False,
    smooth_safe: bool = False,
    smoothing: float = 1e-7,
) -> torch.Tensor:
    current = initial
    for _ in range(steps):
        fz = _face_fz_pair(source, current, faces)
        velocity = bhf_near_far_apply_torch_differentiable_real_pair(
            source,
            current,
            faces,
            fz,
            variation,
            near_order=near_order,
            target_block_size=target_block_size,
            pair_block_size=pair_block_size,
        )
        effective_step = step_size
        if adaptive_safe:
            if smooth_safe:
                effective_step = _smooth_safe_step_torch(
                    current, velocity, faces, step_size,
                    min_det_margin, safety, smoothing,
                )
            else:
                selected_step, active = _safe_step_active_torch(
                    current, velocity, faces, step_size, min_det_margin, safety
                )
                if differentiate_safe_step and torch.is_grad_enabled() and active >= 0:
                    effective_step = _fixed_active_safe_step_torch(
                        current, velocity, faces, step_size,
                        min_det_margin, safety, active,
                    )
                else:
                    effective_step = selected_step
        if step_history is not None:
            step_history.append(float(effective_step))
        current = current + effective_step * velocity
    return current


class _RecomputeBHF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        initial: torch.Tensor,
        source: torch.Tensor,
        faces: torch.Tensor,
        variation: torch.Tensor,
        step_size: float,
        steps: int,
        near_order: int,
        target_block_size: int,
        pair_block_size: int,
        adaptive_safe: bool,
        min_det_margin: float,
        safety: float,
        smooth_safe: bool,
        smoothing: float,
    ) -> torch.Tensor:
        if initial.ndim != 2 or initial.shape[1] != 2:
            raise ValueError("initial must have shape (n_vertices,2)")
        if source.shape != initial.shape or source.shape[1] != 2:
            raise ValueError("source must match initial and have shape (n_vertices,2)")
        if variation.ndim != 2 or variation.shape[1] != 2:
            raise ValueError("variation must have shape (n_faces,2)")
        if steps < 1 or step_size <= 0.0:
            raise ValueError("steps must be positive and step_size must be positive")
        ctx.save_for_backward(initial, source, faces, variation)
        ctx.step_size = float(step_size)
        ctx.steps = int(steps)
        ctx.near_order = int(near_order)
        ctx.target_block_size = int(target_block_size)
        ctx.pair_block_size = int(pair_block_size)
        ctx.adaptive_safe = bool(adaptive_safe)
        ctx.min_det_margin = float(min_det_margin)
        ctx.safety = float(safety)
        ctx.smooth_safe = bool(smooth_safe)
        ctx.smoothing = float(smoothing)
        with torch.no_grad():
            return _flow_forward(
                initial,
                source,
                faces,
                variation,
                ctx.step_size,
                ctx.steps,
                ctx.near_order,
                ctx.target_block_size,
                ctx.pair_block_size,
                ctx.adaptive_safe,
                ctx.min_det_margin,
                ctx.safety,
                None,
                False,
                ctx.smooth_safe,
                ctx.smoothing,
            )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        initial, source, faces, variation = ctx.saved_tensors
        replay_initial = initial.detach().requires_grad_(True)
        replay_variation = variation.detach().requires_grad_(ctx.needs_input_grad[3])
        with torch.enable_grad():
            replay = _flow_forward(
                replay_initial,
                source,
                faces,
                replay_variation,
                ctx.step_size,
                ctx.steps,
                ctx.near_order,
                ctx.target_block_size,
                ctx.pair_block_size,
                ctx.adaptive_safe,
                ctx.min_det_margin,
                ctx.safety,
                None,
                True,
                ctx.smooth_safe,
                ctx.smoothing,
            )
            # ``backward`` (rather than autograd.grad) is intentional: legacy
            # Torch checkpointing supports this path even when its explicit
            # grad API is unavailable.
            replay.backward(grad_output)
        grad_initial = replay_initial.grad
        grad_variation = replay_variation.grad if ctx.needs_input_grad[3] else None
        return grad_initial, None, None, grad_variation, None, None, None, None, None, None, None, None, None, None


def recompute_bhf_flow_real_pair(
    initial: torch.Tensor,
    source: torch.Tensor,
    faces: torch.Tensor,
    variation: torch.Tensor,
    *,
    step_size: float = 0.05,
    steps: int = 2,
    near_order: int = 8,
    target_block_size: int = 128,
    pair_block_size: int = 2048,
    adaptive_safe: bool = False,
    min_det_margin: float = 1e-8,
    safety: float = 0.95,
    smooth_safe: bool = False,
    smoothing: float = 1e-7,
) -> torch.Tensor:
    """Run a multi-step BHF flow while replaying its trajectory in backward."""

    return _RecomputeBHF.apply(
        initial,
        source,
        faces,
        variation,
        float(step_size),
        int(steps),
        int(near_order),
        int(target_block_size),
        int(pair_block_size),
        bool(adaptive_safe),
        float(min_det_margin),
        float(safety),
        bool(smooth_safe),
        float(smoothing),
    )


__all__ = ["recompute_bhf_flow_real_pair"]
