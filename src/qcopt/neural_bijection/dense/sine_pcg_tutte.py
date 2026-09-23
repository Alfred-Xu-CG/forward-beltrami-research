"""Bounded positive-conductance Tutte layer with Dirichlet-sine PCG.

The exact weighted equilibrium is a P1 homeomorphism. A returned finite-
precision iterate is separately face-checked; it is never labelled an exact
solve merely because the conductances are positive.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .sine_spectral import dst2, idst2


def _conductances(logits: tuple[torch.Tensor, ...], minimum: float, maximum: float) -> tuple[torch.Tensor, ...]:
    return tuple(minimum + (maximum - minimum) * value.sigmoid() for value in logits)


def _laplacian(
    full: torch.Tensor, horizontal: torch.Tensor, vertical: torch.Tensor, diagonal: torch.Tensor,
) -> torch.Tensor:
    """Symmetric full-graph Laplacian on a NE-diagonal triangular grid."""
    result = torch.zeros_like(full)
    delta = horizontal[..., None] * (full[:, :, :-1] - full[:, :, 1:])
    result[:, :, :-1] += delta
    result[:, :, 1:] -= delta
    delta = vertical[..., None] * (full[:, :-1] - full[:, 1:])
    result[:, :-1] += delta
    result[:, 1:] -= delta
    delta = diagonal[..., None] * (full[:, :-1, :-1] - full[:, 1:, 1:])
    result[:, :-1, :-1] += delta
    result[:, 1:, 1:] -= delta
    return result


def _interior_action(
    vector: torch.Tensor, horizontal: torch.Tensor, vertical: torch.Tensor, diagonal: torch.Tensor,
) -> torch.Tensor:
    full = F.pad(vector.permute(0, 3, 1, 2), (1, 1, 1, 1)).permute(0, 2, 3, 1)
    return _laplacian(full, horizontal, vertical, diagonal)[:, 1:-1, 1:-1]


def _sine_inverse(rhs: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    coefficients = dst2(rhs.permute(0, 3, 1, 2))
    return idst2(coefficients / eigenvalues).permute(0, 2, 3, 1)


def _pcg(
    rhs: torch.Tensor,
    conductances: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eigenvalues: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, int, float]:
    """Solve the two coordinate systems per batch to a measured true residual."""
    horizontal, vertical, diagonal = conductances
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = torch.linalg.vector_norm(rhs, dim=(1, 2, 3))
    if bool(torch.all(rhs_norm == 0)):
        return solution, 0, 0.0
    zero_rhs = rhs_norm == 0
    preconditioned = _sine_inverse(residual, eigenvalues)
    direction = preconditioned.clone()
    old_pairing = (residual * preconditioned).sum(dim=(1, 2, 3))
    scale = rhs_norm.clamp_min(torch.finfo(rhs.dtype).tiny)
    measured = float("inf")
    for iteration in range(1, max_iterations + 1):
        action = _interior_action(direction, horizontal, vertical, diagonal)
        denominator = (direction * action).sum(dim=(1, 2, 3))
        if bool(torch.any((denominator <= 0) & ~zero_rhs)):
            raise RuntimeError("positive-conductance PCG lost positive definiteness")
        alpha = torch.where(zero_rhs, torch.zeros_like(old_pairing), old_pairing / denominator.clamp_min(torch.finfo(rhs.dtype).tiny))
        solution = solution + alpha[:, None, None, None] * direction
        residual = residual - alpha[:, None, None, None] * action
        if iteration % 5 == 0 or iteration == max_iterations:
            true_residual = rhs - _interior_action(solution, horizontal, vertical, diagonal)
            measured = float((torch.linalg.vector_norm(true_residual, dim=(1, 2, 3)) / scale).max())
            if measured <= tolerance:
                return solution, iteration, measured
        preconditioned = _sine_inverse(residual, eigenvalues)
        new_pairing = (residual * preconditioned).sum(dim=(1, 2, 3))
        beta = torch.where(zero_rhs, torch.zeros_like(new_pairing), new_pairing / old_pairing.clamp_min(torch.finfo(rhs.dtype).tiny))
        direction = preconditioned + beta[:, None, None, None] * direction
        old_pairing = new_pairing
    raise RuntimeError(f"sine-preconditioned CG did not converge: residual={measured:.3e}")


def _minimum_signed_area_ratio(mapped: torch.Tensor) -> float:
    side = mapped.shape[1]
    a = mapped[:, :-1, :-1]
    b = mapped[:, :-1, 1:]
    c = mapped[:, 1:, 1:]
    d = mapped[:, 1:, :-1]
    def cross(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    return float(torch.minimum(cross(b - a, c - a).amin(), cross(c - a, d - a).amin()) * (side - 1) ** 2)


class _SinePCGEquilibrium(torch.autograd.Function):
    @staticmethod
    def forward(ctx, horizontal_logits, vertical_logits, diagonal_logits, layer):
        with torch.no_grad():
            logits = (horizontal_logits, vertical_logits, diagonal_logits)
            conductances = _conductances(logits, layer.minimum_conductance, layer.maximum_conductance)
            source = layer._source[None].expand(horizontal_logits.shape[0], -1, -1, -1)
            rhs = -_laplacian(source, *conductances)[:, 1:-1, 1:-1]
            displacement, iterations, residual = _pcg(
                rhs, conductances, layer._eigenvalues,
                tolerance=layer.tolerance, max_iterations=layer.max_iterations,
            )
            padded = F.pad(displacement.permute(0, 3, 1, 2), (1, 1, 1, 1)).permute(0, 2, 3, 1)
            mapped = source + padded
            area = _minimum_signed_area_ratio(mapped)
            if not torch.isfinite(mapped).all() or area <= 0:
                raise RuntimeError("finite-precision PCG iterate is not an oriented P1 homeomorphism")
            layer.last_forward_stats = {"iterations": iterations, "true_relative_residual": residual,
                                        "minimum_signed_area_ratio": area}
            ctx.save_for_backward(*logits, *conductances, mapped)
            ctx.layer = layer
        return mapped

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        horizontal_logits, vertical_logits, diagonal_logits, horizontal, vertical, diagonal, mapped = ctx.saved_tensors
        layer = ctx.layer
        with torch.no_grad():
            adjoint_interior, iterations, residual = _pcg(
                grad_output[:, 1:-1, 1:-1].contiguous(),
                (horizontal, vertical, diagonal), layer._eigenvalues,
                tolerance=layer.tolerance, max_iterations=layer.max_iterations,
            )
            adjoint = F.pad(adjoint_interior.permute(0, 3, 1, 2), (1, 1, 1, 1)).permute(0, 2, 3, 1)
            def edge_gradient(left: torch.Tensor, right: torch.Tensor,
                              adjoint_left: torch.Tensor, adjoint_right: torch.Tensor,
                              logits: torch.Tensor) -> torch.Tensor:
                weight_gradient = -((left - right) * (adjoint_left - adjoint_right)).sum(dim=-1)
                sigmoid = logits.sigmoid()
                return weight_gradient * (layer.maximum_conductance - layer.minimum_conductance) * sigmoid * (1 - sigmoid)
            gradients = (
                edge_gradient(mapped[:, :, :-1], mapped[:, :, 1:], adjoint[:, :, :-1], adjoint[:, :, 1:], horizontal_logits),
                edge_gradient(mapped[:, :-1], mapped[:, 1:], adjoint[:, :-1], adjoint[:, 1:], vertical_logits),
                edge_gradient(mapped[:, :-1, :-1], mapped[:, 1:, 1:], adjoint[:, :-1, :-1], adjoint[:, 1:, 1:], diagonal_logits),
            )
            layer.last_backward_stats = {"iterations": iterations, "true_relative_residual": residual}
        return *gradients, None


class SinePreconditionedTutteLayer(torch.nn.Module):
    """All-edge positive-weight P1 equilibrium, float64 CPU/CUDA, implicit VJP.

    Input is three batched logit tensors for horizontal (N,N-1), vertical
    (N-1,N), and northeast diagonal (N-1,N-1) edges. Boundary-boundary edges
    are present in these arrays but have identically zero influence/gradient.
    """

    def __init__(self, side: int, *, minimum_conductance: float = 1.0,
                 maximum_conductance: float = 4.0, tolerance: float = 1e-10,
                 max_iterations: int = 100) -> None:
        super().__init__()
        if side < 3 or not 0 < minimum_conductance < maximum_conductance:
            raise ValueError("side and strictly positive conductance bounds required")
        if not 0 < tolerance < 1 or max_iterations < 1:
            raise ValueError("invalid PCG tolerance or iteration budget")
        line = torch.linspace(0, 1, side, dtype=torch.float64)
        yy, xx = torch.meshgrid(line, line, indexing="ij")
        self.register_buffer("_source", torch.stack((xx, yy), dim=-1), persistent=False)
        angle = math.pi * torch.arange(1, side - 1, dtype=torch.float64) / (side - 1)
        eigenvalues = 4 - 2 * angle.cos()[:, None] - 2 * angle.cos()[None, :]
        self.register_buffer("_eigenvalues", eigenvalues, persistent=False)
        self.side = side
        self.minimum_conductance = float(minimum_conductance)
        self.maximum_conductance = float(maximum_conductance)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.last_forward_stats: dict[str, float | int] = {}
        self.last_backward_stats: dict[str, float | int] = {}

    def forward(self, horizontal_logits: torch.Tensor, vertical_logits: torch.Tensor,
                diagonal_logits: torch.Tensor) -> torch.Tensor:
        side = self.side
        batch = horizontal_logits.shape[0]
        expected = ((batch, side, side - 1), (batch, side - 1, side), (batch, side - 1, side - 1))
        logits = (horizontal_logits, vertical_logits, diagonal_logits)
        if any(value.shape != shape or value.dtype != torch.float64 or value.device != self._source.device
               for value, shape in zip(logits, expected)):
            raise ValueError("batched edge logits must match layer side/device and use float64")
        if not all(bool(torch.isfinite(value).all()) for value in logits):
            raise ValueError("edge logits must be finite")
        return _SinePCGEquilibrium.apply(*logits, self)
