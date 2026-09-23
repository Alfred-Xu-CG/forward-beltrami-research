"""Independent topology and gradient checks for prefix-sum fiber maps."""

import math

import torch

from qcopt.neural_bijection.dense import (
    MonotoneFiberP1Layer, MultiscaleMonotoneFiberP1Layer,
)
from qcopt.neural_bijection.dense.convex_quad import certify_convex_quad_output


def test_each_axis_is_boundary_fixed_and_has_positive_original_faces() -> None:
    generator = torch.Generator().manual_seed(4021)
    for axis in ("horizontal", "vertical"):
        side = 33
        shape = (2, side - 2, side - 1) if axis == "horizontal" else (
            2, side - 1, side - 2
        )
        logits = 1000 * torch.randn(shape, generator=generator)
        mapped = MonotoneFiberP1Layer(side, axis=axis)(logits)
        assert torch.isfinite(mapped).all()
        assert certify_convex_quad_output(mapped) > 0.049
        unit = torch.arange(side, dtype=mapped.dtype) / (side - 1)
        yy, xx = torch.meshgrid(unit, unit, indexing="ij")
        identity = torch.stack((xx, yy), dim=-1)
        assert torch.equal(mapped[:, 0], identity[None, 0].expand(2, -1, -1))
        assert torch.equal(mapped[:, -1], identity[None, -1].expand(2, -1, -1))
        assert torch.equal(mapped[:, :, 0], identity[None, :, 0].expand(2, -1, -1))
        assert torch.equal(mapped[:, :, -1], identity[None, :, -1].expand(2, -1, -1))


def test_strong_shear_has_finite_inverse_latent() -> None:
    side = 17
    axis = torch.arange(side, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    target = torch.stack((
        xx + 0.28 * torch.sin(math.pi * xx).square() * torch.sin(2 * math.pi * yy),
        yy,
    ), dim=-1)[None]
    floor_fraction, logit_span = 0.05, 8.0
    edge = target[:, 1:-1, 1:, 0] - target[:, 1:-1, :-1, 0]
    weights = (edge - floor_fraction / (side - 1)) / (1 - floor_fraction)
    assert weights.amin() > 0
    bounded = weights.log()
    bounded = bounded - bounded.mean(dim=-1, keepdim=True)
    assert (bounded / logit_span).abs().amax() < 1
    latent = torch.atanh(bounded / logit_span)
    output = MonotoneFiberP1Layer(
        side, floor_fraction=floor_fraction, logit_span=logit_span,
    )(latent)
    assert (output - target).abs().max() < 1e-12
    assert certify_convex_quad_output(output) > 0.11


def test_fiber_vjp_matches_finite_difference() -> None:
    generator = torch.Generator().manual_seed(483)
    for axis in ("horizontal", "vertical"):
        side = 9
        shape = (1, side - 2, side - 1) if axis == "horizontal" else (
            1, side - 1, side - 2
        )
        latent = 0.1 * torch.randn(shape, generator=generator,
                                   dtype=torch.float64)
        latent.requires_grad_(True)
        layer = MonotoneFiberP1Layer(side, axis=axis)
        cotangent = torch.randn((1, side, side, 2), generator=generator,
                                 dtype=torch.float64)
        exact = torch.autograd.grad((layer(latent) * cotangent).sum(), latent)[0]
        index = (0, 2, 3)
        step = 1e-6
        with torch.no_grad():
            plus, minus = latent.clone(), latent.clone()
            plus[index] += step
            minus[index] -= step
            numerical = ((layer(plus) * cotangent).sum()
                         - (layer(minus) * cotangent).sum()) / (2 * step)
        assert torch.allclose(exact[index], numerical, rtol=1e-6, atol=1e-8)


def test_multiscale_log_density_has_safe_nontrivial_fine_residual() -> None:
    generator = torch.Generator().manual_seed(615)
    coarse = torch.randn((2, 7, 8), generator=generator,
                          dtype=torch.float64, requires_grad=True)
    fine = (0.1 * torch.randn((2, 31, 32), generator=generator,
                                dtype=torch.float64)).requires_grad_()
    layer = MultiscaleMonotoneFiberP1Layer(33)
    baseline = layer((coarse, torch.zeros_like(fine)))
    output = layer((coarse, fine))
    assert (output - baseline).abs().amax() > 1e-5
    assert certify_convex_quad_output(output) > 0.049
    loss = output.square().sum()
    derivatives = torch.autograd.grad(loss, (coarse, fine))
    assert all(torch.isfinite(g).all() and g.abs().amax() > 0 for g in derivatives)
