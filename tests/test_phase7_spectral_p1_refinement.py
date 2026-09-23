"""An image-estimated scalar can drive fine-only safe P1 detail."""

import math

import torch

from qcopt.neural_bijection.dense import SineModeP1Refiner, exact_dyadic_p1_refine
from qcopt.neural_bijection.dense.convex_quad import certify_convex_quad_output


def test_fine_sine_latent_is_exact_and_differentiable_for_small_amplitude() -> None:
    side = 17
    axis = torch.arange(side, dtype=torch.float64) / (side - 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    bump = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    coarse = torch.stack((xx + 0.01 * bump, yy + 0.015 * bump), dim=-1)[None]
    baseline = exact_dyadic_p1_refine(exact_dyadic_p1_refine(coarse))
    fine_axis = torch.arange(65, dtype=torch.float64) / 64
    fine_yy, fine_xx = torch.meshgrid(fine_axis, fine_axis, indexing="ij")
    mode = torch.sin(16 * math.pi * fine_xx) * torch.sin(16 * math.pi * fine_yy)
    target = baseline + 0.0005 * mode[None, :, :, None]

    for mechanism in ("colored", "patch"):
        amplitude = torch.tensor([0.0005], dtype=torch.float64, requires_grad=True)
        decoder = SineModeP1Refiner(17, 65, cycles=8, mechanism=mechanism)
        output = decoder(coarse, amplitude)
        assert (output - target).abs().max() < 1e-12
        assert certify_convex_quad_output(output) > 0.05
        gradient = torch.autograd.grad((output * target).sum(), amplitude)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_large_spectral_latent_still_preserves_fixed_grid_faces() -> None:
    axis = torch.arange(17, dtype=torch.float64) / 16
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coarse = torch.stack((xx, yy), dim=-1)[None]
    for mechanism in ("colored", "patch"):
        output = SineModeP1Refiner(17, 33, cycles=8, mechanism=mechanism)(
            coarse, torch.tensor([10.0], dtype=torch.float64),
        )
        assert certify_convex_quad_output(output) >= 0.05 - 1e-12
        assert torch.equal(output[:, 0], exact_dyadic_p1_refine(coarse)[:, 0])


def test_three_independent_fine_modes_are_safe_and_differentiable() -> None:
    axis = torch.arange(17, dtype=torch.float64) / 16
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coarse = torch.stack((xx, yy), dim=-1)[None]
    fine_axis = torch.arange(65, dtype=torch.float64) / 64
    fy, fx = torch.meshgrid(fine_axis, fine_axis, indexing="ij")
    modes = ((8, 8), (8, 4), (4, 8))
    amplitudes = torch.tensor([[0.0003, -0.0002, 0.00025]], dtype=torch.float64,
                              requires_grad=True)
    target = exact_dyadic_p1_refine(exact_dyadic_p1_refine(coarse))
    for k, (kx, ky) in enumerate(modes):
        wave = torch.sin(2 * math.pi * kx * fx) * torch.sin(2 * math.pi * ky * fy)
        target = target + amplitudes[:, k, None, None, None] * wave[None, :, :, None]
    for mechanism in ("colored", "patch"):
        refiner = SineModeP1Refiner(17, 65, cycles=modes, mechanism=mechanism)
        output = refiner(coarse, amplitudes)
        assert (output - target).abs().max() < 1e-12
        assert certify_convex_quad_output(output) > 0.05
        gradient = torch.autograd.grad(output.square().sum(), amplitudes,
                                       retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().amin() > 0


def test_many_extreme_finite_amplitudes_cannot_overflow_mode_sum() -> None:
    axis = torch.arange(17, dtype=torch.float32) / 16
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coarse = torch.stack((xx, yy), dim=-1)[None]
    amplitudes = torch.tensor([[1e30, -1e30, 1e30]], dtype=torch.float32)
    modes = ((8, 8), (8, 4), (4, 8))
    for mechanism in ("colored", "patch"):
        output = SineModeP1Refiner(17, 33, cycles=modes,
                                   mechanism=mechanism)(coarse, amplitudes)
        assert torch.isfinite(output).all()
        assert certify_convex_quad_output(output) > 0


def test_three_mode_latent_vjp_matches_central_difference() -> None:
    axis = torch.arange(9, dtype=torch.float64) / 8
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coarse = torch.stack((xx, yy), dim=-1)[None]
    amplitudes = torch.tensor([[0.0002, -0.0001, 0.00015]], dtype=torch.float64,
                              requires_grad=True)
    modes = ((4, 4), (4, 2), (2, 4))
    weight = torch.Generator().manual_seed(143)
    cotangent = torch.randn((1, 33, 33, 2), generator=weight, dtype=torch.float64)
    for mechanism in ("colored", "patch"):
        for checkpoint_updates in (False, True):
            refiner = SineModeP1Refiner(
                9, 33, cycles=modes, mechanism=mechanism,
                checkpoint_updates=checkpoint_updates,
            )
            objective = (refiner(coarse, amplitudes) * cotangent).sum()
            exact = torch.autograd.grad(objective, amplitudes, retain_graph=True)[0]
            numerical = torch.zeros_like(exact)
            step = 1e-7
            with torch.no_grad():
                for k in range(3):
                    plus, minus = amplitudes.clone(), amplitudes.clone()
                    plus[0, k] += step
                    minus[0, k] -= step
                    lp = (refiner(coarse, plus) * cotangent).sum()
                    lm = (refiner(coarse, minus) * cotangent).sum()
                    numerical[0, k] = (lp - lm) / (2 * step)
            assert torch.allclose(exact, numerical, rtol=1e-6, atol=1e-7)


def test_four_compact_support_modes_are_independent_and_safe() -> None:
    axis = torch.arange(17, dtype=torch.float64) / 16
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coarse = torch.stack((xx, yy), dim=-1)[None]
    windows = tuple((i / 2, (i + 1) / 2, j / 2, (j + 1) / 2)
                    for j in range(2) for i in range(2))
    amplitude = torch.tensor([[0.0003, -0.0002, 0.00025, -0.0001]],
                             dtype=torch.float64, requires_grad=True)
    axis_fine = torch.arange(65, dtype=torch.float64) / 64
    fy, fx = torch.meshgrid(axis_fine, axis_fine, indexing="ij")
    wave = torch.sin(16 * math.pi * fx) * torch.sin(16 * math.pi * fy)
    target = exact_dyadic_p1_refine(exact_dyadic_p1_refine(coarse))
    for k, (xlo, xhi, ylo, yhi) in enumerate(windows):
        window_x = torch.where((fx >= xlo) & (fx <= xhi),
                               torch.sin(math.pi * (fx - xlo) / (xhi - xlo)).square(), 0)
        window_y = torch.where((fy >= ylo) & (fy <= yhi),
                               torch.sin(math.pi * (fy - ylo) / (yhi - ylo)).square(), 0)
        target = target + amplitude[:, k, None, None, None] * (
            wave * window_x * window_y
        )[None, :, :, None]
    for mechanism in ("colored", "patch"):
        refiner = SineModeP1Refiner(
            17, 65, cycles=((8, 8),) * 4, windows=windows,
            mechanism=mechanism, checkpoint_updates=True,
        )
        output = refiner(coarse, amplitude)
        assert (output - target).abs().max() < 1e-12
        assert certify_convex_quad_output(output) > 0
        derivative = torch.autograd.grad(output.square().sum(), amplitude,
                                         retain_graph=True)[0]
        assert torch.isfinite(derivative).all() and derivative.abs().amin() > 0
