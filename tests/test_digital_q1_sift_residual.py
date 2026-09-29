"""Image-derived sparse correspondence coordinates use the Q1 image frame."""

from __future__ import annotations

import torch
import numpy as np
import pytest

from qcopt.neural_bijection.dense.q1_image_sampling import warp_moving_at_q1_map
from tools.digital_q1_sift_residual import warp_moving_to_fixed
from tools.digital_q1_superglue_residual import select_ransac_matches
from tools.digital_q1_match_fit import fit_correspondences, q1_map_at_points
from tools.digital_q1_forward_kernel import (
    _gaussian_displacement, _gaussian_displacement_at_points,
    _gaussian_displacement_direct, forward_kernel_map,
)


def test_affine_warp_matches_factorized_q1_sampling_at_pixel_centers() -> None:
    torch.manual_seed(179)
    moving = torch.rand((1, 1, 31, 27), dtype=torch.float32)
    matrix = torch.tensor([[.96, .07], [-.03, 1.04]], dtype=torch.float32)
    offset = torch.tensor([.02, -.015], dtype=torch.float32)
    side = 17
    axis = torch.arange(side, dtype=torch.float32) / (side - 1)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    residual = torch.stack((x, y), dim=-1)[None]
    effective = residual @ matrix.T + offset
    expected = warp_moving_at_q1_map(moving, effective, height=31, width=27)
    actual = warp_moving_to_fixed(moving, matrix, offset, height=31, width=27)
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=0)


def test_superglue_match_filter_keeps_direction_and_half_pixel_centers() -> None:
    xx, yy = np.meshgrid(np.arange(8, 120, 24), np.arange(10, 125, 24))
    source = np.stack((xx.ravel(), yy.ravel()), axis=-1).astype(np.float32)
    target = source + np.array([3., -2.], dtype=np.float32)
    result = select_ransac_matches(source, target, np.arange(len(source)),
                                   width=128, height=128)
    assert result["status"] == "ok"
    assert result["ransac_inliers"] == len(source)
    assert result["occupied_quadrants_4x4"] > 2
    np.testing.assert_allclose(result["source_points_unit"], (source + .5) / 128)
    np.testing.assert_allclose(result["target_points_unit"], (target + .5) / 128)


def test_sparse_q1_queries_are_affine_exact_and_differentiable() -> None:
    axis = torch.linspace(0, 1, 17)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    vertices = torch.stack((1.03 * x + .08 * y + .02,
                            -.04 * x + .97 * y - .01), dim=-1)[None]
    vertices.requires_grad_(True)
    points = torch.tensor([[[.13, .24], [.51, .67], [.88, .91]]])
    values = q1_map_at_points(vertices, points)
    expected = torch.stack((1.03 * points[..., 0] + .08 * points[..., 1] + .02,
                            -.04 * points[..., 0] + .97 * points[..., 1] - .01), -1)
    torch.testing.assert_close(values, expected, atol=2e-7, rtol=0)
    values.square().sum().backward()
    assert vertices.grad is not None and bool(torch.isfinite(vertices.grad).all())


def test_sparse_fit_keeps_safe_dense_map_and_finite_latent_gradients() -> None:
    axis = torch.linspace(.15, .85, 5)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    source = torch.stack((x.ravel(), y.ravel()), dim=-1)
    bump = torch.sin(torch.pi * source[:, 0]) * torch.sin(torch.pi * source[:, 1])
    target = source + torch.stack((.012 * bump, -.008 * bump), dim=-1)
    mapped, report = fit_correspondences(
        source, target, final_side=33, steps=3, learning_rate=.04,
        device="cpu",
    )
    assert mapped.shape == (1, 33, 33, 2)
    assert report["finite_gradient_steps"] == 3
    assert report["residual_nonpositive_corners"] == 0
    assert report["residual_boundary_max_error"] == 0


def test_forward_kernel_decoder_is_safe_and_differentiable_in_matches() -> None:
    axis = torch.linspace(.12, .88, 6)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    source = torch.stack((x.ravel(), y.ravel()), dim=-1).requires_grad_(True)
    bump = torch.sin(torch.pi * source[:, 0]) * torch.sin(torch.pi * source[:, 1])
    target = source + torch.stack((.015 * bump, -.009 * bump), dim=-1)
    mapped = forward_kernel_map(source, target, final_side=65, sigma=.12)
    assert mapped.shape == (1, 65, 65, 2)
    from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 65),
                            torch.linspace(0, 1, 65), indexing="ij")
    reference = torch.stack((xx, yy), dim=-1)[None]
    validity = validate_q1_map(mapped, reference)
    assert validity["nonpositive_corners"] == 0
    assert validity["boundary_max_error"] == 0
    (mapped - reference).square().mean().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())


def test_multiband_forward_kernel_moves_new_fine_vertices_safely() -> None:
    source = torch.tensor([[.21, .21], [.49, .21], [.79, .21],
                           [.21, .49], [.49, .49], [.79, .49],
                           [.21, .79], [.49, .79], [.79, .79]],
                          dtype=torch.float64, requires_grad=True)
    target = source + torch.stack((.009 * source[:, 1],
                                   -.007 * source[:, 0]), dim=-1)
    coarse = forward_kernel_map(source, target, final_side=65,
                                update_sigmas=(.12, .06))
    fine = forward_kernel_map(source, target, final_side=65,
                              update_sigmas=(.12, .06, .03))
    from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 65, dtype=torch.float64),
                            torch.linspace(0, 1, 65, dtype=torch.float64),
                            indexing="ij")
    reference = torch.stack((xx, yy), dim=-1)[None]
    assert torch.max(torch.abs(fine - coarse)) > 1e-7
    validity = validate_q1_map(fine, reference)
    assert validity["nonpositive_corners"] == 0
    assert validity["boundary_max_error"] == 0
    (fine - reference).square().mean().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())


def test_multiband_forward_kernel_rejects_invalid_width_schedule() -> None:
    source = torch.tensor([[.2, .2]], dtype=torch.float64)
    target = torch.tensor([[.21, .2]], dtype=torch.float64)
    with pytest.raises(ValueError, match="numerically valid update widths"):
        forward_kernel_map(source, target, final_side=33, update_sigmas=())
    with pytest.raises(ValueError, match="numerically valid update widths"):
        forward_kernel_map(source, target, final_side=33,
                           update_sigmas=(.12, .06, .03))


def test_forward_kernel_rejects_float32_underflowing_width() -> None:
    source = torch.tensor([[.5, .5]], dtype=torch.float32)
    target = torch.tensor([[.51, .5]], dtype=torch.float32)
    with pytest.raises(ValueError, match="numerically valid update widths"):
        forward_kernel_map(source, target, final_side=17,
                           update_sigmas=(1e-30,))


def test_separable_gaussian_matches_direct_value_and_input_vjp() -> None:
    from tools.digital_q1_dhr_distill import identity_vertices
    grid = identity_vertices(17, device=torch.device("cpu")).double()
    source = torch.tensor([[.13, .19], [.73, .82], [.51, .29], [.34, .69]],
                          dtype=torch.float64)
    target = source + torch.tensor([[.02, -.01], [-.01, .01],
                                    [.005, .003], [-.007, -.004]],
                                   dtype=torch.float64)
    probe = torch.linspace(-1, 1, 17 * 17 * 2,
                           dtype=torch.float64).reshape(1, 17, 17, 2)
    direct_source = source.clone().requires_grad_(True)
    direct_target = target.clone().requires_grad_(True)
    separable_source = source.clone().requires_grad_(True)
    separable_target = target.clone().requires_grad_(True)
    direct = _gaussian_displacement_direct(
        grid, direct_source, direct_target, .13)
    separable = _gaussian_displacement(
        grid, separable_source, separable_target, .13)
    torch.testing.assert_close(separable, direct, atol=1e-14, rtol=1e-12)
    (direct * probe).sum().backward()
    (separable * probe).sum().backward()
    torch.testing.assert_close(separable_source.grad, direct_source.grad,
                               atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(separable_target.grad, direct_target.grad,
                               atol=1e-12, rtol=1e-10)


def test_residual_innovation_mode_preserves_identity_and_has_finite_vjp() -> None:
    source = torch.tensor([[.17, .23], [.41, .38], [.76, .62], [.53, .84]],
                          dtype=torch.float64)
    identity = forward_kernel_map(source, source, final_side=33,
                                  update_sigmas=(.12, .06),
                                  proposal_mode="residual")
    from tools.digital_q1_dhr_distill import identity_vertices
    reference = identity_vertices(33, device=torch.device("cpu")).double()
    torch.testing.assert_close(identity, reference, atol=1e-14, rtol=0)
    src = source.clone().requires_grad_(True)
    dst = source + torch.tensor([[.01, -.006], [.008, .003],
                                 [-.005, .004], [.006, -.007]],
                                dtype=torch.float64)
    output = forward_kernel_map(src, dst, final_side=33,
                                update_sigmas=(.12, .06),
                                proposal_mode="residual")
    from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
    check = validate_q1_map(output, reference)
    assert check["nonpositive_corners"] == 0
    assert check["boundary_max_error"] == 0
    (output - reference).square().mean().backward()
    assert src.grad is not None and bool(torch.isfinite(src.grad).all())


def test_calibrated_gain_matches_scalar_least_squares_and_keeps_safe_map() -> None:
    source = torch.tensor([[.17, .23], [.41, .38], [.76, .62], [.53, .84]],
                          dtype=torch.float64, requires_grad=True)
    target = source + torch.tensor([.01, -.004], dtype=torch.float64)
    calibration = torch.tensor([[.29, .31], [.62, .71]],
                               dtype=torch.float64, requires_grad=True)
    calibration_target = calibration + torch.tensor([.005, -.002],
                                                    dtype=torch.float64)
    field = _gaussian_displacement_at_points(calibration, source, target, .12)
    error = calibration_target - calibration
    expected = ((error * field).sum() / (field.square().sum() + 1e-8)).clamp(0, 1)
    gains: list[float] = []
    mapped = forward_kernel_map(
        source, target, final_side=17, update_sigmas=(.12,),
        proposal_mode="residual", gain_mode="calibrated",
        calibration_source=calibration,
        calibration_target=calibration_target, gain_trace=gains,
    )
    assert len(gains) == 1
    np.testing.assert_allclose(gains[0], float(expected), atol=1e-12)
    from tools.digital_q1_dhr_distill import identity_vertices
    from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
    reference = identity_vertices(17, device=torch.device("cpu")).double()
    check = validate_q1_map(mapped, reference)
    assert check["nonpositive_corners"] == 0
    assert check["boundary_max_error"] == 0
    (mapped - reference).square().mean().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())
    assert calibration.grad is not None and bool(torch.isfinite(calibration.grad).all())


def test_calibrated_gain_rejects_opposed_calibration_motion() -> None:
    source = torch.tensor([[.25, .25], [.75, .75]], dtype=torch.float64)
    target = source + torch.tensor([.01, 0.], dtype=torch.float64)
    calibration = torch.tensor([[.28, .3], [.72, .68]], dtype=torch.float64)
    gains: list[float] = []
    mapped = forward_kernel_map(
        source, target, final_side=17, update_sigmas=(.12,),
        proposal_mode="residual", gain_mode="calibrated",
        calibration_source=calibration,
        calibration_target=calibration + torch.tensor([-.01, 0.],
                                                      dtype=torch.float64),
        gain_trace=gains,
    )
    from tools.digital_q1_dhr_distill import identity_vertices
    assert gains == [0.]
    torch.testing.assert_close(mapped,
                               identity_vertices(17, device=torch.device("cpu")).double(),
                               atol=1e-14, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_forward_kernel_decoder_keeps_layer_indices_on_cuda() -> None:
    source = torch.tensor([[.2, .2], [.7, .2], [.2, .7], [.7, .7]],
                          device="cuda:0")
    target = source + torch.tensor([.01, -.005], device="cuda:0")
    mapped = forward_kernel_map(source, target, final_side=65, sigma=.12)
    assert mapped.is_cuda and bool(torch.isfinite(mapped).all())
