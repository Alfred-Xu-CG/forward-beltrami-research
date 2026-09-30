import numpy as np

from tools.digital_lung_all20_nonaffine import project_affine, rms_canvas_px


def test_affine_projection_exact_and_orthogonal() -> None:
    axis = np.linspace(0., 1., 5)
    x, y = np.meshgrid(axis, axis)
    affine = np.stack((1.1 * x + .2 * y + .03,
                       -.1 * x + .9 * y + .04), axis=-1)
    projected, residual = project_affine(affine)
    np.testing.assert_allclose(projected, affine, rtol=0., atol=1e-15)
    assert rms_canvas_px(residual) < 1e-12

    perturbation = np.stack((x * (1. - x) * y, np.zeros_like(x)), axis=-1)
    projected, residual = project_affine(affine + perturbation)
    assert rms_canvas_px(residual) > 1.
    np.testing.assert_allclose(np.sum((affine + perturbation) ** 2),
                               np.sum(projected ** 2) + np.sum(residual ** 2),
                               rtol=1e-13, atol=1e-13)
