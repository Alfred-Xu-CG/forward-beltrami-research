import numpy as np

from qcopt.forward.beurling import periodic_beurling_apply
from qcopt.forward.beurling_particle_mesh import particle_mesh_beurling_apply


def test_particle_mesh_is_exact_on_a_matching_uniform_fourier_grid():
    nx, ny = 32, 24
    x = np.arange(nx, dtype=np.float64) / nx
    y = np.arange(ny, dtype=np.float64) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = (xx + 1j * yy).ravel()
    values = np.exp(2j * np.pi * (3.0 * xx - 2.0 * yy)).ravel()
    weights = np.full(points.shape, 1.0 / (nx * ny))
    result = particle_mesh_beurling_apply(points, values, weights, grid_shape=(ny, nx))
    expected = periodic_beurling_apply(values.reshape(ny, nx)).ravel()
    assert np.max(np.abs(result - expected)) < 1e-12


def test_particle_mesh_has_convergent_error_for_scattered_fourier_samples():
    n = 128
    xx, yy = np.meshgrid(
        (np.arange(n, dtype=np.float64) + 0.23) / n,
        (np.arange(n, dtype=np.float64) + 0.37) / n,
        indexing="xy",
    )
    points = (xx + 1j * yy).ravel()
    kx, ky = 3, -2
    values = np.exp(2j * np.pi * (kx * points.real + ky * points.imag))
    weights = np.full(len(points), 1.0 / len(points))
    expected = ((kx - 1j * ky) / (kx + 1j * ky)) * values
    coarse = particle_mesh_beurling_apply(points, values, weights, grid_shape=(32, 32))
    fine = particle_mesh_beurling_apply(points, values, weights, grid_shape=(128, 128))
    assert np.mean(np.abs(fine - expected)) < np.mean(np.abs(coarse - expected))


def test_cic_window_deconvolution_improves_low_frequency_scattered_mode():
    n = 128
    xx, yy = np.meshgrid(
        (np.arange(n, dtype=np.float64) + 0.23) / n,
        (np.arange(n, dtype=np.float64) + 0.37) / n,
        indexing="xy",
    )
    points = (xx + 1j * yy).ravel()
    kx, ky = 3, -2
    values = np.exp(2j * np.pi * (kx * points.real + ky * points.imag))
    weights = np.full(len(points), 1.0 / len(points))
    expected = ((kx - 1j * ky) / (kx + 1j * ky)) * values
    plain = particle_mesh_beurling_apply(points, values, weights, grid_shape=(64, 64))
    corrected = particle_mesh_beurling_apply(
        points, values, weights, grid_shape=(64, 64), deconvolve=True
    )
    assert np.mean(np.abs(corrected - expected)) < np.mean(np.abs(plain - expected))


def test_cubic_bspline_deconvolution_improves_over_cic_at_same_grid():
    n = 128
    xx, yy = np.meshgrid(
        (np.arange(n, dtype=np.float64) + 0.23) / n,
        (np.arange(n, dtype=np.float64) + 0.37) / n,
        indexing="xy",
    )
    points = (xx + 1j * yy).ravel()
    kx, ky = 3, -2
    values = np.exp(2j * np.pi * (kx * points.real + ky * points.imag))
    weights = np.full(len(points), 1.0 / len(points))
    expected = ((kx - 1j * ky) / (kx + 1j * ky)) * values
    cubic = particle_mesh_beurling_apply(
        points, values, weights, grid_shape=(64, 64), kernel="cubic", deconvolve=True
    )
    cic = particle_mesh_beurling_apply(
        points, values, weights, grid_shape=(64, 64), kernel="cic", deconvolve=True
    )
    assert np.mean(np.abs(cubic - expected)) < np.mean(np.abs(cic - expected))


def test_particle_mesh_supports_disjoint_target_points():
    rng = np.random.default_rng(44)
    source = 0.2 + 0.25 * rng.random(128) + 1j * (0.2 + 0.6 * rng.random(128))
    target = 0.55 + 0.25 * rng.random(64) + 1j * (0.2 + 0.6 * rng.random(64))
    values = np.exp(-((source.real - 0.3) ** 2 + (source.imag - 0.5) ** 2) / 0.03)
    weights = np.full(len(source), 0.15 / len(source))
    output = particle_mesh_beurling_apply(
        source,
        values,
        weights,
        grid_shape=(64, 64),
        kernel="cubic",
        deconvolve=True,
        target_points=target,
    )
    assert output.shape == target.shape
    assert np.all(np.isfinite(output))
