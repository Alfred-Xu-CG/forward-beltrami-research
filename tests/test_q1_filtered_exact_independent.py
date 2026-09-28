"""Independent exact-arithmetic checks of the saved-binary Q1 certificate."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from qcopt.neural_bijection.dense.q1_filtered_sign import (
    _orientation_interval,
    certify_q1_binary_map,
)


def _rational(value: np.floating) -> Fraction:
    return Fraction(*float(value).as_integer_ratio())


def _sign_of_det(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> int:
    px, py, qx, qy, rx, ry = map(_rational, (*p, *q, *r))
    det = (qx - px) * (ry - py) - (qy - py) * (rx - px)
    return (det > 0) - (det < 0)


def _corner_signs(cell: np.ndarray) -> tuple[int, int, int, int]:
    a, b, c, d = cell[0, 0], cell[0, 1], cell[1, 1], cell[1, 0]
    return tuple(
        _sign_of_det(p, q, r)
        for p, q, r in ((a, b, d), (a, b, c), (d, b, c), (a, c, d))
    )


def _bound_contains(bound: float, value: Fraction, *, lower: bool) -> bool:
    if np.isnan(bound):
        return True  # Indeterminate interval forces the exact fallback.
    if np.isneginf(bound):
        return lower
    if np.isposinf(bound):
        return not lower
    rational_bound = Fraction(*float(bound).as_integer_ratio())
    return rational_bound <= value if lower else value <= rational_bound


def test_binary64_interval_bounds_against_fraction_oracle() -> None:
    tiny = np.nextafter(np.float64(0), np.float64(1))
    largest = np.finfo(np.float64).max
    values = np.array(
        [
            0, -0.0, tiny, -tiny, np.finfo(np.float64).tiny,
            -np.finfo(np.float64).tiny, 1, -1,
            np.nextafter(1.0, 2.0), np.nextafter(1.0, 0.0),
            2.0**53, -(2.0**53), largest, -largest,
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(70421)
    pool_points = rng.choice(values, size=(450, 3, 2))
    random_bits = rng.integers(0, np.iinfo(np.uint64).max, size=(450, 3, 2), dtype=np.uint64)
    random_points = random_bits.view(np.float64)
    random_points = random_points[np.isfinite(random_points).all(axis=(1, 2))]
    triples = np.concatenate((pool_points, random_points))

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lower, upper = _orientation_interval(triples[:, 0], triples[:, 1], triples[:, 2])
    assert len(lower) == len(triples)
    for (p, q, r), lo, hi in zip(triples, lower, upper):
        px, py, qx, qy, rx, ry = map(_rational, (*p, *q, *r))
        det = (qx - px) * (ry - py) - (qy - py) * (rx - px)
        assert _bound_contains(lo, det, lower=True), (p, q, r, lo, det)
        assert _bound_contains(hi, det, lower=False), (p, q, r, hi, det)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_cell_certificate_agrees_with_exact_four_corner_oracle(
    tmp_path: Path, dtype: type[np.floating],
) -> None:
    rng = np.random.default_rng(90632)
    specimens: list[np.ndarray] = [
        np.array([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], dtype=dtype),
        np.array([[[0, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=dtype),
        np.array([[[0, 0], [1, 0]], [[0, 1], [-1, 1]]], dtype=dtype),
    ]
    if dtype == np.float64:
        small = np.nextafter(dtype(0), dtype(1))
        huge = np.finfo(dtype).max
        specimens.extend(
            [
                np.array([[[0, 0], [small, 0]], [[0, 1], [small, 1]]], dtype=dtype),
                np.array([[[-huge, -huge], [huge, -huge]],
                          [[-huge, huge], [huge, huge]]], dtype=dtype),
            ]
        )
    for _ in range(150):
        cell = rng.normal(size=(2, 2, 2)).astype(dtype)
        exponent = int(rng.integers(-120, 120)) if dtype == np.float32 else int(rng.integers(-1000, 1000))
        with np.errstate(over="ignore", under="ignore"):
            cell = np.ldexp(cell, exponent).astype(dtype)
        specimens.append(cell)

    path = tmp_path / "cell.npz"
    for cell in specimens:
        np.savez(path, vertices=cell[None], boundary_reference=cell[None])
        certified = certify_q1_binary_map(path)
        signs = _corner_signs(cell)
        assert certified["positive_corners"] == sum(sign > 0 for sign in signs)
        assert certified["nonpositive_corners"] == sum(sign <= 0 for sign in signs)
        assert not certified["valid"] or all(sign > 0 for sign in signs)
        if np.array_equal(cell, specimens[0]):
            assert certified["valid"]


def test_batch_row_indexing_and_boundary_rejection(tmp_path: Path) -> None:
    rows, columns = np.meshgrid(np.arange(5), np.arange(6), indexing="ij")
    rectangular = np.stack((columns, rows), axis=-1).astype(np.float64)
    vertices = np.stack((rectangular, rectangular.copy()))
    vertices[1, 2, 2] = [3.5, 2.0]
    expected = [
        _corner_signs(vertices[batch, row:row + 2, col:col + 2])
        for batch in range(2) for row in range(4) for col in range(5)
    ]
    path = tmp_path / "batch.npz"
    np.savez(path, vertices=vertices, boundary_reference=vertices)
    result = certify_q1_binary_map(path, chunk_rows=1)
    assert result["corners"] == 2 * 4 * 5 * 4
    assert result["positive_corners"] == sum(s > 0 for cell in expected for s in cell)
    assert result["nonpositive_corners"] == sum(s <= 0 for cell in expected for s in cell)

    reference = vertices.copy()
    reference[0, 0, 1, 0] += 1
    np.savez(path, vertices=vertices, boundary_reference=reference)
    mismatch = certify_q1_binary_map(path, chunk_rows=2)
    assert not mismatch["valid"]
    assert not mismatch["boundary_exact"]


def test_positive_corners_need_independently_simple_boundary(tmp_path: Path) -> None:
    # A rectangular parameter domain wraps twice around an annulus. Every
    # corner is oriented, but columns 0 and 16 coincide, so it is not injective.
    theta = -2 * np.pi * (np.arange(33) % 16) / 16
    unit = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    vertices = np.stack((unit, 2 * unit), axis=0)[None]
    assert np.array_equal(vertices[:, :, 0], vertices[:, :, 16])
    path = tmp_path / "wrapped_boundary.npz"
    np.savez(path, vertices=vertices, boundary_reference=vertices)
    result = certify_q1_binary_map(path)
    assert result["positive_corners"] == result["corners"] == 128
    assert not result["valid"]


def test_empty_batch_is_not_a_certified_map(tmp_path: Path) -> None:
    empty = np.empty((0, 2, 2, 2), dtype=np.float32)
    path = tmp_path / "empty.npz"
    np.savez(path, vertices=empty, boundary_reference=empty)
    try:
        result = certify_q1_binary_map(path)
    except ValueError:
        return
    assert not result["valid"]


def test_saved_float32_map_all_four_corners_with_integer_oracle() -> None:
    path = Path(__file__).resolve().parents[1] / "docs/digital_topology_wsi/q1_257_f1_float32.npz"
    with np.load(path, allow_pickle=False) as archive:
        vertices = archive["vertices"]
        reference = archive["boundary_reference"]
    assert vertices.dtype == np.float32 and vertices.shape == (1, 257, 257, 2)
    assert np.isfinite(vertices).all() and np.isfinite(reference).all()
    axis = np.arange(257, dtype=np.float32) / np.float32(256)
    assert np.array_equal(reference[0, :, :, 0], np.broadcast_to(axis[None], (257, 257)))
    assert np.array_equal(reference[0, :, :, 1], np.broadcast_to(axis[:, None], (257, 257)))
    assert np.array_equal(vertices[:, 0], reference[:, 0])
    assert np.array_equal(vertices[:, -1], reference[:, -1])
    assert np.array_equal(vertices[:, :, 0], reference[:, :, 0])
    assert np.array_equal(vertices[:, :, -1], reference[:, :, -1])

    # Every finite binary32 coordinate is an integer multiple of 2**-149.
    scale = 1 << 149
    integers = []
    for value in vertices.flat:
        numerator, denominator = float(value).as_integer_ratio()
        assert scale % denominator == 0
        integers.append(numerator * (scale // denominator))
    exact = np.array(integers, dtype=object).reshape(vertices.shape)
    a, b = exact[:, :-1, :-1], exact[:, :-1, 1:]
    c, d = exact[:, 1:, 1:], exact[:, 1:, :-1]
    assert all(
        np.all((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
               - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]) > 0)
        for p, q, r in ((a, b, d), (a, b, c), (d, b, c), (a, c, d))
    )
    result = certify_q1_binary_map(path, chunk_rows=17)
    assert result["valid"] and result["positive_corners"] == 4 * 256 * 256
