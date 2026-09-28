"""Independent NumPy audit of one actual exported 257² float32 Q1 map."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_saved_remote_q1_map_has_positive_corners_and_fixed_boundary() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "docs" / "digital_topology_wsi" / "q1_257_f1_float32.npz"
    )
    with np.load(artifact, allow_pickle=False) as arrays:
        vertices = arrays["vertices"]
        reference = arrays["boundary_reference"]
    assert vertices.shape == reference.shape == (1, 257, 257, 2)
    assert vertices.dtype == reference.dtype == np.float32
    assert np.isfinite(vertices).all()
    for actual, expected in (
        (vertices[:, 0], reference[:, 0]),
        (vertices[:, -1], reference[:, -1]),
        (vertices[:, :, 0], reference[:, :, 0]),
        (vertices[:, :, -1], reference[:, :, -1]),
    ):
        assert np.array_equal(actual, expected)

    # Convert the *saved binary32 coordinates* to exact-representable binary64
    # before an independent four-corner calculation, not the production torch
    # helper. This is a strong numerical check, not an outward-rounded proof.
    v = vertices.astype(np.float64)
    a, b = v[:, :-1, :-1], v[:, :-1, 1:]
    c, d = v[:, 1:, 1:], v[:, 1:, :-1]

    def cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

    corners = np.stack((
        cross(b - a, d - a),
        cross(b - a, c - b),
        cross(c - d, c - b),
        cross(c - d, d - a),
    ), axis=-1)
    assert corners.shape == (1, 256, 256, 4)
    assert np.isfinite(corners).all()
    assert corners.min() > 0
