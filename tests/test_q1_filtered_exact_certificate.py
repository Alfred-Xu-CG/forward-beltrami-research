"""A saved binary map needs a sign check independent of torch arithmetic."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def test_saved_float32_257_map_has_certified_positive_binary_corners() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs" / "digital_topology_wsi" / "q1_257_f1_float32.npz"
    )
    result = certify_q1_binary_map(path, chunk_rows=32)
    assert result["valid"]
    assert result["corners"] == 262144
    assert result["positive_corners"] == 262144
    assert result["nonpositive_corners"] == 0
    assert result["boundary_exact"]


def test_collapsed_float32_export_is_not_certified() -> None:
    # The source map is a positive translated square in exact arithmetic,
    # but binary32 erases its unit-sized edges at this large origin.
    original = np.array([[[[0, 0], [1, 0]], [[0, 1], [1, 1]]]], dtype=np.float64)
    collapsed = (original + 2**24).astype(np.float32)
    destination = Path(__file__).resolve().parents[1] / "docs" / "digital_topology_wsi"
    with TemporaryDirectory(dir=destination) as temporary:
        path = Path(temporary) / "collapsed.npz"
        np.savez(path, vertices=collapsed, boundary_reference=collapsed)
        result = certify_q1_binary_map(path)
        assert not result["valid"]
        assert result["nonpositive_corners"] == 4


def test_fraction_fallback_classifies_exact_zero_and_tiny_positive() -> None:
    # A unit-square cell compressed in x to the smallest positive binary64
    # subnormal has a positive *exact* determinant that naive products can
    # underflow to zero. The filtered sign check must not call it folded.
    tiny = np.nextafter(np.float64(0), np.float64(1))
    positive = np.array([[[[0., 0.], [tiny, 0.]], [[0., 1.], [tiny, 1.]]]])
    reference = positive.copy()
    destination = Path(__file__).resolve().parents[1] / "docs" / "digital_topology_wsi"
    with TemporaryDirectory(dir=destination) as temporary:
        path = Path(temporary) / "tiny.npz"
        np.savez(path, vertices=positive, boundary_reference=reference)
        result = certify_q1_binary_map(path)
        assert result["valid"]
        assert result["exact_fallback_corners"] > 0
        zero = positive.copy()
        zero[:, :, 1, 0] = 0
        np.savez(path, vertices=zero, boundary_reference=zero)
        result = certify_q1_binary_map(path)
        assert not result["valid"]
        assert result["nonpositive_corners"] == 4


def test_empty_batch_is_not_a_valid_map() -> None:
    vertices = np.empty((0, 2, 2, 2), dtype=np.float32)
    destination = Path(__file__).resolve().parents[1] / "docs" / "digital_topology_wsi"
    with TemporaryDirectory(dir=destination) as temporary:
        path = Path(temporary) / "empty.npz"
        np.savez(path, vertices=vertices, boundary_reference=vertices)
        with pytest.raises(ValueError, match="nonempty batch"):
            certify_q1_binary_map(path)
