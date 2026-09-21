"""Receipt-level checks for Route-II MVC encode/decode experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "experiments/phase5/route2_mvc_roundtrip.py"
    spec = importlib.util.spec_from_file_location("phase5_route2_mvc_roundtrip", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_small_roundtrip_reports_geometry_conditioning_and_redundancy() -> None:
    row = _module().run_case(control_side=7, seed=1701, strength=0.25)
    assert row["topology_certified"]
    assert row["maximum_vertex_error"] < 1e-12
    assert row["map_rmse"] < 1e-12
    assert row["maximum_barycentric_residual"] < 1e-12
    assert row["minimum_mvc_probability"] > 0.0
    assert row["maximum_covariance_condition"] >= 1.0
    assert row["minimum_triangle_quality"] > 0.0
    assert row["minimum_area_ratio"] > 0.0
    assert row["canonical_system_condition_one_estimate"] >= 1.0
    assert row["raw_to_canonical_probability_l2"] > 0.0
    assert row["maximum_canonical_supported_row_mean"] < 1e-12

