"""Contract tests for the compact Route-II incremental benchmark receipt."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "experiments/phase5/route2_incremental_benchmark.py"
    spec = importlib.util.spec_from_file_location("phase5_route2_incremental", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_small_local_and_global_cases_report_identical_system_accuracy() -> None:
    module = _module()
    local = module.run_case(control_side=7, seed=501, update_kind="local_one")
    global_case = module.run_case(control_side=7, seed=501, update_kind="global_large")

    for row in (local, global_case):
        assert row["topology"]["certified"]
        assert row["cold"]["maximum_error_vs_refactor"] < 1e-9
        assert row["warm"]["maximum_error_vs_refactor"] < 1e-9
        assert row["correction"]["maximum_error_vs_refactor"] < 1e-9
        assert row["changed_row_count"] >= 1
        assert row["new_factorization_and_solve_seconds"] >= 0.0
        assert row["old_factorization_seconds"] >= 0.0

    assert local["woodbury"]["status"] == "success"
    assert local["woodbury"]["maximum_error_vs_refactor"] < 1e-10
    assert global_case["woodbury"]["status"] == "not_run"
    assert "dense Schur" in global_case["woodbury"]["reason"]
