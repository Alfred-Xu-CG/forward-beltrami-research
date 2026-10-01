from types import SimpleNamespace

import pytest

from tools.coordinated_nested_priors_benchmark import benchmark_case,float32_rounding_diagnostics


@pytest.mark.parametrize("diagonal",["ac","bd"])
def test_small_prior_benchmark_keeps_setup_checks_and_full_y_gradient(diagonal):
    args=SimpleNamespace(device="cpu",warmup=3,repeats=10)
    result=benchmark_case(3,9,diagonal,args)
    assert result["status"]=="success" and result["factor"]==4
    assert result["weighted_gradient_relative_l2_error"]<1e-10
    assert result["fresh_actual_normalized_fine_qmin"]>.001
    assert result["fresh_actual_fine_corner_check_seconds"]>=0
    for method in ("explicit_fine","coarse_quadrature"):
        row=result[method]
        assert row["constructor_seconds"]>=0 and row["constant_buffer_bytes"]>0
        assert len(row["forward_seconds"])==len(row["vjp_seconds"])==10
        assert row["resident_before_constructor_bytes"]==row["resident_after_cleanup_bytes"]


def test_float32_thin_rounding_errors_are_explicit_not_hidden():
    rows=float32_rounding_diagnostics("cpu")
    assert len(rows)==4 and all(row["finite"] for row in rows)
    assert any(row["relative_gradient_l2_error"]>1e-5 for row in rows)
