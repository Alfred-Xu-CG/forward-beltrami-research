"""CPU capture tests are NOT Inductor performance evidence."""
import argparse
import json

import numpy as np
import pytest
import torch

from tools import coordinated_compiled_priors as benchmark


def vertices(rows=5, columns=7, scale=1.):
    y, x = torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64),
                          torch.linspace(0,1,columns,dtype=torch.float64), indexing="ij")
    return torch.stack((scale*x+.02*x*y, y+.03*x*(1-x)), -1)[None]


def save(path, *, scale=1., interpolation="p1_ac", dtype=np.float64):
    np.savez(path, vertices=vertices(scale=scale).numpy().astype(dtype), interpolation=np.asarray(interpolation))
    return path


def test_factory_calls_exact_existing_functions_and_compile_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "p1_arap_energy", lambda x, **kw: calls.append((x,kw)) or x.sum())
    monkeypatch.setattr(benchmark, "corner_symmetric_dirichlet", lambda x: x.square().sum())
    x = vertices()
    expected = (x.sum(), x.square().sum())
    actual = benchmark.make_joint_priors()(x)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert calls[0][1] == dict(diagonal="ac",validate=False)
    seen = {}
    def compile_mock(function, **kwargs):
        seen.update(kwargs)
        return function
    monkeypatch.setattr(torch,"compile",compile_mock)
    assert benchmark.make_joint_priors("inductor") is benchmark.joint_priors
    assert seen == dict(backend="inductor",fullgraph=True,dynamic=False,mode="default")
    with pytest.raises(ValueError):
        benchmark.make_joint_priors("unknown")


def test_actual_cpu_fullgraph_eager_backend_values_and_full_gradient():
    x = vertices()
    eager = benchmark.measure_call(benchmark.make_joint_priors(),x)
    capture = benchmark.measure_call(benchmark.make_joint_priors("eager"),x)
    agreement = benchmark.compare(eager[1:],capture[1:])
    assert agreement["passed"]
    assert agreement["full_vertex_gradient"]["shape"] == list(x.shape)
    reference = x.clone().requires_grad_(True)
    arap, shape = benchmark.joint_priors(reference)
    weighted_gradient, = torch.autograd.grad(3*arap+1e-4*shape,reference)
    torch.testing.assert_close(eager[2],weighted_gradient)
    assert eager[0]["forward_seconds"] > 0 and eager[0]["weighted_vjp_seconds"] > 0
    assert eager[0]["cuda_forward_peak_allocated_bytes"] is None


def test_selection_keeps_predeclared_anchor_first_and_uses_qmin_only(tmp_path):
    anchor=save(tmp_path/"he_to_cc10_analytic.npz")
    easy=save(tmp_path/"easy.npz",scale=.8)
    hard=save(tmp_path/"hard.npz",scale=.3)
    cases,candidates=benchmark.select_cases(anchor,[easy,hard,anchor])
    assert [row["path"] for _,row in cases] == [str(anchor),str(hard)]
    assert len(candidates)==2 and cases[1][1]["qmin"] < cases[0][1]["qmin"]


@pytest.mark.parametrize("kwargs",[dict(interpolation="q1"),dict(dtype=np.float32),dict(scale=-1)])
def test_saved_geometry_guard(tmp_path,kwargs):
    with pytest.raises(ValueError):
        benchmark.load_anchor(save(tmp_path/"bad.npz",**kwargs))


def test_repeat_order_reporting_and_capture_evidence_label(tmp_path):
    args=argparse.Namespace(anchor=save(tmp_path/"he_to_cc10_analytic.npz"),geometry_candidates=[],
        output=tmp_path/"report.json",device="cpu",backend="eager",warmups=1,repeats=2,threads=1)
    report=benchmark.run(args)
    assert report["status"]=="complete"
    assert "NOT Inductor performance evidence" in report["evidence_scope"]
    assert report["annotations_images_matcher_read"] is False
    case=report["cases"][0]
    assert len(case["cold"])==2 and len(case["warmups"])==2 and len(case["repeats"])==4
    assert [row["implementation"] for row in case["repeats"]]==["eager","compiled","compiled","eager"]
    assert all(row["passed"] for row in case["comparisons"])
    assert json.loads(args.output.read_text())["weights"]==dict(arap=3.,shape=1e-4)
    with pytest.raises(FileExistsError):
        benchmark.run(args)


def test_backend_error_propagates_without_eager_fallback(tmp_path,monkeypatch):
    def failing_compile(*args,**kwargs):
        raise RuntimeError("deliberate backend failure")
    monkeypatch.setattr(torch,"compile",failing_compile)
    args=argparse.Namespace(anchor=save(tmp_path/"he_to_cc10_analytic.npz"),geometry_candidates=[],
        output=tmp_path/"failed.json",device="cpu",backend="inductor",warmups=0,repeats=1,threads=1)
    with pytest.raises(RuntimeError,match="deliberate backend failure"):
        benchmark.run(args)
    assert json.loads(args.output.read_text())["status"]=="failed_no_fallback"


def test_gradient_mismatch_prevents_speed_summary(monkeypatch):
    original=benchmark.make_joint_priors
    def changed(backend=None):
        if backend is None:
            return original()
        return lambda x: (benchmark.joint_priors(x)[0]*1.01, benchmark.joint_priors(x)[1])
    monkeypatch.setattr(benchmark,"make_joint_priors",changed)
    case=benchmark.benchmark_case(vertices(),{},backend="eager",warmups=1,repeats=2)
    assert case["status"]=="numerical_mismatch_no_speed_claim"
    assert "steady_summary" not in case and not case["repeats"]


def test_nonfinite_failure_diagnostics_remain_strict_json_serializable():
    reference=(torch.ones(2),torch.ones(1,2,2,2))
    candidate=(torch.tensor([float("nan"),1.]),torch.full((1,2,2,2),float("inf")))
    comparison=benchmark.compare(reference,candidate)
    assert not comparison["passed"]
    assert comparison["values"]["max_abs"] is None
    json.dumps(comparison,allow_nan=False)
