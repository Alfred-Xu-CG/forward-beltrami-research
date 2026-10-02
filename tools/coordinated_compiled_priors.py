"""Isolated joint-prior compilation experiment; never changes production dispatch.

The closure calls the existing P1 AC ARAP and four-corner shape energies.
Only saved float64 geometry is loaded: no images, labels, or matcher evidence.
Cold calls are observed costs, NOT an isolated measurement of compiler time.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from tools.coordinated_real_case import corner_symmetric_dirichlet


def joint_priors(vertices):
    """Trusted geometry path: validation remains outside the compiled closure."""
    return (p1_arap_energy(vertices, diagonal="ac", validate=False),
            corner_symmetric_dirichlet(vertices))


def make_joint_priors(compile_backend=None):
    """None = original eager; 'eager' = capture test; 'inductor' = compilation.

    Exact torch 2.5.1 source supports these keyword arguments. No retry,
    alternate backend, graph-break allowance, or eager fallback is installed.
    """
    if compile_backend is None:
        return joint_priors
    if compile_backend not in ("eager", "inductor"):
        raise ValueError("compile_backend must be None, eager, or inductor")
    return torch.compile(joint_priors, backend=compile_backend,
                         fullgraph=True, dynamic=False, mode="default")


def validate_geometry(vertices):
    if (vertices.dtype != torch.float64 or vertices.ndim != 4 or
            vertices.shape[0] != 1 or vertices.shape[-1] != 2 or
            min(vertices.shape[1:3]) < 2):
        raise ValueError("saved geometry must have float64 shape (1,R>=2,C>=2,2)")
    if not bool(torch.isfinite(vertices).all()):
        raise ValueError("saved geometry must be finite")
    rows, columns = vertices.shape[1:3]
    corners = q1_corner_determinants(vertices) * ((rows-1)*(columns-1))
    qmin = float(corners.amin())
    if not bool(torch.isfinite(corners).all()) or qmin <= 0:
        raise ValueError("all four normalized corner determinants must be positive")
    return qmin


def load_anchor(path):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        if "interpolation" not in archive or str(archive["interpolation"].item()) != "p1_ac":
            raise ValueError("saved map must explicitly declare p1_ac")
        vertices = torch.from_numpy(archive["vertices"].copy())
    qmin = validate_geometry(vertices)
    return vertices, dict(path=str(path), qmin=qmin,
        qmin_definition="minimum of all four Q1 corner determinants divided by uniform source cell area; geometry selection only, not global certification",
        shape=list(vertices.shape), input_side=list(vertices.shape[1:3]),
        control_vertices=int(vertices.shape[1]*vertices.shape[2]),
        dtype=str(vertices.dtype), interpolation="p1_ac")


def select_cases(anchor, geometry_candidates=()):
    """Anchor always first; optional second is minimum qmin, tie broken by path."""
    first, metadata = load_anchor(anchor)
    metadata["selection"] = "predeclared HE->CC10 anchor supplied by caller; identity not inferred from geometry"
    cases = [(first, metadata)]
    candidates = []
    for path in sorted({str(Path(p).resolve()) for p in geometry_candidates}):
        if path == metadata["path"]:
            continue
        vertices, row = load_anchor(path)
        candidates.append((vertices, row))
    if candidates:
        second = min(candidates, key=lambda item: (item[1]["qmin"], item[1]["path"]))
        second[1]["selection"] = "smallest positive qmin among explicitly supplied candidate maps; no labels read"
        cases.append(second)
    return cases, [row for _, row in candidates]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_call(function, vertices):
    """Grad-enabled F and weighted VJP separately; input cloning is outside timing."""
    x = vertices.detach().clone().requires_grad_(True)
    seeds = (x.new_tensor(3.), x.new_tensor(1e-4))
    device = x.device
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    outputs = function(x)
    synchronize(device)
    forward_seconds = time.perf_counter()-start
    forward_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    vjp_baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    gradient, = torch.autograd.grad(outputs, x, grad_outputs=seeds)
    synchronize(device)
    vjp_seconds = time.perf_counter()-start
    vjp_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    row = dict(forward_seconds=forward_seconds, weighted_vjp_seconds=vjp_seconds,
        forward_plus_vjp_seconds=forward_seconds+vjp_seconds,
        cuda_baseline_bytes=baseline, cuda_forward_peak_allocated_bytes=forward_peak,
        cuda_forward_peak_increment_bytes=None if baseline is None else forward_peak-baseline,
        cuda_vjp_baseline_bytes=vjp_baseline, cuda_vjp_peak_allocated_bytes=vjp_peak,
        cuda_vjp_peak_increment_bytes=None if baseline is None else vjp_peak-vjp_baseline,
        device=str(device), input_side=list(x.shape[1:3]),
        control_vertices=int(x.shape[1]*x.shape[2]), dtype=str(x.dtype))
    # Transfers, comparisons, and reductions below are excluded from timing.
    values = torch.stack(tuple(value.detach() for value in outputs)).cpu()
    return row, values, gradient.detach().cpu()


def compare(eager, compiled, *, rtol=1e-8, atol=1e-10):
    def finite_number(value):
        value = float(value)
        return value if math.isfinite(value) else None
    result = dict(rtol=rtol, atol=atol)
    for name, reference, candidate in zip(("values", "full_vertex_gradient"), eager, compiled):
        difference = candidate-reference
        finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
        passed = finite and bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol))
        result[name] = dict(passed=passed, shape=list(reference.shape),
            finite=finite, max_abs=finite_number(difference.abs().amax()), rms=finite_number(difference.square().mean().sqrt()),
            relative_l2=finite_number(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(reference).clamp_min(1e-30)))
    result["passed"] = all(result[name]["passed"] for name in ("values", "full_vertex_gradient"))
    result["eager_values_arap_shape"] = [finite_number(value) for value in eager[0]]
    result["compiled_values_arap_shape"] = [finite_number(value) for value in compiled[0]]
    return result


def benchmark_case(vertices, metadata, *, backend="inductor", warmups=3, repeats=10):
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats positive")
    validate_geometry(vertices)
    functions = {"eager": make_joint_priors()}
    factory_start = time.perf_counter()
    functions["compiled"] = make_joint_priors(backend)
    result = dict(input=metadata, compiled_factory_seconds=time.perf_counter()-factory_start,
        cold=[], warmups=[], repeats=[], comparisons=[])
    for phase, count in (("cold", 1), ("warmups", warmups), ("repeats", repeats)):
        for index in range(count):
            order = ("eager", "compiled") if index % 2 == 0 else ("compiled", "eager")
            outputs = {}
            for position, name in enumerate(order):
                row, values, gradient = measure_call(functions[name], vertices)
                row.update(phase=phase, repeat=index, order=list(order), order_position=position,
                           implementation=name, compiled_backend=backend if name == "compiled" else None)
                result[phase].append(row)
                outputs[name] = (values, gradient)
            agreement = compare(outputs["eager"], outputs["compiled"])
            agreement.update(phase=phase, repeat=index)
            result["comparisons"].append(agreement)
            if not agreement["passed"]:
                result["status"] = "numerical_mismatch_no_speed_claim"
                return result
    summary = {}
    for name in functions:
        rows = [row for row in result["repeats"] if row["implementation"] == name]
        summary[name] = {key+"_median": statistics.median(row[key] for row in rows)
                        for key in ("forward_seconds", "weighted_vjp_seconds", "forward_plus_vjp_seconds")}
        for key in ("cuda_forward_peak_allocated_bytes", "cuda_vjp_peak_allocated_bytes",
                    "cuda_forward_peak_increment_bytes", "cuda_vjp_peak_increment_bytes"):
            summary[name][key+"_max"] = max(row[key] for row in rows) if rows[0][key] is not None else None
    summary["eager_over_compiled_median_ratios"] = {
        key: summary["eager"][key]/summary["compiled"][key]
        for key in ("forward_seconds_median", "weighted_vjp_seconds_median", "forward_plus_vjp_seconds_median")}
    result.update(status="complete", steady_summary=summary)
    return result


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark report: {output}")
    cases, selection_rows = select_cases(args.anchor, args.geometry_candidates)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("only cpu and cuda benchmark devices are supported")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    report = dict(status="running", question="Does one unchanged joint P1 AC ARAP + corner-shape closure gain value/VJP throughput with fullgraph default Inductor?",
        torch_version=torch.__version__, device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        threads=torch.get_num_threads(), compile_backend=args.backend,
        compile_settings=dict(fullgraph=True, dynamic=False, mode="default"),
        evidence_scope="Inductor operator evidence only; not end-to-end optimizer evidence" if args.backend == "inductor" else "eager-backend graph capture correctness only; NOT Inductor performance evidence",
        weights=dict(arap=3., shape=1e-4), warmup_pairs=args.warmups, timed_pairs=args.repeats,
        annotations_images_matcher_read=False, production_dispatch_modified=False,
        cold_scope="First observed forward and first weighted VJP recorded separately for eager and compiled, in eager-then-compiled order. Prior process/disk compiler caches are not cleared. Costs include runtime initialization and execution; difference is NOT exclusively compilation cost. Later cases may reuse compilation caches.",
        timing_scope="Synchronized host elapsed for grad-enabled joint forward and VJP seeded (3,1e-4), separately. Input clone, output CPU copies, correctness checks and input validation excluded. F+VJP sums two separately synchronized intervals; no speedup claim for trial-only no_grad forward.",
        memory_scope="CUDA allocator allocated peaks per phase plus phase-start baselines/increments; forward graph stays live at VJP baseline. Not reserved/process/compiler-host memory or isolated graph-save size.",
        geometry_candidate_selection=selection_rows, cases=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for vertices, metadata in cases:
            report["cases"].append(benchmark_case(vertices.to(device), metadata, backend=args.backend,
                                                 warmups=args.warmups, repeats=args.repeats))
        report["status"] = "complete" if all(row["status"] == "complete" for row in report["cases"]) else "numerical_mismatch_no_speed_claim"
    except Exception as exc:
        report.update(status="failed_no_fallback", error=f"{type(exc).__name__}: {exc}")
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        raise
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True, help="predeclared saved HE->CC10 analytic P1 AC NPZ")
    parser.add_argument("--geometry-candidates", type=Path, nargs="*", default=[], help="optional saved maps; choose ONE by smallest positive qmin, never labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("inductor", "eager"), default="inductor")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    report = run(parser.parse_args())
    print(json.dumps(dict(status=report["status"], cases=len(report["cases"]))))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
