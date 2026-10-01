"""Geometry-only full-latent cascade benchmark; no encoder/image optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import weakref

import torch
import numpy as np

from qcopt.neural_bijection.dense.coordinated_multilevel import CoordinatedMultilevelDecoder
from qcopt.neural_bijection.dense.coordinated_refinement import refine_p1_vertices
from qcopt.neural_bijection.dense.coordinated_update import interpolate_proposal
from tools.coordinated_dense_operator_benchmark import normalized_corners


BACKENDS = ("existing", "manual", "checkpointed_manual")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _allocated(device):
    return torch.cuda.memory_allocated(device) if device.type == "cuda" else None


def make_latents(levels, batch, dtype, device, seed):
    """Deterministic smooth plus fine coefficient probes, not registration truth."""
    latents = []
    for level in levels:
        generator = torch.Generator(device=device).manual_seed(seed + level)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, level, dtype=dtype, device=device),
                                torch.linspace(0, 1, level, dtype=dtype, device=device),
                                indexing="ij")
        smooth = torch.stack((torch.sin(2 * torch.pi * xx) * torch.sin(torch.pi * yy),
                              torch.cos(torch.pi * xx) * torch.sin(2 * torch.pi * yy)))
        noise = torch.randn(batch, 2, level - 2, level - 2, generator=generator,
                            dtype=dtype, device=device)
        latents.append(.35 * smooth[None, :, 1:-1, 1:-1] + .5 * noise)
    return latents


def retained_storage(function, variables):
    """Detached hook payloads retain storage without introducing graph cycles."""
    class Handle:
        def __init__(self, tensor):
            self.tensor = tensor.detach()

    handles = []

    def pack(tensor):
        handle = Handle(tensor)
        handles.append(weakref.ref(handle))
        return handle

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda h: h.tensor):
        candidate = function(*variables)
    retained = [ref() for ref in handles if ref() is not None]
    unique = {}
    logical = 0
    for handle in retained:
        tensor = handle.tensor
        storage = tensor.untyped_storage()
        unique[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        logical += tensor.numel() * tensor.element_size()
    result = dict(retained_saved_tensor_count=len(retained),
                  retained_saved_logical_bytes=logical,
                  retained_saved_unique_storage_bytes=sum(unique.values()),
                  retained_saved_unique_storage_count=len(unique))
    del candidate, retained
    # The loop's last local handle is also a strong reference.
    if handles:
        try:
            del handle, tensor, storage
        except UnboundLocalError:
            pass
    result["all_hook_handles_collected"] = all(ref() is None for ref in handles)
    if not result["all_hook_handles_collected"]:
        raise RuntimeError("saved-storage probe leaked hook handles")
    return result


def _new_variables(anchor, latents):
    return [anchor.detach().clone().requires_grad_(),
            *[z.detach().clone().requires_grad_() for z in latents]]


def _boundary_equal(candidate, anchor):
    return all(torch.equal(a, b) for a, b in (
        (candidate[:, 0], anchor[:, 0]), (candidate[:, -1], anchor[:, -1]),
        (candidate[:, :, 0], anchor[:, :, 0]), (candidate[:, :, -1], anchor[:, :, -1])))


def trace_stages(anchor, module, latents):
    """Untimed ordinary-layer diagnostics and independent NumPy actual margins."""
    current = anchor
    trace = []
    with torch.no_grad():
        for block, latent in zip(module.blocks, latents):
            for stage in (block.x, block.y):
                coefficients = torch.nn.functional.pad(
                    stage.amplitude * 16 / (stage.level - 1) *
                    torch.tanh(latent[:, stage.component]), (1, 1, 1, 1))
                proposal = interpolate_proposal(coefficients, stage.shape)
                diagnostics = stage.layer(current, proposal, reference=module.reference,
                                          validate=True)
                current = diagnostics.vertices
                qmin = normalized_corners(current, module.reference)
                trace.append(dict(stage_index=stage.stage_index, level=stage.level,
                                  component=stage.component,
                                  raw_coefficient_amplitude=stage.amplitude * 16 / (stage.level - 1),
                                  scale=diagnostics.scale.cpu().tolist(),
                                  gauge=diagnostics.gauge.cpu().tolist(),
                                  actual_normalized_qmin=qmin,
                                  fixed_boundary_equal=_boundary_equal(current, anchor)))
    return trace


def _module(anchor, reference, levels, backend, args):
    return CoordinatedMultilevelDecoder(
        anchor.shape[1], anchor.shape[2], levels, reference=reference,
        backend=backend, minimum_jacobian=args.minimum_jacobian,
        theta=args.theta, amplitude=args.amplitude, diagonal=args.diagonal)


def compare_chain(anchor, reference, levels, latents, upstream, args):
    """Separate correctness pass; all gradients discarded before memory timing."""
    ordinary = _module(anchor, reference, levels, "existing", args)
    variables = _new_variables(anchor, latents)
    expected = ordinary(variables[0], variables[1:])
    gradients = torch.autograd.grad(expected, variables, upstream)
    expected = expected.detach()
    labels = ["initial_vertices", *[f"latent_{l}" for l in levels]]
    metrics = {}
    tiny = torch.finfo(anchor.dtype).tiny
    for backend in BACKENDS:
        module = _module(anchor, reference, levels, backend, args)
        probe = _new_variables(anchor, latents)
        actual = module(probe[0], probe[1:])
        actual_gradients = torch.autograd.grad(actual, probe, upstream)
        value_error = float((actual - expected).abs().max())
        errors = []
        for label, actual_g, reference_g in zip(labels, actual_gradients, gradients):
            errors.append(dict(variable=label,
                maximum_absolute_error=float((actual_g - reference_g).abs().max()),
                relative_l2_error=float(torch.linalg.vector_norm(actual_g - reference_g) /
                                        torch.linalg.vector_norm(reference_g).clamp_min(tiny)),
                reference_gradient_rms=float(reference_g.square().mean().sqrt())))
        finite = bool(torch.isfinite(actual).all() and all(torch.isfinite(g).all() for g in actual_gradients))
        if value_error != 0 or not finite or any(e["relative_l2_error"] > 1e-9 for e in errors):
            raise RuntimeError(f"{backend} full-chain value/gradient comparison failed")
        metrics[backend] = dict(candidate_maximum_absolute_error=value_error,
                               gradient_errors=errors, gradients_finite=finite)
        del module, probe, actual, actual_gradients, actual_g, reference_g
    del ordinary, variables, expected, gradients
    return metrics


def measure_backend(anchor, reference, levels, latents, upstream, backend, args):
    if args.warmup < 3 or args.repeats < 10:
        raise ValueError("at least3 warmups and10 repeats required")
    device = anchor.device
    _sync(device)
    constructor_baseline = _allocated(device)
    tick = time.perf_counter()
    module = _module(anchor, reference, levels, backend, args)
    _sync(device)
    constructor_seconds = time.perf_counter() - tick
    constructor_after = _allocated(device)
    function = lambda y, *z: module(y, z)
    probe_baseline = _allocated(device)
    variables = _new_variables(anchor, latents)
    storage = retained_storage(function, variables)
    del variables
    _sync(device)
    probe_after = _allocated(device)
    if probe_after != probe_baseline:
        raise RuntimeError("storage probe CUDA allocation did not return to baseline")
    storage.update(storage_probe_resident_before_bytes=probe_baseline,
                   storage_probe_resident_after_bytes=probe_after)
    forwards, backwards, copies, residents, peaks = [], [], [], [], []
    for repeat in range(args.warmup + args.repeats):
        _sync(device)
        tick = time.perf_counter()
        variables = _new_variables(anchor, latents)
        _sync(device)
        copy_seconds = time.perf_counter() - tick
        resident = _allocated(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        candidate = module(variables[0], variables[1:])
        _sync(device)
        forward = time.perf_counter() - tick
        tick = time.perf_counter()
        gradients = torch.autograd.grad(candidate, variables, upstream)
        _sync(device)
        backward = time.perf_counter() - tick
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        if not bool(torch.isfinite(candidate).all() and all(torch.isfinite(g).all() for g in gradients)):
            raise RuntimeError("nonfinite complete-chain candidate or gradient")
        if repeat >= args.warmup:
            forwards.append(forward); backwards.append(backward); copies.append(copy_seconds)
            residents.append(resident); peaks.append(peak)
        last = candidate.detach()
        del variables, candidate, gradients
        if repeat != args.warmup + args.repeats - 1:
            del last
    qmin = normalized_corners(last, reference)
    boundary_equal = _boundary_equal(last, anchor)
    if qmin <= args.minimum_jacobian or not boundary_equal:
        raise RuntimeError("fresh actual final margin/boundary verification failed")
    result = dict(**storage, constructor_seconds=constructor_seconds,
        constructor_resident_increment_bytes=None if constructor_baseline is None else constructor_after - constructor_baseline,
        buffer_bytes=sum(b.numel() * b.element_size() for b in module.buffers()),
        full_control_latent=module.full_control_latent, stage_count=module.stage_count,
        input_leaf_copy_seconds=copies, forward_seconds=forwards, vjp_seconds=backwards,
        median_input_leaf_copy_seconds=statistics.median(copies),
        median_forward_seconds=statistics.median(forwards), median_vjp_seconds=statistics.median(backwards),
        median_forward_plus_vjp_seconds=statistics.median([f+b for f,b in zip(forwards,backwards)]),
        resident_allocated_bytes=residents, peak_total_allocated_bytes=peaks,
        peak_increment_over_resident_bytes=[p-r for p,r in zip(peaks,residents)] if device.type == "cuda" else None,
        actual_normalized_qmin=qmin, fixed_boundary_equal=boundary_equal)
    del last, function, module
    _sync(device)
    return result


def benchmark_case(anchor, reference, levels, args):
    latents = make_latents(levels, anchor.shape[0], anchor.dtype, anchor.device, args.seed)
    generator = torch.Generator(device=anchor.device).manual_seed(args.seed)
    upstream = torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype,
                           device=anchor.device) / anchor.numel()
    comparison = compare_chain(anchor, reference, levels, latents, upstream, args)
    module = _module(anchor, reference, levels, "existing", args)
    stages = trace_stages(anchor, module, latents)
    del module
    result = dict(status="success", side=anchor.shape[1], batch=anchor.shape[0],
                  levels=levels, stage_count=2*len(levels),
                  initial_normalized_qmin=normalized_corners(anchor, reference),
                  latent_parameter_count=sum(z.numel() for z in latents),
                  latent_shapes=[list(z.shape) for z in latents],
                  upstream_rms=float(upstream.square().mean().sqrt()), stage_diagnostics=stages)
    for backend in BACKENDS:
        measurements = measure_backend(anchor, reference, levels, latents, upstream, backend, args)
        result[backend] = dict(**comparison[backend], **measurements)
        print(json.dumps(dict(side=anchor.shape[1], backend=backend,
                             forward_ms=1000*measurements["median_forward_seconds"],
                             vjp_ms=1000*measurements["median_vjp_seconds"],
                             peak_bytes=measurements["peak_total_allocated_bytes"])), flush=True)
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.warmup < 3 or args.repeats < 10:
        raise ValueError("at least3 warmups and10 repeats required")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    tick = time.perf_counter()
    with np.load(args.anchor) as archive:
        base = torch.from_numpy(archive["vertices"].copy()).double()
        reference = torch.from_numpy(archive["boundary_reference"].copy()).double()
        diagonal = str(archive["interpolation"].item()).removeprefix("p1_")
    if diagonal != args.diagonal or base.shape[0] != 1 or base.shape[1] != 257 or base.shape[2] != 257:
        raise ValueError("declared257 batch1 P1 anchor with requested diagonal required")
    load_seconds = time.perf_counter() - tick
    payload = dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        anchor_load_seconds=load_seconds, results=[], valid_memory_measurement=True,
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        scope="same real lambda3 anchor exact P1 refinement; finest latent equals control; all initialY/latent FIRST derivatives; analyticx/y with default rounded validation EVERYstage; geometry only, no encoder/image objective",
        timing="same-process warm3 repeat10 minimum; constructors/source-buffer copies, leaf copies, CPU refinement/device transfer separate; checkpoint recomputation INCLUDED in VJP; no hooks in timed passes",
        memory="total allocated/resident including inputs/upstream/source cache, NOT reserved CUDA cache; detached hook payloads unique saved storage includes input views and is not additive to resident; all hooks and allocations collected after probe",
        caveat="nested fixed-source basis, uniform global scale perstage, no frequency gain from repeat updates; not universality/neural training/whole image pipeline; manual backends first-order only; exactties ALLprocessed and may enlarge storage, forward tie compaction can host-sync; otherGPU CPUlaunch interference possible")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for side in args.sizes:
        try:
            if side not in (257, 1025):
                raise ValueError("primary full-latent sizes257/1025 only")
            levels = [17,33,65,129,257] + ([513,1025] if side == 1025 else [])
            tick = time.perf_counter()
            factor = (side - 1) // 256
            anchor = refine_p1_vertices(base, factor, diagonal)
            ref = refine_p1_vertices(reference, factor, diagonal)
            refinement_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            anchor = anchor.to(device); ref = ref.to(device)
            _sync(device)
            transfer_seconds = time.perf_counter() - tick
            result = benchmark_case(anchor, ref, levels, args)
            result.update(cpu_integer_refinement_seconds=refinement_seconds,
                          device_transfer_seconds=transfer_seconds)
            del anchor, ref
        except Exception as error:
            # Preserve this seed/amplitude and failure stage; do not search for a
            # numerically convenient substitute. Other declared sizes still run.
            result = dict(status="failure", side=side, error_type=type(error).__name__, error=str(error))
            print(json.dumps(result), flush=True)
        payload["results"].append(result)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sizes",type=int,nargs="+",default=[257,1025]);p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2);p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=10);p.add_argument("--seed",type=int,default=20261001)
    p.add_argument("--amplitude",type=float,default=.005);p.add_argument("--minimum-jacobian",type=float,default=.001)
    p.add_argument("--theta",type=float,default=.95);p.add_argument("--diagonal",choices=("ac","bd"),default="ac")
    run(p.parse_args())


if __name__=="__main__":
    main()
