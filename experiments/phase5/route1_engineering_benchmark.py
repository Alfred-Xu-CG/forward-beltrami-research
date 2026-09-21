"""One-process, one-JSON Route I engineering measurement (not optimization).

Example: PYTHONPATH=src python experiments/phase5/route1_engineering_benchmark.py
  --control 17 --batch 4 --layers 2 --image-resolution 256
  --backend directed_iterative --dtype float32 --device cuda

Forward includes boundary generation, certified solves and coordinate composition;
end-to-end means forward + scalar surrogate loss + backward, NOT registration or
an optimizer step. Dense-only time is a separately labelled replay, not a
subtracted/additive estimate. Use a fresh process per configuration for Linux
process-lifetime VmHWM comparisons. No claims of official KLU replication.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
import scipy
import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.metrics import compute_p1_map_metrics
from qcopt.neural_bijection.tutte.composition import SquareTutteComposition, compose_control_maps
from qcopt.neural_bijection.tutte.decoder import TutteRectangleDecoder
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer
from qcopt.neural_bijection.tutte.reference import LegacyReferenceTutteLayer
from qcopt.neural_bijection.tutte.iterative import MatrixFreeDirectedTutteLayer
from qcopt.neural_bijection.tutte.symmetric import MatrixFreeSymmetricTutteLayer


REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkConfig:
    control: int = 17
    batch: int = 1
    layers: int = 1
    image_resolution: int = 256
    backend: str = 'directed_iterative'
    dtype: str = 'float32'
    device: str = 'cpu'
    warmup: int = 1
    repeats: int = 3
    seed: int = 1701
    strength: float = 0.15
    threads: int = 1


def _git_commit() -> str | None:
    """Return ordinary Git provenance when this script runs inside a checkout."""
    try:
        result = subprocess.run(
            ('git', '-C', str(REPOSITORY), 'rev-parse', 'HEAD'),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().lower()
    if result.returncode != 0 or len(value) != 40 or any(ch not in '0123456789abcdef' for ch in value):
        return None
    return value


def _plain(value: Any) -> Any:
    """JSON-safe snapshots; unknown/nonfinite measurements are null, never zero."""
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _linux_memory() -> dict | None:
    """Linux kB fields are KiB; VmHWM is process-lifetime high water, not delta."""
    if not sys.platform.startswith('linux'):
        return None
    try:
        lines = Path('/proc/self/status').read_text().splitlines()
    except OSError:
        return None
    fields = {}
    for line in lines:
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, amount, unit = line.split()
            if unit == 'kB':
                fields[key[:-1] + '_bytes'] = int(amount) * 1024
    return fields or None


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _seeded(shape, seed: int, strength: float, dtype, device) -> torch.nn.Parameter:
    # Generate on CPU in double first: directed backend/dtype/device comparisons
    # start from the same deterministic underlying numbers, then cast explicitly.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    value = strength * torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.nn.Parameter(value.to(device=device, dtype=dtype))


def _true_residual(solver, latent, control) -> dict:
    """Independent CPU double accumulation from returned coordinates and weights.

    Weights are re-evaluated in their native device/dtype, then promoted. This
    measures the returned precision, rather than only a hidden double solution.
    It is outside timed forward/backward and does not reuse solver matvec code.
    """
    system = solver.system
    y = control.detach().double().cpu().numpy()
    batch = len(y)
    residual = np.zeros((batch, system.n_rows, 2))
    rhs = np.zeros_like(residual)
    with torch.no_grad():
        if isinstance(solver, MatrixFreeSymmetricTutteLayer):
            c = solver.conductances(latent)
            c = c / c.amax(dim=-1, keepdim=True)
            c = c.double().cpu().numpy()
            index = {int(vertex): row for row, vertex in enumerate(system.interior)}
            for edge, (first, second) in enumerate(solver.active_edges):
                for i, j in ((int(first), int(second)), (int(second), int(first))):
                    if i in index:
                        row = index[i]
                        residual[:, row] += c[:, edge, None] * (y[:, i] - y[:, j])
                        if j not in index:
                            rhs[:, row] += c[:, edge, None] * y[:, j]
            contrast = float(np.max(np.max(c, axis=1) / np.min(c, axis=1)))
            conditioning = {'normalized_conductance_contrast': contrast}
        else:
            if isinstance(solver, LegacyReferenceTutteLayer):
                # Legacy realizes its probabilities in NumPy float64 even when
                # the supplied logits and returned coordinates are float32.
                p = np.stack([system._probabilities(sample)
                              for sample in latent.detach().double().cpu().numpy()])
            else:
                mask = torch.tensor(system.valid_mask.copy(), device=latent.device)
                p = torch.softmax(latent.masked_fill(~mask, -torch.inf), dim=-1).double().cpu().numpy()
            residual[:] = y[:, system.interior]
            for row, slot in zip(*np.nonzero(system.valid_mask)):
                neighbor = system.neighbors[row, slot]
                boundary = system.neighbor_is_boundary[row, slot]
                vertex = system.loop[neighbor] if boundary else system.interior[neighbor]
                weighted = p[:, row, slot, None] * y[:, vertex]
                residual[:, row] -= weighted
                if boundary:
                    rhs[:, row] += weighted
            conditioning = {'minimum_supported_probability': float(p[:, system.valid_mask].min()),
                            'maximum_supported_probability': float(p[:, system.valid_mask].max())}
    # Inputs here are bounded square coordinates and moderate positive weights.
    absolute = np.linalg.norm(residual, axis=1)
    denominator = np.linalg.norm(rhs, axis=1)
    relative = np.divide(absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0)
    if np.any((denominator == 0) & (absolute != 0)):
        raise RuntimeError('zero-RHS residual is nonzero')
    return dict(true_primal_absolute_residual=absolute.tolist(),
                true_primal_relative_residual=relative.tolist(),
                true_primal_relative_residual_max=float(relative.max()),
                conditioning_proxy=conditioning)


def run_benchmark(config: BenchmarkConfig) -> dict:
    """Run one configuration; failures preserve config, stage and partial data."""
    started = perf_counter()
    report = dict(status='failure', question='Route I forward/backward/control-query scaling',
                  config=asdict(config), stage='validation', samples=[], per_layer=[],
                  timing={}, memory={'linux_before_setup': _linux_memory(), 'gpu': None},
                  composite_original_mesh_p1_certificate=None)
    handles = []
    old_threads = torch.get_num_threads()
    counts = {'forward': 0, 'backward': 0}
    try:
        if config.backend not in ('reference', 'direct', 'directed_iterative', 'symmetric'):
            raise ValueError('unknown backend')
        if config.control < 3 or config.batch < 1 or config.layers not in (1, 2, 4):
            raise ValueError('control>=3, batch>=1, and layers in 1/2/4 are required')
        if config.image_resolution < 2 or config.warmup < 0 or config.repeats < 1 or config.threads < 1:
            raise ValueError('invalid image resolution, warmup/repeats, or thread count')
        if config.dtype not in ('float32', 'float64'):
            raise ValueError('dtype must be float32 or float64')
        if not math.isfinite(config.strength) or config.strength <= 0:
            raise ValueError('strength must be finite and positive')
        device = torch.device(config.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('device must be CPU or CUDA')
        if config.backend in ('reference', 'direct') and device.type != 'cpu':
            raise ValueError(f'{config.backend} backend is CPU-only')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        dtype = getattr(torch, config.dtype)
        torch.set_num_threads(config.threads)
        report['environment'] = dict(host=platform.node(), platform=platform.platform(),
            python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            scipy=scipy.__version__, cuda_build=torch.version.cuda, cpu=platform.processor(),
            commit=_git_commit(),
            torch_threads=torch.get_num_threads(), omp_num_threads=os.getenv('OMP_NUM_THREADS'),
            mkl_num_threads=os.getenv('MKL_NUM_THREADS'),
            gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
        report['stage'] = 'setup'
        setup_start = perf_counter()
        mesh = structured_rectangle(config.control - 1, config.control - 1)
        decoders, latents, boundaries = [], [], []
        for layer in range(config.layers):
            solver = {'reference': LegacyReferenceTutteLayer,
                      'direct': DirectTutteLayer, 'directed_iterative': MatrixFreeDirectedTutteLayer,
                      'symmetric': MatrixFreeSymmetricTutteLayer}[config.backend](mesh)
            decoder = TutteRectangleDecoder(mesh, solver, image_height=config.image_resolution,
                                            image_width=config.image_resolution)
            decoders.append(decoder)
            shape = ((config.batch, solver.n_conductances) if config.backend == 'symmetric'
                     else (config.batch, solver.system.n_rows, solver.system.max_degree))
            latents.append(_seeded(shape, config.seed + 2*layer, config.strength, dtype, device))
            boundaries.append(_seeded((config.batch, decoder.boundary.n_segments),
                                     config.seed + 2*layer + 1, config.strength, dtype, device))
        module = SquareTutteComposition(decoders).to(device=device)
        _sync(device)
        report['timing']['setup_seconds'] = perf_counter() - setup_start
        prepare_start = perf_counter()
        module.prepare(device=device, dtype=dtype)
        _sync(device)
        report['timing']['prepare_seconds'] = perf_counter() - prepare_start
        parameters = latents + boundaries
        report['parameter_count'] = sum(p.numel() for p in parameters)
        report['mesh'] = dict(vertices=mesh.n_vertices, interior=decoders[0].solver.system.n_rows,
                              boundary=len(mesh.boundary_loops[0]), faces=len(mesh.faces))
        report['input_summary'] = dict(latent_l2=float(torch.sqrt(sum(p.detach().double().square().sum() for p in latents)).cpu()),
            boundary_logits_l2=float(torch.sqrt(sum(p.detach().double().square().sum() for p in boundaries)).cpu()),
            fixed_height=1.0)

        def forward_count(_module, _args, _output):
            counts['forward'] += 1

        def backward_count(_module, _gin, _gout):
            counts['backward'] += 1

        for decoder in decoders:
            handles.append(decoder.solver.register_forward_hook(forward_count))
            handles.append(decoder.solver.register_full_backward_hook(backward_count))

        def step():
            for parameter in parameters:
                parameter.grad = None
            _sync(device)
            start = perf_counter()
            result = module(latents, boundaries)
            _sync(device)
            forward_end = perf_counter()
            loss = result.dense.square().mean()
            _sync(device)
            backward_start = perf_counter()
            loss.backward()
            _sync(device)
            end = perf_counter()
            return result, dict(forward_seconds=forward_end-start,
                loss_seconds=backward_start-forward_end, backward_seconds=end-backward_start,
                end_to_end_seconds=end-start, objective=float(loss.detach().cpu()))

        report['stage'] = 'warmup'
        for _ in range(config.warmup):
            warm_result, _ = step()
            del warm_result
        warm_counts = counts.copy()
        counts.update(forward=0, backward=0)
        for parameter in parameters:
            parameter.grad = None
        _sync(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            report['memory']['gpu'] = dict(baseline_allocated_bytes=torch.cuda.memory_allocated(device),
                baseline_reserved_bytes=torch.cuda.memory_reserved(device))
        report['memory']['linux_before_measurement'] = _linux_memory()
        report['stage'] = 'measurement'
        for repeat in range(config.repeats):
            result, sample = step()
            report['samples'].append(sample)
            if repeat + 1 < config.repeats:
                del result
        report['memory']['linux_after_measurement'] = _linux_memory()
        if device.type == 'cuda':
            report['memory']['gpu'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        report['solve_counts'] = dict(measured_forward_module_calls=counts['forward'],
            measured_backward_module_calls=counts['backward'],
            measured_primal_systems=counts['forward']*config.batch,
            measured_adjoint_systems=counts['backward']*config.batch,
            warmup_primal_systems=warm_counts['forward']*config.batch,
            warmup_adjoint_systems=warm_counts['backward']*config.batch)
        if config.backend in ('reference', 'direct'):
            factors_per_sample = 2 if config.backend == 'reference' else 1
            report['solve_counts'].update(
                measured_factorizations=counts['forward']*config.batch*factors_per_sample,
                warmup_factorizations=warm_counts['forward']*config.batch*factors_per_sample,
                factorization_count_basis='completed forward hooks times batch times backend factorization contract; call counts independently tested')
        for field in ('forward_seconds', 'loss_seconds', 'backward_seconds', 'end_to_end_seconds'):
            report['timing']['mean_' + field] = statistics.mean(row[field] for row in report['samples'])
            report['timing']['median_' + field] = statistics.median(row[field] for row in report['samples'])
        report['throughput_samples_per_second'] = config.batch / report['timing']['mean_end_to_end_seconds']
        report['gradients_finite'] = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
        if not report['gradients_finite']:
            raise RuntimeError('missing or nonfinite parameter gradients')
        report['stage'] = 'audit'
        audit_start = perf_counter()
        for layer, (decoder, latent, control) in enumerate(zip(decoders, latents, result.controls)):
            solver = decoder.solver
            diagnostics = getattr(solver, 'last_diagnostics', None)
            row = dict(layer=layer, solver_forward=_plain(diagnostics.forward) if diagnostics else
                       _plain(getattr(solver, 'last_forward_stats', None)),
                       solver_adjoint=_plain(diagnostics.adjoint) if diagnostics else
                       _plain(getattr(solver, 'last_adjoint_stats', None)), topology=[])
            if isinstance(solver, MatrixFreeDirectedTutteLayer):
                row['solver_settings'] = dict(algorithm='BiCGStab primary; actual method in solver diagnostics',
                    relative_tolerance=solver.rtol if solver.rtol is not None else
                    (1e-5 if dtype == torch.float32 else 1e-10),
                    absolute_tolerance=solver.atol, maximum_iterations=solver.max_iter,
                    cuda_float32_stationary_fallback_maximum_iterations=20*solver.max_iter)
            elif isinstance(solver, MatrixFreeSymmetricTutteLayer):
                row['solver_settings'] = dict(algorithm='CG',
                    relative_tolerance=solver.relative_tolerance if solver.relative_tolerance is not None else
                    (1e-5 if dtype == torch.float32 else 1e-11),
                    absolute_tolerance=solver.absolute_tolerance, maximum_iterations=solver.max_iterations)
            elif isinstance(solver, LegacyReferenceTutteLayer):
                row['solver_settings'] = dict(
                    algorithm='legacy SciPy factorized(A) and factorized(A.T)',
                    internal_dtype='float64', probability_dtype='float64',
                    factorizations_per_sample_forward=2, batch_execution='sequential CPU',
                    relative_tolerance=None, absolute_tolerance=None, maximum_iterations=None)
            else:
                row['solver_settings'] = dict(algorithm='SuperLU', internal_dtype='float64',
                    factorizations_per_sample_forward=1, batch_execution='sequential CPU',
                    relative_tolerance=None, absolute_tolerance=None, maximum_iterations=None)
            row.update(_true_residual(solver, latent, control))
            for sample in control.detach().double().cpu().numpy():
                metrics = compute_p1_map_metrics(mesh, sample)
                row['topology'].append({key: value for key, value in asdict(metrics).items()
                    if key in ('flip_count', 'minimum_signed_area', 'minimum_area_ratio',
                               'boundary_order_min_gap', 'global_injectivity_certificate')})
            report['per_layer'].append(row)
            if not all(item['global_injectivity_certificate'] for item in row['topology']):
                raise RuntimeError('independent per-layer topology audit failed')
        report['timing']['post_measurement_audit_seconds'] = perf_counter() - audit_start
        report['output_summary'] = dict(dense_shape=list(result.dense.shape),
            dense_sum=float(result.dense.detach().double().sum().cpu()),
            dense_squared_mean=float(result.dense.detach().double().square().mean().cpu()))
        report['stage'] = 'dense_replay'
        controls = [control.detach().requires_grad_() for control in result.controls]
        _sync(device)
        replay_start = perf_counter()
        replay = compose_control_maps([decoder.query_table for decoder in decoders], controls)
        _sync(device)
        report['timing']['dense_replay_forward_seconds'] = perf_counter() - replay_start
        report['dense_replay_max_error'] = float((replay-result.dense.detach()).abs().max().detach().cpu())
        report['semantics'] = dict(
            timing='synchronized wall clock; end_to_end=composition forward+surrogate loss+backward; no optimizer/image sampler',
            dense='separate autograd-enabled coordinate replay, no solver, excluded from primary timing and GPU peak; not additive',
            setup='mesh/solver/boundary/fixed-table construction and parameter placement; imports excluded; prepare separately timed',
            memory='Linux VmHWM is process-lifetime including earlier setup/warmup; GPU peaks cover measured F/B with setup baseline, before post-audit/dense replay',
            counts='module hooks counted; systems=module_calls*explicit batch, each x/y multi-RHS counts once; excludes dense replay',
            topology='per-layer original P1 certificate on final identical-parameter repeat only; not composite resampled P1',
            residual='independent native-weight/returned-coordinate CPU-double accumulation; adjoint residual from solver diagnostics when available',
            iterations='diagnostic method/primary_failure are preserved; fallback iterations are not total work including the failed primary attempt',
            family='symmetric changes the matrix/parameter family; directed backends share seeded logits and boundaries',
            direct='CPU SuperLU internally double, one factor per sample per forward; no cross-forward symbolic cache',
            reference='legacy CPU-only sequential adapter; NumPy float64 weights and solves; factorized(A) and factorized(A.T) per sample forward, saved transpose factor in backward; no cross-forward cache or official KLU replication; realized-probability and returned-dtype positive-face screens included',
            gradients='finite gradients checked; no finite-difference accuracy claim from this performance run')
        report.update(status='ok', stage='complete')
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        if hasattr(error, 'report'):
            report['error']['solver_report'] = _plain(error.report)
        report['memory']['linux_on_failure'] = _linux_memory()
        report['partial_module_calls'] = counts.copy()
    finally:
        for handle in handles:
            handle.remove()
        torch.set_num_threads(old_threads)
        report['harness_wall_seconds'] = perf_counter() - started
    return _plain(report)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None) -> int:
    try:
        parser = _Parser(description=__doc__)
        for name in ('control', 'batch', 'layers', 'image_resolution', 'warmup', 'repeats', 'seed', 'threads'):
            parser.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(BenchmarkConfig(), name))
        for name in ('backend', 'dtype', 'device'):
            parser.add_argument('--' + name, default=getattr(BenchmarkConfig(), name))
        parser.add_argument('--strength', type=float, default=BenchmarkConfig().strength)
        report = run_benchmark(BenchmarkConfig(**vars(parser.parse_args(argv))))
    except Exception as error:
        report = dict(status='failure', stage='arguments', error=dict(type=type(error).__name__, message=str(error)))
    print(json.dumps(_plain(report), allow_nan=False))
    return 0 if report['status'] == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())
