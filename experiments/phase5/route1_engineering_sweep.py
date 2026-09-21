"""Resumable Route I Cartesian sweep: one JSONL and one compact CSV.

Example (use separate files for concurrently running CPU/GPU submatrices):
  PYTHONPATH=src python experiments/phase5/route1_engineering_sweep.py
    --backends direct symmetric --controls 17 25 --batches 1 4 --layers 1 2
    --resolutions 256 512 --seeds 1701 --device cpu --jsonl results.jsonl

Each configuration runs in a fresh process so Linux process-lifetime VmHWM
does not carry across configurations. Append/flush/fsync precedes the next job.
Resume skips all existing receipts (including failures), using EVERY config
field; it never repairs corrupt/truncated lines or silently drops duplicates.
Use one writer per JSONL. Existing environment/commit provenance stays in the
receipt; absent provenance is not inferred from the summary-generating host.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import importlib.util
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys


_HARNESS = Path(__file__).with_name('route1_engineering_benchmark.py')
_spec = importlib.util.spec_from_file_location('phase5_sweep_harness', _HARNESS)
_harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _harness
_spec.loader.exec_module(_harness)
BenchmarkConfig = _harness.BenchmarkConfig


def config_key(config) -> str:
    """Canonical complete config; missing/extra fields and wrong types fail."""
    values = asdict(config) if isinstance(config, BenchmarkConfig) else config
    defaults = asdict(BenchmarkConfig())
    if not isinstance(values, dict) or values.keys() != defaults.keys():
        raise ValueError('receipt must contain the complete BenchmarkConfig')
    values = values.copy()
    for name, default in defaults.items():
        value = values[name]
        if isinstance(default, float):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'invalid config field {name}')
            values[name] = float(value)
        elif type(value) is not type(default):
            raise ValueError(f'invalid config field {name}')
    return json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate JSON key: {key}')
        value[key] = item
    return value


def _loads(text):
    def invalid(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    def finite_float(value):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else invalid(value)
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid,
                      parse_float=finite_float)


def _receipt_key(receipt):
    if not isinstance(receipt, dict) or receipt.get('status') not in ('ok', 'failure', 'not_implemented'):
        raise ValueError('invalid benchmark receipt status')
    return config_key(receipt.get('config'))


def _read_receipts(path):
    receipts, keys = [], set()
    if path.exists():
        with path.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                try:
                    if not line.endswith('\n'):
                        raise ValueError('incomplete final line (missing newline)')
                    receipt = _loads(line)
                    key = _receipt_key(receipt)
                    if key in keys:
                        raise ValueError('duplicate config')
                except (ValueError, TypeError) as error:
                    raise ValueError(f'{path}: line {number}: {error}') from error
                keys.add(key)
                receipts.append(receipt)
    return receipts, keys


def _fresh_process(config):
    """Child imports and calls the harness public config/run API, not its CLI."""
    result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
                            input=json.dumps(asdict(config), allow_nan=False),
                            capture_output=True, text=True)
    if result.returncode:
        return dict(config=asdict(config), status='failure', stage='worker',
                    error=dict(type='WorkerProcessError',
                               message=f'exit {result.returncode}: {result.stderr[-4000:]}'))
    return _loads(result.stdout)


def _numbers(value):
    if isinstance(value, list):
        return [item for part in value for item in _numbers(part)]
    return [value] if type(value) in (int, float) and math.isfinite(value) else []


def summary_row(receipt):
    """Unknown measurements remain blank, never fabricated zero or success."""
    layers = receipt.get('per_layer', [])
    topology = [item for layer in layers for item in layer.get('topology', [])]
    memory = receipt.get('memory', {})
    gpu = memory.get('gpu') or {}
    linux = memory.get('linux_after_measurement') or memory.get('linux_on_failure') or {}
    environment = receipt.get('environment', {})
    row = dict(receipt['config'], status=receipt['status'], stage=receipt.get('stage'),
               host=environment.get('host'), commit=receipt.get('commit', environment.get('commit')),
               error=(receipt.get('error') or {}).get('message'),
               gpu_peak_allocated_bytes=gpu.get('peak_allocated_bytes'),
               gpu_peak_reserved_bytes=gpu.get('peak_reserved_bytes'),
               process_lifetime_hwm_bytes=linux.get('VmHWM_bytes'),
               throughput_samples_per_second=receipt.get('throughput_samples_per_second'),
               gradients_finite=receipt.get('gradients_finite'))
    for key in ('mean_forward_seconds', 'mean_backward_seconds', 'mean_end_to_end_seconds'):
        row[key] = receipt.get('timing', {}).get(key)
    residuals = _numbers([layer.get('true_primal_relative_residual_max') for layer in layers])
    row['true_primal_relative_residual_max'] = max(residuals, default=None)
    row['flip_count_sum'] = sum(item['flip_count'] for item in topology) if topology else None
    row['minimum_signed_area'] = min((item['minimum_signed_area'] for item in topology), default=None)
    expected_layers = receipt['config']['layers']
    expected_batch = receipt['config']['batch']
    complete = expected_layers > 0 and expected_batch > 0 and bool(topology) and (
        len(layers) == expected_layers
        and all(len(layer.get('topology', [])) == expected_batch for layer in layers)
    )
    row['all_layer_sample_topology_certified'] = all(
        item['global_injectivity_certificate'] for item in topology) if complete else None
    for side in ('forward', 'adjoint'):
        reports = [layer.get('solver_' + side) for layer in layers]
        reports = [report for report in reports if report]
        row[side + '_methods'] = '|'.join(sorted({report.get('method', 'unspecified') for report in reports})) or None
        row['fallback_' + side + '_layers'] = sum(report.get('method') == 'stationary_fallback' for report in reports) if reports else None
        row['max_' + side + '_iterations'] = max(_numbers([report.get('iterations') for report in reports]), default=None)
        row['max_' + side + '_reported_relative_residual'] = max(_numbers([report.get('relative_residual') for report in reports]), default=None)
    row['primary_failures'] = '|'.join(sorted({report['primary_failure'] for layer in layers
        for report in (layer.get('solver_forward'), layer.get('solver_adjoint')) if report and report.get('primary_failure')})) or None
    for key in ('measured_primal_systems', 'measured_adjoint_systems'):
        row[key] = receipt.get('solve_counts', {}).get(key)
    return row


def _write_summary(receipts, path):
    rows = [summary_row(receipt) for receipt in receipts]
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(configs, jsonl, csv_path=None, *, resume=False, runner=None):
    """Append receipts in order; interruption leaves completed receipts intact.

    runner is an optional callable for small orchestration tests. Production
    default always isolates configurations in fresh subprocesses.
    """
    path = Path(jsonl)
    csv_path = Path(csv_path) if csv_path is not None else path.with_suffix('.csv')
    if path.resolve() == csv_path.resolve():
        raise ValueError('JSONL and CSV paths must differ')
    configs = list(configs)
    keys = [config_key(config) for config in configs]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate requested config')
    if path.exists() and not resume:
        raise FileExistsError(f'{path} exists; use --resume')
    receipts, seen = _read_receipts(path)
    runner = _fresh_process if runner is None else runner
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a' if resume else 'x', encoding='utf-8', newline='\n') as stream:
        for config, key in zip(configs, keys):
            if key in seen:
                continue
            receipt = runner(config)
            if _receipt_key(receipt) != key:
                raise ValueError('returned receipt config differs from requested config')
            line = json.dumps(receipt, allow_nan=False, separators=(',', ':'))
            stream.write(line + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            receipts.append(receipt)
            seen.add(key)
    _write_summary(receipts, csv_path)
    return receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    for plural, field in (('backends','backend'),('controls','control'),('batches','batch'),
                          ('layers','layers'),('resolutions','image_resolution'),('seeds','seed')):
        default = getattr(BenchmarkConfig(), field)
        parser.add_argument('--'+plural, nargs='+', type=type(default), default=[default])
    for field in ('dtype','device','warmup','repeats','strength','threads'):
        default = getattr(BenchmarkConfig(),field)
        parser.add_argument('--'+field, type=type(default), default=default)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        config = BenchmarkConfig(**_loads(sys.stdin.read()))
        print(json.dumps(_harness.run_benchmark(config), allow_nan=False))
        return 0
    if args.jsonl is None:
        parser.error('--jsonl is required')
    fixed = {name:getattr(args,name) for name in ('dtype','device','warmup','repeats','strength','threads')}
    configs = [BenchmarkConfig(**fixed, backend=backend, control=control, batch=batch,
                               layers=layers, image_resolution=resolution, seed=seed)
               for backend, control, batch, layers, resolution, seed in product(
                   args.backends,args.controls,args.batches,args.layers,args.resolutions,args.seeds)]
    try:
        receipts = run_sweep(configs,args.jsonl,args.csv,resume=args.resume)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    requested = {config_key(config) for config in configs}
    return 0 if all(row['status']=='ok' for row in receipts if config_key(row['config']) in requested) else 2


if __name__ == '__main__':
    raise SystemExit(main())
