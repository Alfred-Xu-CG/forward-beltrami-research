from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from scipy.sparse import linalg as sparse_linalg


SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/phase5/route1_engineering_benchmark.py'


def _module():
    assert SCRIPT.is_file(), 'engineering harness is not implemented'
    spec = importlib.util.spec_from_file_location('phase5_engineering_harness', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(**kwargs):
    module = _module()
    defaults = dict(control=5, batch=1, layers=1, image_resolution=16,
                    backend='direct', dtype='float64', device='cpu', warmup=0, repeats=1)
    defaults.update(kwargs)
    return module.run_benchmark(module.BenchmarkConfig(**defaults))


def test_receipt_records_repository_commit(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, '_git_commit', lambda: 'a' * 40)
    receipt = module.run_benchmark(module.BenchmarkConfig(
        control=3,
        batch=1,
        layers=1,
        image_resolution=8,
        backend='directed_iterative',
        dtype='float64',
        device='cpu',
        warmup=0,
        repeats=1,
    ))
    assert receipt['status'] == 'ok', receipt
    assert receipt['environment']['commit'] == 'a' * 40


@pytest.mark.parametrize('batch,layers', [(1, 1), (2, 2), (2, 4)])
def test_cpu_tiny_receipt_counts_actual_factorizations_and_fields(batch, layers, monkeypatch):
    original = sparse_linalg.splu
    calls = []
    def recording(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(sparse_linalg, 'splu', recording)
    receipt = _run(batch=batch, layers=layers, repeats=2, warmup=1)
    assert receipt['status'] == 'ok', receipt
    assert len(calls) == 3 * batch * layers
    counts = receipt['solve_counts']
    assert counts['measured_primal_systems'] == counts['measured_adjoint_systems'] == 2*batch*layers
    assert counts['measured_forward_module_calls'] == counts['measured_backward_module_calls'] == 2*layers
    assert counts['warmup_primal_systems'] == counts['warmup_adjoint_systems'] == batch*layers
    assert receipt['parameter_count'] > 0
    for key in ['setup_seconds', 'prepare_seconds', 'mean_forward_seconds',
                'mean_backward_seconds', 'mean_end_to_end_seconds', 'dense_replay_forward_seconds']:
        assert receipt['timing'][key] >= 0
    assert receipt['throughput_samples_per_second'] > 0
    assert receipt['memory']['gpu'] is None
    assert 'linux_after_measurement' in receipt['memory']
    assert len(receipt['per_layer']) == layers
    for layer in receipt['per_layer']:
        assert len(layer['topology']) == batch
        assert all(t['global_injectivity_certificate'] for t in layer['topology'])
        assert layer['true_primal_relative_residual_max'] < 1e-12
        assert layer['solver_forward'] is None  # direct is not an iterative solver
    assert receipt['composite_original_mesh_p1_certificate'] is None
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize('backend', ['directed_iterative', 'symmetric'])
def test_iterative_fields_and_nontrivial_deterministic_inputs(backend):
    receipt = _run(backend=backend, layers=2, batch=2)
    assert receipt['status'] == 'ok', receipt
    assert receipt['input_summary']['boundary_logits_l2'] > 0
    assert receipt['input_summary']['latent_l2'] > 0
    assert all(layer['solver_forward'] is not None and layer['solver_adjoint'] is not None
               for layer in receipt['per_layer'])
    assert all(layer['solver_settings']['relative_tolerance'] > 0
               and layer['solver_settings']['maximum_iterations'] > 0
               for layer in receipt['per_layer'])
    assert max(layer['true_primal_relative_residual_max'] for layer in receipt['per_layer']) < 1e-8
    again = _run(backend=backend, layers=2, batch=2)
    assert again['input_summary'] == receipt['input_summary']
    assert again['output_summary'] == receipt['output_summary']


def test_symmetric_float32_receipt_matches_solver_default_tolerance():
    receipt = _run(backend='symmetric', dtype='float32')
    assert receipt['status'] == 'ok', receipt
    assert all(
        layer['solver_settings']['relative_tolerance'] == 1.0e-5
        for layer in receipt['per_layer']
    )


def test_direct_and_directed_iterative_use_same_latent_boundary_and_map():
    direct = _run(backend='direct')
    iterative = _run(backend='directed_iterative')
    assert direct['status'] == iterative['status'] == 'ok'
    assert direct['input_summary'] == iterative['input_summary']
    assert direct['output_summary']['dense_sum'] == pytest.approx(
        iterative['output_summary']['dense_sum'], abs=2e-8)


@pytest.mark.parametrize('kwargs', [dict(control=2), dict(repeats=0), dict(layers=3),
                                   dict(backend='direct', device='cuda'), dict(dtype='float16')])
def test_invalid_or_unavailable_configuration_is_structured_failure(kwargs):
    receipt = _run(**kwargs)
    assert receipt['status'] == 'failure'
    assert receipt['error']['type'] and receipt['error']['message']
    assert receipt['config']
    json.dumps(receipt, allow_nan=False)


def test_reference_receipt_counts_two_actual_legacy_factors_per_sample(monkeypatch):
    from qcopt.forward import tutte_directed_implicit as legacy
    original = legacy.factorized
    calls = []
    def recording(matrix):
        calls.append(1)
        return original(matrix)
    monkeypatch.setattr(legacy, 'factorized', recording)
    receipt = _run(backend='reference', batch=2, layers=2, warmup=1, repeats=2)
    assert receipt['status'] == 'ok', receipt
    assert len(calls) == 2 * 2 * 2 * 3
    assert receipt['solve_counts']['measured_primal_systems'] == 8
    assert receipt['solve_counts']['measured_adjoint_systems'] == 8
    assert receipt['solve_counts']['measured_factorizations'] == 16
    assert receipt['solve_counts']['warmup_factorizations'] == 8
    assert 'legacy' in receipt['semantics']['reference'].lower()
    assert 'sequential' in receipt['semantics']['reference'].lower()
    assert all(row['solver_settings']['factorizations_per_sample_forward'] == 2
               for row in receipt['per_layer'])
    assert all(row['solver_forward'] is None for row in receipt['per_layer'])
    assert all(row['true_primal_relative_residual_max'] < 1e-12 for row in receipt['per_layer'])
    direct = _run(backend='direct', batch=2, layers=2, warmup=0, repeats=1)
    assert direct['input_summary'] == receipt['input_summary']
    assert receipt['output_summary']['dense_sum'] == pytest.approx(
        direct['output_summary']['dense_sum'], abs=2e-10)
    json.dumps(receipt, allow_nan=False)


def test_reference_cuda_is_explicit_cpu_only_failure():
    receipt = _run(backend='reference', device='cuda')
    assert receipt['status'] == 'failure'
    assert 'CPU-only' in receipt['error']['message']


def test_reference_float32_receipt_audits_legacy_double_realized_weights():
    receipt = _run(backend='reference', dtype='float32')
    assert receipt['status'] == 'ok', receipt
    row = receipt['per_layer'][0]
    assert row['solver_settings']['probability_dtype'] == 'float64'
    assert row['true_primal_relative_residual_max'] < 1e-6
    assert all(item['global_injectivity_certificate'] for item in row['topology'])


def test_cli_emits_exactly_one_json_document():
    _module()
    result = subprocess.run([sys.executable, str(SCRIPT), '--control', '5', '--batch', '1',
                             '--layers', '1', '--image-resolution', '16', '--backend', 'direct',
                             '--dtype', 'float64', '--warmup', '0', '--repeats', '1'],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)['status'] == 'ok'


def test_bad_cli_is_structured_failure_not_usage_text():
    _module()
    result = subprocess.run([sys.executable, str(SCRIPT), '--layers', 'invalid'],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert json.loads(result.stdout)['status'] == 'failure'


def test_runtime_failure_preserves_stage_and_config():
    receipt = _run(strength=10000.0)
    assert receipt['status'] == 'failure'
    assert receipt['stage'] == 'measurement'
    assert receipt['config']['strength'] == 10000.0
    assert receipt['error']['type'] == 'ValueError'
    assert 'positive' in receipt['error']['message']
    json.dumps(receipt, allow_nan=False)


def test_linux_memory_fields_are_bytes_and_hwm_is_not_delta(monkeypatch):
    module = _module()
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.setattr(module.Path, 'read_text', lambda self: 'VmRSS:\t400 kB\nVmHWM:\t900 kB\n')
    assert module._linux_memory() == {'VmRSS_bytes': 400*1024, 'VmHWM_bytes': 900*1024}


def test_256_queries_float32_cpu_is_supported():
    receipt = _run(image_resolution=256, dtype='float32', layers=2)
    assert receipt['status'] == 'ok', receipt
    assert receipt['output_summary']['dense_shape'] == [1, 256, 256, 2]
    assert receipt['dense_replay_max_error'] == 0
