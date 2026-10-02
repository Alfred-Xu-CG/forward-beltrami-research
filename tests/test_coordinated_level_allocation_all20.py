"""Matched allocation changes no evidence, initializer or cohort membership."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pytest

from tests.test_coordinated_lung_all20 import assets, fake_match, fake_optimize, fake_dhr
from tests.test_coordinated_f2_floor_all20 import reserved_optimize


@pytest.fixture
def prepared(tmp_path):
    from tools.coordinated_lung_all20 import run as original
    from tools.coordinated_f2_floor_all20 import run as reserve
    old = assets(tmp_path)
    original(old, production=False, match_extractor=fake_match, optimizer=fake_optimize, dhr_runner=fake_dhr)
    corrected = argparse.Namespace(**vars(old), predictions=old.output, floor_safety_fraction=.95)
    corrected.output = tmp_path/'corrected'
    reserve(corrected, production=False, optimizer=reserved_optimize)
    return argparse.Namespace(canvas=old.canvas, affines_from=old.affines_from,
        predictions=corrected.output, uniform_output=tmp_path/'uniform',
        redistributed_output=tmp_path/'redistributed', comparison_output=tmp_path/'comparison.json',
        device='cpu', threads=1)


def mock_optimizer(config):
    result = fake_optimize(config)
    gradients = 2*sum(config.inner_steps_by_level)
    stages = 2*len(config.levels)
    result.update(gradient_steps=gradients, evaluations=gradients+stages,
        objective_evaluations=gradients+3*stages+2, joint_prior_backend='inductor',
        inner_steps_by_level=config.inner_steps_by_level, stages=[{}]*stages)
    config.output.with_suffix('.json').write_text(json.dumps(result))
    return result


def test_only_allocation_differs_and_production_budget(prepared):
    from tools.coordinated_level_allocation_all20 import make_configuration
    from tools.coordinated_lung_all20 import pairs
    pair = pairs(prepared)[0]
    configs = [make_configuration(pair, prepared, arm, prepared.predictions/'raw.json')
               for arm in ('uniform', 'redistributed')]
    assert configs[0].inner_steps_by_level == [30]*5
    assert configs[1].inner_steps_by_level == [30,20,20,30,50]
    for cfg in configs:
        assert cfg.joint_prior_backend == 'inductor'
        assert cfg.method == 'analytic' and cfg.cycles == 1
        assert 2*sum(cfg.inner_steps_by_level) == 300
        assert 2*len(cfg.levels) == 10
    left, right = [dict(vars(cfg)) for cfg in configs]
    for item in (left, right):
        item.pop('output'); item.pop('inner_steps_by_level')
    assert left == right


def test_paired_order_archives_relative_unchanged_and_new_timings_only(prepared):
    from tools.coordinated_level_allocation_all20 import run
    from tools.coordinated_lung_all20_score import _load_predictions
    before = {p:p.read_bytes() for p in prepared.predictions.rglob('*') if p.is_file()}
    calls = []
    def optimize(config):
        calls.append((config.output.parent.name, config.output.name))
        return mock_optimizer(config)
    result = run(prepared, production=False, optimizer=optimize)
    assert len(calls) == 40 and result['prediction_complete']
    assert calls[:4] == [('uniform','he_to_cc10_analytic.npz'),
        ('redistributed','he_to_cc10_analytic.npz'), ('redistributed','he_to_cd31_analytic.npz'),
        ('uniform','he_to_cd31_analytic.npz')]
    for arm in ('uniform', 'redistributed'):
        folder = getattr(prepared, arm+'_output')
        report, _ = _load_predictions(folder)
        assert len(report['rows']) == 20 and report['annotations_read'] is False
        assert report['recomputed_methods'] == ['analytic']
        assert report['old_timing_included_in_new_total'] is False
        old = json.loads((prepared.predictions/'predictions.json').read_text())
        for row, source in zip(report['rows'], old['rows'], strict=True):
            analytic = row['methods']['analytic']
            assert analytic['status'] == 'ok' and analytic['budget_complete']
            assert analytic['gradient_steps'] == 8 and analytic['evaluations'] == 12
            assert analytic['objective_evaluations'] == 22
            for method in ('f2','dhr'):
                archived = row['methods'][method]
                assert archived['recomputed'] is False
                for key in ('output','report','output_directory','field','postprocessing_params','configuration'):
                    if isinstance(source['methods'][method].get(key),str):
                        assert (folder/archived[key]).resolve() == (prepared.predictions/source['methods'][method][key]).resolve()
            assert (folder/row['raw_matches']['path']).resolve() == (prepared.predictions/source['raw_matches']['path']).resolve()
        assert report['recomputed_analytic_optimize_seconds_total'] == pytest.approx(.003*20)
    assert result['hidden_warmup_calls'] == 0 and result['call_order'][0]['first_optimizer_call']
    assert all(p.read_bytes() == content for p,content in before.items())


@pytest.mark.parametrize('failure', ['exception','certificate','strict_floor','budget'])
def test_failed_arm_retains20_and_other_arm(prepared, failure, monkeypatch):
    from tools import coordinated_level_allocation_all20 as runner
    def optimize(config):
        if config.output.parent.name == 'redistributed' and config.output.stem == 'he_to_cc10_analytic':
            if failure == 'exception': raise RuntimeError('deliberate failed call')
            result = mock_optimizer(config)
            if failure == 'certificate': result['saved_binary_certificate']['valid'] = False
            if failure == 'budget': result['gradient_steps'] = 7
            return result
        return mock_optimizer(config)
    if failure == 'strict_floor':
        real = runner._actual_ratio
        monkeypatch.setattr(runner,'_actual_ratio',lambda path: .001 if path.parent.name=='redistributed' and path.stem=='he_to_cc10_analytic' else real(path))
    result = runner.run(prepared, production=False, optimizer=optimize)
    assert result['prediction_complete']
    left = json.loads((prepared.uniform_output/'predictions.json').read_text())
    right = json.loads((prepared.redistributed_output/'predictions.json').read_text())
    assert len(left['rows']) == len(right['rows']) == 20
    assert left['method_success_counts']['analytic'] == 20
    assert right['method_success_counts']['analytic'] == 19
    assert right['rows'][0]['methods']['analytic']['status'] == 'failed'


@pytest.mark.parametrize('failure', ['incomplete','wrong_reserve','duplicate','old_pending'])
def test_invalid_source_before_calls_or_new_outputs(prepared, failure):
    from tools.coordinated_level_allocation_all20 import run
    path = prepared.predictions/'predictions.json'
    source = json.loads(path.read_text())
    if failure == 'incomplete': source['prediction_complete'] = False
    elif failure == 'wrong_reserve': source['floor_safety_fraction'] = 1.
    elif failure == 'duplicate': source['rows'][-1] = source['rows'][0]
    else: source['rows'][0]['methods']['dhr']['status'] = 'pending'
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError): run(prepared, production=False, optimizer=lambda *a:pytest.fail('called'))
    assert not prepared.uniform_output.exists() and not prepared.redistributed_output.exists()


def test_no_overwrite_shared_output_or_annotation_import(prepared):
    import ast
    from tools import coordinated_level_allocation_all20 as runner
    prepared.redistributed_output = prepared.uniform_output
    with pytest.raises(ValueError): runner.run(prepared, production=False, optimizer=mock_optimizer)
    imports = [node.module or '' for node in ast.walk(ast.parse(Path(runner.__file__).read_text())) if isinstance(node,ast.ImportFrom)]
    assert all('score' not in name and 'landmark' not in name for name in imports)


def test_first_actual_optimizer_call_after_failed_input(prepared, monkeypatch):
    from tools import coordinated_level_allocation_all20 as runner
    check = runner._check_input
    visits = []
    def fail_first(*args):
        visits.append(1)
        if len(visits)==1: raise ValueError('deliberate first input failure')
        return check(*args)
    monkeypatch.setattr(runner,'_check_input',fail_first)
    result = runner.run(prepared,production=False,optimizer=mock_optimizer)
    assert result['call_order'][0]['first_optimizer_call'] is False
    assert result['call_order'][1]['first_optimizer_call'] is True
    assert result['actual_optimizer_calls']==39


@pytest.mark.parametrize('where',['source','arm'])
def test_comparison_never_writes_into_source_or_arm_manifest(prepared,where):
    from tools.coordinated_level_allocation_all20 import run
    prepared.comparison_output=(prepared.predictions if where=='source' else prepared.uniform_output)/'predictions.json'
    with pytest.raises((ValueError,FileExistsError)):
        run(prepared,production=False,optimizer=lambda *a:pytest.fail('called'))
