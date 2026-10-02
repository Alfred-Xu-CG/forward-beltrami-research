"""Only F2 is rerun; frozen matcher/A/DHR references retain all20 directions."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pytest

from tests.test_coordinated_lung_all20 import assets,fake_match,fake_optimize,fake_dhr


@pytest.fixture
def frozen(tmp_path):
    from tools.coordinated_lung_all20 import run
    original=assets(tmp_path)
    run(original,production=False,match_extractor=fake_match,optimizer=fake_optimize,dhr_runner=fake_dhr)
    return argparse.Namespace(canvas=original.canvas,affines_from=original.affines_from,
        predictions=original.output,output=tmp_path/'corrected',device='cpu',threads=1,
        floor_safety_fraction=.95)


def reserved_optimize(config):
    result=fake_optimize(config)
    result['f2_floor_safety_fraction']=config.f2_floor_safety_fraction
    config.output.with_suffix('.json').write_text(json.dumps(result))
    return result


def test_recipe_matches_original_except_explicit_reserve_and_archive_paths(frozen):
    from tools.coordinated_f2_floor_all20 import make_configuration
    from tools.coordinated_lung_all20 import make_configuration as original,pairs
    pair=pairs(frozen)[0]
    left=original(pair,'f2',frozen);right=make_configuration(pair,frozen,frozen.predictions/'he_to_cc10_raw_matches.json')
    data=dict(vars(right));data.pop('f2_floor_safety_fraction');data['matches']=left.matches
    assert data==vars(left)
    assert right.f2_floor_safety_fraction==.95
    assert right.cycles*right.inner_steps*len(right.levels)==300


def test_all20_only_f2_calls_and_relative_archive_paths_source_unchanged(frozen):
    from tools.coordinated_f2_floor_all20 import run
    from tools.coordinated_lung_all20_score import _load_predictions
    before={p:p.read_bytes() for p in frozen.predictions.rglob('*') if p.is_file()}
    calls=[]
    def optimizer(config):
        calls.append(config.output.stem)
        assert config.method=='f2' and config.f2_floor_safety_fraction==.95
        return reserved_optimize(config)
    result=run(frozen,production=False,optimizer=optimizer)
    assert len(result['rows'])==len(calls)==20
    assert result['prediction_complete'] and result['annotations_read'] is False
    assert result['recomputed_methods']==['f2'] and result['archived_methods']==['analytic','dhr']
    assert result['old_timing_included_in_new_total'] is False
    assert result['recomputed_f2_complete_call_seconds_total']==pytest.approx(sum(
        row['methods']['f2']['complete_call_seconds'] for row in result['rows']))
    _load_predictions(frozen.output)
    original=json.loads((frozen.predictions/'predictions.json').read_text())
    for row,old in zip(result['rows'],original['rows'],strict=True):
        assert row['name']==old['name']
        assert row['raw_matches']['recomputed'] is False
        assert (frozen.output/row['raw_matches']['path']).resolve()==(frozen.predictions/old['raw_matches']['path']).resolve()
        for method in ('analytic','dhr'):
            item=row['methods'][method]
            assert item['recomputed'] is False and item['timing_reused_from_original_run'] is True
            for key in ('output','report','output_directory','field','postprocessing_params','configuration'):
                if isinstance(old['methods'][method].get(key),str):
                    assert not Path(item[key]).is_absolute()
                    assert (frozen.output/item[key]).resolve()==(frozen.predictions/old['methods'][method][key]).resolve()
        assert row['methods']['f2']['status']=='ok' and row['methods']['f2']['budget_complete']
        assert row['methods']['f2']['actual_strict_floor_valid']
    assert all(path.read_bytes()==content for path,content in before.items())


@pytest.mark.parametrize('bad',['incomplete','duplicate','pending','annotations','raw_failed'])
def test_source_validation_before_any_optimizer_call(frozen,bad):
    from tools.coordinated_f2_floor_all20 import run
    path=frozen.predictions/'predictions.json'
    source=json.loads(path.read_text())
    if bad=='incomplete':source['prediction_complete']=False
    elif bad=='duplicate':source['rows'][-1]=source['rows'][0]
    elif bad=='pending':source['rows'][0]['methods']['dhr']['status']='pending'
    elif bad=='annotations':source['annotations_read']=True
    else:source['rows'][0]['raw_matches']['status']='failed'
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        run(frozen,production=False,optimizer=lambda *args:pytest.fail('optimizer called'))
    assert not frozen.output.exists()


@pytest.mark.parametrize('bad',['exception','certificate','strict_floor'])
def test_new_invalid_or_failed_f2_keeps20_and_archives(frozen,bad,monkeypatch):
    from tools import coordinated_f2_floor_all20 as runner
    def optimizer(config):
        if config.output.name=='he_to_cc10_f2.npz':
            if bad=='exception':raise RuntimeError('deliberate failed newF2')
            result=reserved_optimize(config)
            if bad=='certificate':result['saved_binary_certificate']['valid']=False
            return result
        return reserved_optimize(config)
    if bad=='strict_floor':
        actual=runner._actual_ratio
        monkeypatch.setattr(runner,'_actual_ratio',lambda path:.001 if path.name=='he_to_cc10_f2.npz' else actual(path))
    result=runner.run(frozen,production=False,optimizer=optimizer)
    assert result['prediction_complete'] and len(result['rows'])==20
    assert result['method_success_counts']['f2']==19
    assert result['rows'][0]['methods']['f2']['status']=='failed'
    assert result['rows'][0]['methods']['analytic']['status']=='ok'
    assert result['rows'][0]['methods']['dhr']['status']=='ok'


def test_one_actual_tiny_optimization_and_no_overwrite(frozen):
    from tools.coordinated_f2_floor_all20 import run
    from tools.coordinated_real_case import optimize
    seen=[]
    def optimizer(config):
        seen.append(config.output.name)
        return optimize(config) if len(seen)==1 else reserved_optimize(config)
    result=run(frozen,production=False,optimizer=optimizer)
    item=result['rows'][0]['methods']['f2']
    assert item['status']=='ok' and item['gradient_steps']==4
    assert item['saved_binary_certificate']['valid'] and item['actual_strict_floor_valid']
    with pytest.raises(FileExistsError):run(frozen,production=False,optimizer=optimizer)


def test_no_fraction_sweep_or_annotation_imports(frozen):
    import ast
    from tools import coordinated_f2_floor_all20 as runner
    frozen.floor_safety_fraction=1.
    with pytest.raises(ValueError):runner.run(frozen,production=False,optimizer=reserved_optimize)
    imports=[]
    for node in ast.walk(ast.parse(Path(runner.__file__).read_text())):
        if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        elif isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
    assert all('score' not in name and 'landmark' not in name for name in imports)
