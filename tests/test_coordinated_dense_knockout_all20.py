"""Dense knockout changes only beta, never labels or archived predictions."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import pytest

from tests.test_coordinated_level_allocation_all20 import prepared, mock_optimizer


@pytest.fixture
def uniform(prepared):
    from tools.coordinated_level_allocation_all20 import run
    run(prepared,production=False,optimizer=mock_optimizer)
    return argparse.Namespace(canvas=prepared.canvas,affines_from=prepared.affines_from,
        predictions=prepared.uniform_output,output=prepared.uniform_output.parent/'knockout',device='cpu',threads=1)


def mock_knockout(config):
    result=mock_optimizer(config)
    result.update(image_weight=config.image_weight)
    result['final']['image']=.15
    result['initial']['image']=.2
    config.output.with_suffix('.json').write_text(json.dumps(result))
    return result


def test_production_exact_uniform_recipe_except_weight(uniform):
    from tools.coordinated_dense_knockout_all20 import make_configuration
    from tools.coordinated_level_allocation_all20 import make_configuration as original
    from tools.coordinated_lung_all20 import pairs
    pair=pairs(uniform)[0]
    source=argparse.Namespace(**{key:value for key,value in vars(uniform).items() if key!='output'},uniform_output=uniform.output)
    left=original(pair,source,'uniform',uniform.predictions/'raw.json')
    right=make_configuration(pair,uniform,uniform.predictions/'raw.json')
    assert right.image_weight==0 and right.inner_steps_by_level==[30]*5
    assert right.joint_prior_backend=='inductor' and right.match_p1_sampling=='existing'
    a,b=dict(vars(left)),dict(vars(right))
    b.pop('image_weight');b.pop('match_p1_sampling')
    assert a==b
    assert 2*sum(right.inner_steps_by_level)==300


def test_all20_only_analytic_and_relative_archives_source_unchanged(uniform):
    from tools.coordinated_dense_knockout_all20 import run
    from tools.coordinated_lung_all20_score import _load_predictions
    before={p:p.read_bytes() for p in uniform.predictions.rglob('*') if p.is_file()}
    calls=[]
    def optimizer(config):
        calls.append(config.output.name)
        assert config.image_weight==0 and config.method=='analytic'
        return mock_knockout(config)
    result=run(uniform,production=False,optimizer=optimizer)
    _load_predictions(uniform.output)
    assert len(calls)==len(result['rows'])==20 and result['prediction_complete']
    assert result['annotations_read'] is False and result['image_weight']==0
    assert result['objective_totals_comparable_to_source'] is False
    assert result['old_timing_included_in_new_total'] is False
    old=json.loads((uniform.predictions/'predictions.json').read_text())
    for row,source in zip(result['rows'],old['rows'],strict=True):
        record=row['methods']['analytic']
        assert record['status']=='ok' and record['budget_complete']
        assert record['final']['image']==.15 and record['image_weight']==0
        assert record['gradient_steps']==8 and record['evaluations']==12 and record['objective_evaluations']==22
        assert (uniform.output/record['paired_beta1_output']).resolve()==(uniform.predictions/source['methods']['analytic']['output']).resolve()
        assert (uniform.output/record['paired_beta1_report']).resolve()==(uniform.predictions/source['methods']['analytic']['report']).resolve()
        for method in ('f2','dhr'):
            for key in ('output','report','field','configuration','postprocessing_params','output_directory'):
                if isinstance(source['methods'][method].get(key),str):
                    assert (uniform.output/row['methods'][method][key]).resolve()==(uniform.predictions/source['methods'][method][key]).resolve()
    assert all(p.read_bytes()==content for p,content in before.items())


@pytest.mark.parametrize('failure',['exception','budget','certificate','floor','wrong_beta'])
def test_failed_attempt_keeps_direction_denominator(uniform,failure,monkeypatch):
    from tools import coordinated_dense_knockout_all20 as runner
    def optimizer(config):
        if config.output.name=='he_to_cc10_analytic.npz':
            if failure=='exception':raise RuntimeError('deliberate failed knockout')
            value=mock_knockout(config)
            if failure=='budget':value['gradient_steps']=7
            if failure=='certificate':value['saved_binary_certificate']['valid']=False
            if failure=='wrong_beta':value['image_weight']=1
            return value
        return mock_knockout(config)
    if failure=='floor':
        actual=runner._actual_ratio
        monkeypatch.setattr(runner,'_actual_ratio',lambda path:.001 if path.name=='he_to_cc10_analytic.npz' else actual(path))
    result=runner.run(uniform,production=False,optimizer=optimizer)
    assert result['prediction_complete'] and len(result['rows'])==20
    assert result['method_success_counts']['analytic']==19
    assert result['rows'][0]['methods']['analytic']['status']=='failed'


@pytest.mark.parametrize('failure',['incomplete','allocation','source_beta','prior','point_backend'])
def test_wrong_source_refused_before_new_outputs(uniform,failure):
    from tools.coordinated_dense_knockout_all20 import run
    path=uniform.predictions/'predictions.json';value=json.loads(path.read_text())
    if failure=='incomplete':value['prediction_complete']=False
    elif failure=='allocation':value['allocation']=[2,2]
    elif failure=='source_beta':value['rows'][0]['methods']['analytic']['configuration']['image_weight']=0
    elif failure=='prior':value['rows'][0]['methods']['analytic']['configuration']['strain_weight']=.05
    else:value['rows'][0]['methods']['analytic']['configuration']['match_p1_sampling']='frozen'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):run(uniform,production=False,optimizer=lambda *a:pytest.fail('called'))
    assert not uniform.output.exists()


def test_no_source_overwrite_or_scoring_import(uniform):
    import ast
    from tools import coordinated_dense_knockout_all20 as runner
    uniform.output=uniform.predictions
    with pytest.raises((ValueError,FileExistsError)):runner.run(uniform,production=False,optimizer=mock_knockout)
    imports=[node.module or '' for node in ast.walk(ast.parse(Path(runner.__file__).read_text())) if isinstance(node,ast.ImportFrom)]
    assert all('score' not in name and 'landmark' not in name for name in imports)


def test_nonfinite_returned_diagnostics_fail_one_row_without_aborting_json(uniform):
    from tools.coordinated_dense_knockout_all20 import run
    visits=[]
    def optimizer(config):
        visits.append(config.output.name)
        result=mock_knockout(config)
        if config.output.name=='he_to_cc10_analytic.npz':
            result['final'].update(total=float('nan'),diagnostic={'nested':[1.,float('inf'),-float('inf')]})
        return result
    result=run(uniform,production=False,optimizer=optimizer)
    assert result['prediction_complete'] and len(visits)==20
    assert result['method_success_counts']['analytic']==19
    failed=result['rows'][0]['methods']['analytic']
    assert failed['status']=='failed' and 'nonfinite' in failed['error']
    assert failed['final']['total'] is None
    assert failed['final']['diagnostic']['nested']==[1.,None,None]
    assert 'final.total' in failed['nonfinite_diagnostic_paths']
    assert 'final.diagnostic.nested[1]' in failed['nonfinite_diagnostic_paths']
    text=(uniform.output/'predictions.json').read_text()
    # Python's normal decoder accepts NaN extensions; forbid them explicitly.
    saved=json.loads(text,parse_constant=lambda value:pytest.fail('nonstandard JSON '+value))
    assert len(saved['rows'])==20 and saved['rows'][0]['methods']['analytic']['status']=='failed'
