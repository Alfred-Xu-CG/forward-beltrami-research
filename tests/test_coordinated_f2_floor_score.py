"""Three-direction post-prediction scorer fixtures reuse the all20 frame data."""
import json
from pathlib import Path
import shutil

import torch
import numpy as np
import pytest

from tests.test_coordinated_lung_all20_score import cohort


@pytest.fixture
def paired(cohort,tmp_path):
    canvas,labels,predictions,source=cohort
    comparison=tmp_path/'comparison';comparison.mkdir()
    cases=('he_to_cc10','he_to_ki67','ki67_to_he')
    rows=[]
    for name in cases:
        original=next(row for row in source['rows'] if row['name']==name)['methods']['f2']
        for fraction,suffix in ((1.,'100'),(.95,'095')):
            output=name+'_f2_floor'+suffix+'.npz';report=name+'_f2_floor'+suffix+'.json'
            shutil.copyfile(predictions/original['output'],comparison/output)
            old=json.loads((predictions/original['report']).read_text())
            old['configuration']['f2_floor_safety_fraction']=fraction
            old['f2_floor_safety_fraction']=fraction
            (comparison/report).write_text(json.dumps(old))
            # Deliberately stale remote paths: only basenames under comparison are used.
            rows.append(dict(name=name,floor_safety_fraction=fraction,status='ok',
                output='/remote/copied/'+output,report='/remote/copied/'+report,
                actual_strict_floor_valid=True,saved_binary_certificate=dict(valid=True),
                gradient_steps=300,failed_trials=0))
    manifest=dict(annotations_read=False,attempts=6,paired_cases=3,rows=rows)
    (comparison/'comparison.json').write_text(json.dumps(manifest))
    return canvas,labels,predictions,comparison,manifest


def test_identity_all3_both_arms_and_no_source_mutation(paired):
    from tools.coordinated_f2_floor_score import score
    canvas,labels,predictions,comparison,manifest=paired
    before={p:p.read_bytes() for p in comparison.iterdir()}
    result=score(canvas,labels,predictions,comparison,affines_from=predictions)
    assert len(result['rows'])==3 and result['direction_denominator']==3
    assert result['specimen_count']==1
    for row in result['rows']:
        assert row['landmark_count']==80
        for name in ('floor100','floor095','common_affine'):
            item=row['methods'][name]
            assert item['status']=='ok'
            assert item['metrics']['canvas_pixels']['max']<1e-13
            assert item['metrics']['original_moving_5pc_pixels']['max']<3e-13
            assert len(item['metrics']['canvas_pixels']['per_landmark'])==80
    assert all(result['aggregates'][name]['all3_equal_direction_metrics'] is not None
               for name in ('floor100','floor095'))
    assert all(p.read_bytes()==data for p,data in before.items())


@pytest.mark.parametrize('invalid',['missing','duplicate','pending','fraction','annotations','source_incomplete'])
def test_incomplete_or_invalid_cohort_refused_before_csv(paired,monkeypatch,invalid):
    from tools import coordinated_f2_floor_score as scorer
    canvas,labels,predictions,comparison,manifest=paired
    if invalid=='missing':manifest['rows'].pop()
    elif invalid=='duplicate':manifest['rows'][-1]=manifest['rows'][0]
    elif invalid=='pending':manifest['rows'][0]['status']='pending'
    elif invalid=='fraction':manifest['rows'][0]['floor_safety_fraction']=.9
    elif invalid=='annotations':manifest['annotations_read']=True
    else:
        path=predictions/'predictions.json'
        source=json.loads(path.read_text());source['prediction_complete']=False
        path.write_text(json.dumps(source))
    (comparison/'comparison.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(scorer,'scaled_landmarks',lambda *args:pytest.fail('labels opened'))
    with pytest.raises(ValueError):
        scorer.score(canvas,labels,predictions,comparison,affines_from=predictions)


@pytest.mark.parametrize('invalid',['missing_output','fraction_report','invalid_geometry','failed'])
def test_invalid_output_or_terminal_failure_keeps_three_denominator(paired,invalid):
    from tools.coordinated_f2_floor_score import score
    canvas,labels,predictions,comparison,manifest=paired
    row=manifest['rows'][1]  # HE→CC10 reserve.95 arm only.
    output=comparison/Path(row['output']).name;report=comparison/Path(row['report']).name
    if invalid=='missing_output':output.unlink()
    elif invalid=='fraction_report':
        data=json.loads(report.read_text());data['configuration']['f2_floor_safety_fraction']=1.
        report.write_text(json.dumps(data))
    elif invalid=='failed':
        row['status']='failed';row['error']='preserved optimizer failure'
        (comparison/'comparison.json').write_text(json.dumps(manifest))
    else:
        with np.load(output) as archive:data={k:archive[k] for k in archive.files}
        data['vertices'][0,1,1]=[2,2];np.savez(output,**data)
    result=score(canvas,labels,predictions,comparison,affines_from=predictions)
    assert len(result['rows'])==3
    aggregate=result['aggregates']['floor095']
    assert aggregate['direction_denominator']==3 and aggregate['scored_directions']==2
    assert aggregate['all3_equal_direction_metrics'] is None
    assert aggregate['successful_directions_only']['canvas_pixels'] is not None
    assert result['rows'][0]['methods']['floor095']['status']=='failed'
    assert result['aggregates']['floor100']['scored_directions']==3


def test_same80_ids_required_without_intersection(paired):
    from tools.coordinated_f2_floor_score import score
    from tools.coordinated_lung_all20_score import PREFIX,STAIN_NAME
    canvas,labels,predictions,comparison,_=paired
    path=labels/(PREFIX+STAIN_NAME['ki67']+'-les3.csv')
    lines=path.read_text().splitlines();path.write_text('\n'.join(lines[:-1])+'\n')
    with pytest.raises(ValueError,match='eighty'):
        score(canvas,labels,predictions,comparison,affines_from=predictions)
