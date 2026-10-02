"""Small CPU capture fixtures; not actual Inductor or GPU timing validation."""
import json
import numpy as np
import pytest
import torch
from tools import coordinated_dispatch_equivalence as probe
from tools.coordinated_application_profile import load_configuration
from test_coordinated_application_profile import fixture_report
from tools.digital_q1_dhr_distill import identity_vertices


def fixture(tmp_path):
    source=fixture_report(tmp_path)
    c=vars(load_configuration(source,tmp_path/'unused.npz',production=False))
    c.update(mind_frame='shared_affine',match_weight=.2,image_levels=[8,16])
    anchor=tmp_path/'anchor.npz'
    vertices=identity_vertices(c['grid_side'],device='cpu').double().numpy()
    deformed=vertices.copy();deformed[:,1:-1,1:-1,0]+=.002
    with np.load(c['affine'],allow_pickle=False) as a:
        np.savez(anchor,vertices=deformed,boundary_reference=vertices,interpolation='p1_ac',
            post_affine_matrix=a['post_affine_matrix'],post_affine_offset=a['post_affine_offset'])
    return c,anchor


def test_actual_full_evidence_all_states_levels_and_repeats(tmp_path):
    c,anchor=fixture(tmp_path)
    report=probe._check(c,anchor,outputpath=tmp_path/'check.json',production=False,compiler_backend='eager')
    assert report['passed'],report.get('error',report['rows'])
    assert len(report['rows'])==4 and report['compiler_backend']=='eager'
    assert report['annotations_read'] is False
    assert {r['state'] for r in report['rows']}=={'identity','saved_deformed'}
    assert all(len(r['comparisons'])==6 and 'match' in r['component_names'] for r in report['rows'])
    assert all(r['values'][0][r['component_names'].index('match')] >= 0 for r in report['rows'])
    assert json.loads((tmp_path/'check.json').read_text())['passed']


def test_gradient_corruption_and_nonfinite_rejected():
    a=dict(names=['total'],values=torch.ones(1),gradient=torch.ones(2,2))
    b={**a,'gradient':a['gradient']+.001}
    assert not probe._compare(a,b)['passed']
    b={**a,'gradient':torch.full((2,2),float('nan'))}
    result=probe._compare(a,b)
    assert not result['passed'] and result['full_vertex_gradient']['maximum_absolute_error'] is None
    json.dumps(result,allow_nan=False)
    extreme=dict(names=['total'],values=torch.full((1,),1e308,dtype=torch.float64),
                 gradient=torch.full((2,2),1e308,dtype=torch.float64))
    opposed={**extreme,'values':-extreme['values'],'gradient':-extreme['gradient']}
    overflow=probe._compare(extreme,opposed)
    assert not overflow['passed'] and overflow['values']['maximum_absolute_error'] is None
    json.dumps(overflow,allow_nan=False)


def test_existing_file_preserved_and_production_rejects_tiny(tmp_path):
    c,anchor=fixture(tmp_path);out=tmp_path/'keep.json';out.write_text('preserve')
    with pytest.raises(FileExistsError):probe.check(c,anchor,outputpath=out)
    assert out.read_text()=='preserve'
    with pytest.raises(ValueError,match='production'):probe.check(c,anchor,outputpath=tmp_path/'new.json')


def test_bad_anchor_fails_without_warm_success(tmp_path):
    c,anchor=fixture(tmp_path)
    result=probe._check(c,tmp_path/'missing.npz',outputpath=tmp_path/'failed.json',production=False,compiler_backend='eager')
    assert result['passed'] is False and result['status']=='failed' and not result['rows']


@pytest.mark.parametrize('change',[{'mind_frame':'original'},{'match_weight':.1},{'preprocessing':'native_dhr'},{'fixed_mask':'labels.png'}])
def test_changed_evidence_is_not_silently_reconstructed(tmp_path,change):
    c,anchor=fixture(tmp_path);c.update(change)
    with pytest.raises(ValueError):
        probe._check(c,anchor,outputpath=tmp_path/'no.json',production=False,compiler_backend='eager')
