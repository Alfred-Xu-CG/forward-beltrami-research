from dataclasses import replace
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT=Path(__file__).resolve().parents[1]/'experiments/phase5/route1_validity_suite.py'


def module():
    assert SCRIPT.exists(), 'validity suite is not implemented'
    spec=importlib.util.spec_from_file_location('phase5_validity_test',SCRIPT)
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    return m


def tiny(m, **kwargs):
    return replace(m.ValidityConfig(controls=(3,),strengths=(0.25,),seeds=(7,),resolutions=(8,)),**kwargs)


def test_default_coverage_and_tiny_independent_metrics(tmp_path):
    m=module(); defaults=m.ValidityConfig()
    assert defaults.controls==(11,17,25,33,49)
    assert defaults.strengths==(0.25,1.0,3.0) and len(defaults.seeds)>=3
    assert defaults.resolutions==(256,512)
    report=m.run_suite(tiny(m))
    assert report['status']=='ok',report
    analytic=[r for r in report['rows'] if r['kind']=='analytic']
    assert {r['name'] for r in analytic}==set(m.ANALYTIC_DEFORMATION_NAMES)
    assert len(report['rows'])==13
    for row in analytic:
        assert row['continuous_determinant_lower_bound']>0
        assert row['metrics']['global_injectivity_certificate']
        assert row['independent_minimum_signed_area']>0
        assert row['mu_magnitude']['max']<1
        assert row['accuracy'] is None
    assert not next(r for r in analytic if r['name']=='shear')['unit_square_sampling_compatible']
    random=next(r for r in report['rows'] if r['kind']=='positive_tutte')
    assert 0<random['supported_probability_min']<=random['supported_probability_max']<1
    assert random['accuracy'] is None and random['height']==1.0
    assert all(r['inverse_consistency'] is None for r in report['rows'])
    m.write_results(report,tmp_path/'out.json',tmp_path/'out.csv')
    assert json.loads((tmp_path/'out.json').read_text())['status']=='ok'
    with (tmp_path/'out.csv').open(newline='') as csv_file:
        assert len(list(csv.DictReader(csv_file)))==13


def test_cached_sample_certificate_cannot_override_independent_fold_detection(monkeypatch):
    m=module(); original=m.sample_analytic_deformation
    def corrupt(mesh,name):
        sample=original(mesh,name)
        if name!='shear': return sample
        vertices=sample.vertices.copy(); vertices[:,0]*=-1
        return SimpleNamespace(vertices=vertices,continuous_determinant_lower_bound=1.0,
                               injectivity_report=SimpleNamespace(certified=True))
    monkeypatch.setattr(m,'sample_analytic_deformation',corrupt)
    report=m.run_suite(tiny(m))
    row=next(r for r in report['rows'] if r['name']=='shear')
    assert report['status']=='failure' and row['status']=='failure'
    assert not row['eligible_ground_truth']


def test_random_targets_reproducible_and_no_self_accuracy_computation(monkeypatch):
    m=module(); original=m.compute_p1_map_metrics
    def checked(mesh,mapped,**kwargs):
        assert kwargs.get('target') is None,'self-comparison is not an accuracy test'
        return original(mesh,mapped,**kwargs)
    monkeypatch.setattr(m,'compute_p1_map_metrics',checked)
    cfg=tiny(m,strengths=(0.25,1.0,3.0),seeds=(7,8,9))
    first=m.run_suite(cfg); second=m.run_suite(cfg)
    a=[r for r in first['rows'] if r['kind']=='positive_tutte']
    b=[r for r in second['rows'] if r['kind']=='positive_tutte']
    assert len(a)==9 and a==b
    assert all(r['status']=='ok' for r in a)
    assert a[-1]['supported_probability_min']<a[0]['supported_probability_min']


@pytest.mark.parametrize('resolution',[256,512])
def test_all_images_native_sizes_and_independent_backward_oracle(resolution):
    m=module()
    for name in m.IMAGE_NAMES:
        row=m.audit_image(name,resolution,dtype=torch.float32)
        assert row['status']=='ok',row
        assert row['image_min']==0 and row['image_max']==1 and row['deterministic']
        assert row['identity_max_error']<=row['sampling_tolerance']
        assert row['positive_x_shift_max_error']<=row['sampling_tolerance']
        assert row['pair_bilinear_oracle_max_error']<=row['sampling_tolerance']
        assert row['backward_convention']=='fixed_to_moving_align_corners_true'
        assert row['inverse_consistency'] is None


def test_warp_convention_check_detects_ignored_map(monkeypatch):
    m=module()
    monkeypatch.setattr(m,'warp_image_backward',lambda image,q:image[...,:q.shape[-3],:q.shape[-2]])
    with pytest.raises(ValueError,match='shift|convention'):
        m.audit_image('textured',17,dtype=torch.float64)


def test_warp_convention_check_rejects_nonfinite_warp(monkeypatch):
    m=module()
    monkeypatch.setattr(m,'warp_image_backward',lambda image,q:torch.full_like(image[...,:q.shape[-3],:q.shape[-2]],float('nan')))
    with pytest.raises(ValueError,match='finite|convention'):
        m.audit_image('textured',17,dtype=torch.float64)


def test_independent_face_beltrami_known_affine():
    m=module(); mesh=m.structured_rectangle(2,2)
    target=mesh.vertices@np.diag([2.,1.])
    audit=m.audit_map(mesh,target)
    assert audit['independent_minimum_area_ratio']==pytest.approx(2.)
    assert audit['mu_magnitude']['max']==pytest.approx(1/3)
    assert 'mu_rmse' not in audit['metrics']


def test_independent_face_jacobian_and_beltrami_non_diagonal_affine():
    m=module(); mesh=m.structured_rectangle(3,2)
    jacobian=np.array([[1.4,0.3],[0.2,1.1]],dtype=np.float64)
    target=mesh.vertices@jacobian.T+np.array([0.17,-0.23])
    audit=m.audit_map(mesh,target)
    a,b=jacobian[0]
    c,d=jacobian[1]
    expected=abs(complex(a-d,c+b)/complex(a+d,c-b))
    assert audit['independent_minimum_area_ratio']==pytest.approx(np.linalg.det(jacobian))
    assert audit['mu_magnitude']['min']==pytest.approx(expected,abs=2e-15)
    assert audit['mu_magnitude']['max']==pytest.approx(expected,abs=2e-15)


def test_environment_and_csv_record_reproducibility_context(tmp_path):
    m=module(); report=m.run_suite(tiny(m))
    environment=report['environment']
    assert environment['commit']
    assert isinstance(environment['git_dirty'],bool)
    assert environment['scipy']
    m.write_results(report,tmp_path/'out.json',tmp_path/'out.csv')
    with (tmp_path/'out.csv').open(newline='') as csv_file:
        image=next(row for row in csv.DictReader(csv_file)
                   if row['kind']=='image')
    assert float(image['sampling_tolerance'])>0


def test_reject_invalid_config_and_same_output_path(tmp_path):
    m=module()
    with pytest.raises(ValueError): m.run_suite(tiny(m,strengths=(-1.,)))
    with pytest.raises(ValueError): m.run_suite(tiny(m,controls=(3,3)))
    with pytest.raises(ValueError): m.write_results({'rows':[]},tmp_path/'same',tmp_path/'same')
