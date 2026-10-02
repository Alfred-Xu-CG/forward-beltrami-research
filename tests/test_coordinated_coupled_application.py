import copy
import numpy as np
import pytest
import torch
from tools import coordinated_real_case as application
from qcopt.neural_bijection.dense import coordinated_coupled_seed as coupled
from test_coordinated_shared_affine_evidence import configuration


def test_default_identity_initializer_is_exact_original_path(tmp_path):
    cfg=configuration(tmp_path,'analytic');cfg.mind_frame='shared_affine'
    original=application.optimize(cfg)
    with np.load(cfg.output) as data:saved={key:data[key].copy() for key in data.files}
    explicit=copy.deepcopy(cfg);explicit.seed_initializer='identity';explicit.output=tmp_path/'identity.npz'
    result=application.optimize(explicit)
    for key in ('initial','final','stages','trace','objective_evaluations','gradient_steps'):
        assert result[key]==original[key]
    assert result['seed_record'] is None and result['seed_objective_evaluations']==0
    with np.load(explicit.output) as data:
        assert set(data.files)==set(saved)
        for key in data.files:np.testing.assert_array_equal(data[key],saved[key])


def test_always_refine_one_seed_even_when_E_increases_then_identity_can_win(tmp_path,monkeypatch):
    cfg=configuration(tmp_path,'analytic');cfg.mind_frame='shared_affine';cfg.seed_initializer='coupled_mind'
    cfg.grid_side=257;cfg.image_side=512;cfg.levels=[17,33];cfg.image_levels=[128,512];cfg.inner_steps=1
    cfg.output=tmp_path/'coupled.npz';seen=[];calls=[]
    def build(fixed,moving,mask,matrix,offset,reference):
        assert fixed.shape==moving.shape==(1,8,128,128) and mask.shape==(1,1,128,128)
        assert reference.shape==(1,257,257,2)
        proposal=torch.ones(2,3,3,dtype=torch.float64)*.25
        result=coupled.construct_safe_seed(reference,proposal)
        seen.append(result.vertices.clone());return result
    def objective(self,vertices):
        calls.append(self.fixed.shape[-1])
        reference=application.identity_vertices(vertices.shape[1],device=vertices.device).to(vertices.dtype)
        loss=(vertices-reference).square().sum(-1).mean()
        return loss,{'image':loss}
    monkeypatch.setattr(coupled,'build_coupled_mind_seed',build)
    monkeypatch.setattr(application.Evidence,'__call__',objective)
    result=application.optimize(cfg)
    assert len(seen)==1 and result['seed_record']['original_E_increased'] is True
    assert result['seed_record']['original_E_seed_rejection'] is False
    assert result['stages'][0]['anchor_total']>0 and result['gradient_steps']==4
    assert result['objective_evaluations']==len(calls)==19
    assert result['seed_objective_evaluations']==1 and result['selected_stage'] is None
    assert result['initial']['total']==result['final']['total']==0
    with np.load(cfg.output) as data:
        np.testing.assert_array_equal(data['vertices'],data['boundary_reference'])
        np.testing.assert_array_equal(data['initializer_vertices'],seen[0].numpy())
        assert data['initializer_coefficients_128px'].shape==(1,5,5,2)


@pytest.mark.parametrize('change',[{'method':'f2'},{'grid_side':129},{'mind_frame':'original'},
    {'output_selection':'last'},{'capture_prefix':'mind_discrete'},{'strain_weight':2.}])
def test_reject_undeclared_coupled_configuration_before_loading(tmp_path,monkeypatch,change):
    cfg=configuration(tmp_path,'analytic');cfg.mind_frame='shared_affine';cfg.seed_initializer='coupled_mind'
    cfg.grid_side=257;cfg.image_side=512;cfg.levels=[17,33];cfg.image_levels=[128,512]
    for key,value in change.items():setattr(cfg,key,value)
    monkeypatch.setattr(application,'load_registration_evidence',lambda *a,**kw:pytest.fail('loaded unsupported initializer'))
    with pytest.raises(ValueError):application.optimize(cfg)
