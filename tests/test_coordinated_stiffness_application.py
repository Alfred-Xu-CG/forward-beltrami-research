import copy
import numpy as np
from tools import coordinated_real_case as original
from tools import coordinated_stiffness_application as application
from test_coordinated_shared_affine_evidence import configuration


def test_same_initial_evidence_actual_call_counts_budget_prefix_and_export(tmp_path,monkeypatch):
    cfg=configuration(tmp_path,'analytic'); cfg.mind_frame='shared_affine'
    baseline=original.optimize(cfg)
    new=copy.deepcopy(cfg); new.output=tmp_path/'stiffness.npz'
    calls=[]; previous=original.Evidence.__call__
    def counted(self,vertices):
        calls.append(self.fixed.shape[-1])
        return previous(self,vertices)
    monkeypatch.setattr(original.Evidence,'__call__',counted)
    result=application.optimize_stiffness(new)
    assert result['initial']==baseline['initial']
    assert result['objective_evaluations']==len(calls)
    assert 0<result['gradient_steps']<=4 and len(result['stages'])==4
    assert result['final']['total']==min([result['initial']['total']]+[s['accepted_full_total'] for s in result['stages']])
    assert result['output_selection']=='best_full' and result['landmarks_used'] is False
    assert result['saved_binary_certificate']['valid']
    assert set(result['mind_frame_by_resolution'])=={'8','16'}
    assert result['objective_evaluations']==2+len(result['stages'])+sum(s['actual_counts']['objective_evaluations'] for s in result['stages'])
    with np.load(new.output) as z:
        v,r=z['vertices'],z['boundary_reference']
        np.testing.assert_array_equal(v[:,0],r[:,0]); np.testing.assert_array_equal(v[:,:,0],r[:,:,0])
        np.testing.assert_array_equal(v[:,-1],r[:,-1]); np.testing.assert_array_equal(v[:,:,-1],r[:,:,-1])
        assert str(z['interpolation'])=='p1_ac'
