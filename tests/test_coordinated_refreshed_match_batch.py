import copy
import json
from pathlib import Path
import pytest
from tests.test_coordinated_match_fusion import fixtures,table
from tools import coordinated_refreshed_match_batch as module


@pytest.mark.parametrize('failed_index',[None,3])
def test_all_extractions_before_both_suffixes_preserve_failed_case_and_same_start(tmp_path,failed_index):
    cases,_=fixtures(tmp_path);events=[]
    original_fused=tmp_path/'old_fusion.json'
    original_fused.write_text(json.dumps(dict(original_sg_table=str(tmp_path/'sg.json'))))
    for case in cases:
        case['original_configuration']['match_weight']=.2
        case['original_configuration']['matches']=str(original_fused)
    class Model:
        setup_report={'setup_seconds':.1}
        def __init__(self,*a,**kw):self.index=0;events.append('setup')
        def extract(self,fixed,moving,affine,*,incumbent,output):
            index=self.index;self.index+=1;events.append(('extract',index))
            if index==failed_index:raise ValueError('injected extraction failure')
            value={**table(15,.2),'status':'ok'};output.write_text(json.dumps(value));return value
        def close(self):events.append('close')
    original=copy.deepcopy(cases);seen=[]
    def optimizer(cfg,*,initial_map):
        assert 'close' in events and sum(isinstance(x,tuple) and x[0]=='extract' for x in events)==25
        events.append(('optimize',cfg.output.parent.name));seen.append((cfg.output.parent.name,initial_map,cfg.matches))
        return dict(initial_map={'path':str(initial_map.resolve())},gradient_steps=300,failed_trials=0,
            initial={'total':.5},final={'total':.4})
    out=tmp_path/'result'
    report=module.run(tmp_path/'not_read.json',tmp_path,tmp_path/'weights',out,
        source_loader=lambda _:cases,model_factory=Model,optimizer=optimizer,export_validator=lambda *a:.2)
    assert cases==original and report['prediction_complete'] and report['extraction_complete']
    assert report['arms']['frozen_suffix']==dict(successful=25,failed=0)
    assert report['arms']['refreshed_suffix']==dict(successful=25 if failed_index is None else 24,failed=0 if failed_index is None else 1)
    assert len(seen)==50-(failed_index is not None)
    assert all(p==Path('unused.npz') for _,p,_ in seen)
    assert all((p==original_fused)==(arm=='frozen_suffix') for arm,_,p in seen)
    for arm in module.ARMS:
        manifest=json.loads((out/arm/'predictions.json').read_text())
        assert manifest['all50_terminal'] and len(manifest['rows'])==25
        assert (out/arm/'miit_predictions.json').exists() and (out/arm/'existing_predictions.json').exists()
