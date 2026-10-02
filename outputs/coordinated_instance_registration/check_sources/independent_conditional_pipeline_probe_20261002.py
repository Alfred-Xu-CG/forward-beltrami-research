"""Independent CPU lifecycle/peak/timing oracle; no actual inference or labels."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools import coordinated_conditional_pipeline as module


def peaks():
    class Cuda:
        def __init__(self):self.a=self.r=0;self.resets=0;self.fail_sync=False
        def synchronize(self,device):
            if self.fail_sync:raise ArithmeticError('synthetic exit synchronization failure')
        def max_memory_allocated(self,device):return self.a
        def max_memory_reserved(self,device):return self.r
        def reset_peak_memory_stats(self,*a,**kw):self.a=self.r=0;self.resets+=1
    for exit_error in (False,True):
        api=Cuda();original=api.reset_peak_memory_stats;collector=module.ResetAwarePeaks('cuda',cuda_api=api)
        try:
            with collector:
                api.a,api.r=2300,2600;api.reset_peak_memory_stats('cuda')
                api.a,api.r=1900,3100;api.reset_peak_memory_stats(device='cuda')
                api.a,api.r=1800,2100;api.fail_sync=exit_error
        except ArithmeticError:assert exit_error
        assert api.reset_peak_memory_stats==original and module.ResetAwarePeaks._active is False
        assert collector.allocated==2300 and collector.reserved==3100
        assert collector.segments==(2 if exit_error else 3)
    return dict(setup_before_nested_reset_captured=True,allocated_reserved_independent_maxima=True,
        original_API_restored_even_on_exit_sync_error=True)


def lifecycle():
    # Reuse only simple input-fixture construction; state/order/timing oracles
    # below do not call the production algorithm's validation helper.
    from tests.test_coordinated_conditional_pipeline import fixture,table
    with tempfile.TemporaryDirectory(prefix='independent_conditional_') as td:
        td=Path(td);cases=fixture(td);output=td/'fresh';events=[]
        class Clock:
            now=0.
            def __call__(self):self.now+=.001;return self.now
        clock=Clock()
        def sg(args):
            i=int(args.output.stem[4:]);events.append(('sg',i));clock.now+=.2
            declared=json.loads((output/'predictions.json').read_text());assert len(declared['rows'])==25 and not declared['prediction_complete']
            if i==2:raise ValueError('synthetic SG failure')
            r=table();r['confidence'][0]=.7;args.output.write_text(json.dumps(r));return r
        class Model:
            def __init__(self,*args,**kwargs):
                assert len(events)==25;events.append(('setup',None));clock.now+=.3;self.setup_report={}
            def extract(self,fixed,moving,affine,*,output):
                i=int(output.stem[4:]);events.append(('ma',i));clock.now+=.4
                if i==3:raise ValueError('synthetic MA failure')
                r=table();output.write_text(json.dumps(r));return r
            def close(self):events.append(('close',None));clock.now+=.1
        def optimize(cfg):
            assert events.count(('close',None))==1 and sum(e[0]=='ma' for e in events)==25
            i=int(cfg.output.stem.split('_')[0][4:]);events.append(('opt',i));clock.now+=.5
            fresh=json.loads(cfg.matches.read_text());assert abs(fresh['confidence'][0]-.7/(9*.8+.7))<1e-15
            assert cfg.match_weight==.2 and cfg.inner_steps==30 and cfg.levels==[17,33,65,129,257]
            if i==4:raise ValueError('synthetic optimizer failure')
            return dict(gradient_steps=300,failed_trials=0)
        def scoring(m,o):
            assert m['prediction_complete'] and len(m['rows'])==25 and all(r['status'] in ('ok','failed') for r in m['rows'])
            events.append(('score',None));clock.now+=.01
        with patch.object(module,'source_cases',lambda *a:copy.deepcopy(cases)),patch.object(module.time,'perf_counter',clock):
            result=module.run('x','x',td/'ma_manifest.json',td/'fusion_manifest.json',output,'x','x',
                sg_extractor=sg,model_factory=Model,optimizer=optimize,export_validator=lambda *a:.1,scoring_adapter=scoring)
        assert result['failed']==3 and result['successful']==22 and result['attempt_denominator']==25
        assert result['sg_setup_attempts']==25 and result['ma_setup_attempts']==1 and result['ma_extraction_attempts']==25
        success=[r for r in result['rows'] if r['status']=='ok']
        times=[r['completed_map_elapsed_from_batch_start'] for r in success]
        assert times==sorted(times) and result['time_to_first_saved_map']==times[0]>15
        assert times[-1]<result['whole_batch_seconds']
        assert result['batch_seconds_per_attempt']==result['whole_batch_seconds']/25
        assert result['successful_maps_per_batch_second']==22/result['whole_batch_seconds']
        for row in result['rows']:
            assert row['complete_call_seconds']==sum(row['costs'].values())
            if row['status']=='ok':assert row['complete_call_seconds']<row['completed_map_elapsed_from_batch_start']
        assert not result['all_tables_exact']
        assert json.loads((td/'sg.json').read_text())==table()
        assert result['whole_batch_seconds']>=sum(r['complete_call_seconds'] for r in result['rows'])+result['ma_setup_seconds']+result['ma_close_seconds']
        assert result['rows'][2]['optimization_status']==result['rows'][3]['optimization_status']=='skipped'
        assert result['rows'][4]['optimization_status']=='failed'
        try:module.run('x','x','x','x',output,'x','x',sg_extractor=sg,model_factory=Model,optimizer=optimize)
        except FileExistsError:pass
        else:raise AssertionError('existing output must not be overwritten')
    return dict(all25_declared_and_failure_denominator_preserved=True,fresh_changed_SG_reaches_fused_optimizer=True,
        supplied_configuration_unchanged=True,allSG_then_allMA_close_then_optimization=True,
        component_sums_distinct_from_completion_latency=True,first_map_and_batch_offsets_verified=True,
        existing_output_and_archived_tables_preserved=True)


if __name__=='__main__':print(json.dumps(dict(peaks=peaks(),lifecycle=lifecycle()),indent=2))
