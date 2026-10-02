"""Independent real-cohort configuration/order/failure/scoring-path dry run."""
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import json
import sys

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools import coordinated_fusion_f2_control as b
BASE=ROOT/'outputs/coordinated_instance_registration'


def main():
    originals=b.source_cases(BASE/'match_fusion_all50_t23/fusion/predictions.json')
    with TemporaryDirectory(prefix='independent_f2_',dir=BASE) as td:
        target=Path(td)/'out';events=[]
        def loader(_):
            rows=copy.deepcopy(originals)
            for row in rows:
                row['original_configuration']['device']='cpu'
                value=row['source_manifest'].replace('\\','/')
                row['source_manifest']=str(BASE/value.split('/results/',1)[-1].removeprefix('results/'))
                assert Path(row['source_manifest']).is_file()
            return rows
        def optimizer(cfg):
            outer=json.loads((target/'comparison.json').read_text())
            assert len(outer['planned_calls'])==50
            before={m:json.loads((target/m/'predictions.json').read_text()) for m in ('analytic','f2')}
            assert sum(len(a['rows']) for a in before.values())==50
            assert not (target/'analytic/miit_predictions.json').exists()
            events.append((cfg.output.name,cfg.method))
            if len(events)==2:raise RuntimeError('injected optimizer failure')
            result=dict(**b.budget(cfg),failures=[],landmarks_used=False,saved_binary_certificate={'valid':True})
            if len(events)==3:result.update(gradient_steps=217,evaluations=229,objective_evaluations=251,failed_trials=1,
                                           failures=[dict(reason='injected rounded-floor stop')])
            if len(events)==4:result['inject_validator_failure']=True
            return result
        def validator(cfg,result):
            if result.get('inject_validator_failure'):raise ValueError('injected export failure')
            return .1
        report=b.run('unused',target,loader=loader,optimizer=optimizer,export_validator=validator)
        assert report['all50_terminal'] and report['successful_calls']==47 and len(events)==50
        expected=[m for i in range(25) for m in (('analytic','f2') if i%2==0 else ('f2','analytic'))]
        assert [m for _,m in events]==expected
        arms={m:json.loads((target/m/'predictions.json').read_text()) for m in ('analytic','f2')}
        assert all(x['status'] in ('ok','failed') for a in arms.values() for x in a['rows'])
        partial=arms['f2']['rows'][1]
        assert partial['status']=='failed' and partial['gradient_steps']==217 and partial['failed_trials']==1
        assert partial['budget_complete'] is False and partial['saved_binary_certificate']['valid']
        miit=json.loads((target/'analytic/miit_predictions.json').read_text())
        assert miit['actual_methods']==['analytic','f2'] and miit['archived_dhr_reference_only']
        for row in miit['rows']:
            for method in ('analytic','f2'):
                new=row['methods'][method]
                assert 'pilot_reference' not in new
                assert new['configuration']['method']==method
                file=(target/'analytic'/new['output']).resolve()
                assert file.parent==target/method and file.name==row['name']+'_'+method+'.npz'
            assert 'unchanged archived' in row['methods']['dhr']['pilot_reference']
        existing=json.loads((target/'f2/existing_predictions.json').read_text())
        assert existing['method']=='f2' and len(existing['rows'])==22
        assert all(r['configuration']['method']=='f2' and r['output'].endswith('_f2.npz') for r in existing['rows'])
        saved=(target/'comparison.json').read_bytes()
        try:b.run('unused',target,loader=loader)
        except FileExistsError:pass
        else:raise AssertionError('existing result accepted')
        assert (target/'comparison.json').read_bytes()==saved
    print('PASS: actual25 configs, all50 order, 3 failures, partial counters, actual A/F2 scoring paths and no overwrite')


if __name__=='__main__':main()
