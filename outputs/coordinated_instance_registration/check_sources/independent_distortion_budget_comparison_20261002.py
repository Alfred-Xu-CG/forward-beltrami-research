"""Independent reaggregation from verified cap endpoints and archived scores."""
from collections import Counter
import json
from independent_distortion_budget_postrun_20261002 import BASE,ARCHIVE,read,direct_D

DIRECTORY=BASE/'distortion_budget_all25_t28'


def main():
    checked=read(DIRECTORY/'independent_check.json')
    assert checked['successful_maps_independently_recomputed']==25 and checked['total_original_CSV_errors']==2074 and not checked['failures']
    comparison=read(DIRECTORY/'comparison.json');methods=('fusion300','frozen_suffix','distortion_cap')
    paths=dict(fusion300=ARCHIVE,frozen_suffix=BASE/'refreshed_match_all50_t27/frozen_suffix',distortion_cap=DIRECTORY)
    manifests={a:read(p/'predictions.json') for a,p in paths.items()}
    scored={a:{r['name']:r for kind in ('miit','existing') for r in read(p/(kind+'_scores.json'))['rows']} for a,p in paths.items()}
    scores={a:{name:(r['methods']['analytic']['metrics'] if name.startswith('miit_') else r['metrics'])['canvas_pixels'] for name,r in records.items()} for a,records in scored.items()}
    names=[r['name'] for r in checked['rows']]
    cohort=lambda n:'miit' if n.startswith('miit_') else n if n in ('histo','rat_kidney') else 'lung_all20'
    groups={c:[n for n in names if cohort(n)==c] for c in ('miit','lung_all20','histo','rat_kidney')}
    assert {c:len(n) for c,n in groups.items()}==dict(miit=3,lung_all20=20,histo=1,rat_kidney=1)
    assert not comparison['failed_attempts']
    for a,m in manifests.items():
        assert [r['name'] for r in m['rows']]==names and all(r['status']=='ok' for r in m['rows'])
        for n in names:assert scored[a][n]['available_pair_labels']==scored['fusion300'][n]['available_pair_labels']
    for i,row in enumerate(comparison['rows']):
        name=names[i];assert row['name']==name and row['cohort']==cohort(name)
        assert row['statuses']=={a:'ok' for a in methods}
        for metric in ('mean','p90','maximum'):
            for a in methods:assert row['metrics'][metric]['values'][a]==scores[a][name][metric]
            for control in methods[:2]:
                assert row['metrics'][metric]['deltas']['cap_minus_'+control]==scores['distortion_cap'][name][metric]-scores[control][name][metric]
        for a in methods:
            prediction=manifests[a]['rows'][i];actual=row['objectives'][a]
            expected=dict(final_D=direct_D(prediction['final']),initial_D=direct_D(prediction['initial']),
                final_R=prediction['final']['strain'],initial_R=prediction['initial']['strain'],
                original_E=prediction['final']['original_total' if a=='distortion_cap' else 'total'],
                gradient_steps=prediction['gradient_steps'],selected_stage=prediction.get('selected_stage'))
            assert actual==expected
        cap=manifests['distortion_cap']['rows'][i];frozen=manifests['frozen_suffix']['rows'][i]
        assert cap['incumbent']==frozen['incumbent']==cap['initial_map']['path']==frozen['initial_map']['path']
        assert {k:v for k,v in cap['configuration'].items() if k!='output'}=={k:v for k,v in frozen['configuration'].items() if k!='output'}
        assert cap['final']['total']<=direct_D(frozen['initial'])
    for c,selected in groups.items():
        for a in methods:
            actual=comparison['cohorts'][c]['methods'][a]
            assert actual['pair_denominator']==actual['scored_pairs']==len(selected) and actual['failure_count']==0
            for key,metric in (('mean_pair_mean','mean'),('mean_pair_p90','p90'),('worst_pair_maximum','maximum')):
                values=[scores[a][n][metric] for n in selected]
                expected=max(values) if metric=='maximum' else sum(values)/len(values)
                assert abs(actual['all_directions_canvas_pixels'][key]-expected)<1e-12
        for control in methods[:2]:
            delta=comparison['cohorts'][c]['deltas']['cap_minus_'+control]
            for key in delta:
                assert delta[key]==comparison['cohorts'][c]['methods']['distortion_cap']['all_directions_canvas_pixels'][key]-comparison['cohorts'][c]['methods'][control]['all_directions_canvas_pixels'][key]
    for control in methods[:2]:
        label='cap_minus_'+control
        for key,metric in (('mean_pair_mean','mean'),('mean_pair_p90','p90')):
            expected=sum(sum(scores['distortion_cap'][n][metric]-scores[control][n][metric] for n in group)/len(group) for group in groups.values())/4
            assert abs(comparison['equal_specimen_deltas'][label][key]-expected)<1e-12
        for metric in ('mean','p90','maximum'):
            deltas={n:scores['distortion_cap'][n][metric]-scores[control][n][metric] for n in names}
            actual=comparison['regressions'][label][metric]
            assert actual['denominator']==actual['compared']==25
            assert actual['improved']==sum(d<0 for d in deltas.values()) and actual['unchanged']==sum(d==0 for d in deltas.values()) and actual['regressed']==sum(d>0 for d in deltas.values())
            assert actual['adverse_cases']==sorted([dict(name=n,delta=d) for n,d in deltas.items() if d>0],key=lambda r:r['delta'],reverse=True)
    def stats(actual,values,memory=False):
        assert actual['recorded_count']==actual['denominator']==25 and actual['missing_count']==0
        assert actual['minimum']==min(values) and actual['maximum']==max(values) and abs(actual['mean']-sum(values)/25)<1e-8
        if memory:assert actual['total'] is None
        else:assert abs(actual['total']-sum(values))<1e-8
    for a,m in manifests.items():
        cost=comparison['costs'][a]
        for key,field in (('complete_calls','complete_call_seconds'),('peak_allocated_bytes','peak_allocated_bytes'),('gradients','gradient_steps')):
            stats(cost[key],[r[field] for r in m['rows']],key=='peak_allocated_bytes')
        if a!='fusion300':
            stats(cost['recorded_incumbent_plus_suffix_calls'],[old['complete_call_seconds']+new['complete_call_seconds'] for old,new in zip(manifests['fusion300']['rows'],m['rows'])])
    assert manifests['distortion_cap']['elapsed_seconds']>=sum(r['complete_call_seconds'] for r in manifests['distortion_cap']['rows'])
    stops=Counter(s['stop_reason'] for r in manifests['distortion_cap']['rows'] for s in r['stages'])
    assert dict(stops)==comparison['stop_reasons']==checked['stage_stop_counts']
    checked.update(comparison_independently_verified=True,cohorts=comparison['cohorts'],
        equal_specimen_deltas=comparison['equal_specimen_deltas'],regressions=comparison['regressions'],costs=comparison['costs'])
    (DIRECTORY/'independent_check.json').write_text(json.dumps(checked,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(comparison_verified=True,equal_specimen_deltas=checked['equal_specimen_deltas'],
        regressions={k:{metric:v['regressed'] for metric,v in row.items()} for k,row in checked['regressions'].items()},
        costs=checked['costs']),indent=2))


if __name__=='__main__':main()
