"""Reaggregate saved scores/counts/costs after independent all50 map checks."""
from independent_data_metric_postrun_20261002 import BASE,ARCHIVE,ARMS,read
from collections import Counter
from pathlib import Path
import json

def main():
    directory=BASE/'data_metric_all50_t26r';report=read(directory/'independent_check.json')
    assert report['all50_saved_results_checked'] and report['total_landmark_errors']==4148
    checked=report['rows'];comparison=read(directory/'comparison.json');assert len(comparison['rows'])==25
    metric={a:{r['name']:r['arms'][a]['scores'] for r in checked} for a in ARMS}
    metric['frozen300']={r['name']:(r['methods']['analytic']['metrics'] if r['name'].startswith('miit_') else r['metrics'])['canvas_pixels'] for kind in ('miit','existing') for r in read(ARCHIVE/(kind+'_scores.json'))['rows']}
    pairs={'metric_minus_adam':('data_metric','adam'),'metric_minus_frozen300':('data_metric','frozen300'),'adam_minus_frozen300':('adam','frozen300')}
    groups={g:[r['name'] for r in comparison['rows'] if r['cohort']==g] for g in comparison['cohorts']}
    assert {g:len(n) for g,n in groups.items()}=={'miit':3,'lung_all20':20,'histo':1,'rat_kidney':1}
    for row in comparison['rows']:
        name=row['name'];assert all(v=='ok' for v in row['status'].values())
        for key in ('mean','p90','maximum'):
            for a in metric:assert abs(row['metrics'][key]['values'][a]-metric[a][name][key])<1e-12
            for label,(a,b) in pairs.items():assert abs(row['metrics'][key]['deltas'][label]-(metric[a][name][key]-metric[b][name][key]))<1e-12
    for group,names in groups.items():
        for a in metric:
            actual=comparison['cohorts'][group]['methods'][a];assert actual['pair_denominator']==actual['scored_pairs']==len(names) and actual['failure_count']==0
            for label,key in [('mean_pair_mean','mean'),('mean_pair_p90','p90'),('worst_pair_maximum','maximum')]:
                values=[metric[a][n][key] for n in names];expected=max(values) if key=='maximum' else sum(values)/len(values)
                assert abs(actual['all_directions_canvas_pixels'][label]-expected)<1e-12
        for label,(a,b) in pairs.items():
            for key in ('mean_pair_mean','mean_pair_p90','worst_pair_maximum'):
                expected=comparison['cohorts'][group]['methods'][a]['all_directions_canvas_pixels'][key]-comparison['cohorts'][group]['methods'][b]['all_directions_canvas_pixels'][key]
                assert abs(comparison['cohorts'][group]['deltas'][label][key]-expected)<1e-12
    for label,(a,b) in pairs.items():
        for key,m in [('mean_pair_mean','mean'),('mean_pair_p90','p90')]:
            expected=sum(sum(metric[a][n][m]-metric[b][n][m] for n in names)/len(names) for names in groups.values())/4
            assert abs(comparison['equal_specimen_deltas'][label][key]-expected)<1e-12
    def stats(actual,values,*,memory=False):
        assert actual['recorded_count']==actual['denominator']==25 and actual['missing_count']==0
        assert abs(actual['mean']-sum(values)/25)<1e-8 and actual['minimum']==min(values) and actual['maximum']==max(values)
        if memory:assert actual['total'] is None,'memory peaks are not additive'
        else:assert abs(actual['total']-sum(values))<1e-8
    outer=read(directory/'predictions.json');prefixes={r['name']:r for r in outer['prefixes']}
    stats(comparison['common_prefix_calls'],[r['complete_call_seconds'] for r in outer['prefixes']])
    assert comparison['current_batch_wall_seconds']==outer['elapsed_seconds']
    manifest={a:read(directory/a/'predictions.json') for a in ARMS};manifest['frozen300']=read(ARCHIVE/'predictions.json')
    source={a:{r['name']:r for r in m['rows']} for a,m in manifest.items()}
    for arm,m in manifest.items():
        costs=comparison['costs'][arm]
        stats(costs['complete_suffix_calls'],[r['complete_call_seconds'] for r in m['rows']])
        stats(costs['peak_allocated_bytes'],[r['peak_allocated_bytes'] for r in m['rows']],memory=True)
        for row in costs['per_pair']:
            original=source[arm][row['name']]
            assert row['complete_call_seconds']==original['complete_call_seconds'] and row['peak_allocated_bytes']==original['peak_allocated_bytes']
            assert row['minimum_corner_ratio']==original['actual_minimum_corner_ratio']
        if arm in ARMS:
            stats(costs['suffix_gradient_steps'],[r['suffix_gradient_steps'] for r in m['rows']])
            peaks=[max(r['peak_allocated_bytes'],read(directory/'common'/Path(prefixes[r['name']]['prefix_report']).name)['peak_allocated_bytes']) for r in m['rows']]
            stats(costs['maximum_recorded_prefix_suffix_peak'],peaks,memory=True)
            assert costs['stage_stops']==dict(Counter(s['stop_reason'] for r in m['rows'] for s in r['stages']))
    for row in comparison['objectives']:
        for arm,actual in row['arms'].items():
            original=source[arm][row['name']]
            for key,value in actual.items():
                assert value==original.get(key),(arm,row['name'],key)
    regressions={label:{key:[dict(name=n,delta=metric[a][n][key]-metric[b][n][key]) for n in metric[a] if metric[a][n][key]>metric[b][n][key]] for key in ('mean','p90','maximum')} for label,(a,b) in pairs.items()}
    report.update(comparison_aggregates_costs_checked=True,regressions=regressions,cohorts=comparison['cohorts'],equal_specimen_deltas=comparison['equal_specimen_deltas'],
        costs={a:{k:v for k,v in c.items() if k!='per_pair'} for a,c in comparison['costs'].items()},common_prefix_calls=comparison['common_prefix_calls'])
    (directory/'independent_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(comparison_aggregates_costs_checked=True,regression_counts={k:{m:len(v) for m,v in d.items()} for k,d in regressions.items()}),indent=2))

if __name__=='__main__':main()
