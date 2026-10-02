"""Completed same-prefix timed final-stage accuracy, objective and cost comparison."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fusion_helper',Path(__file__).with_name('fusion_comparison_t23.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,flatten,aggregate,summary=helper.read,helper.flatten,helper.aggregate,helper.summary
ARMS=('adam','data_metric');METHODS=('frozen300',)+ARMS
CONTRASTS=(('metric_minus_adam','data_metric','adam'),('metric_minus_frozen300','data_metric','frozen300'),
    ('adam_minus_frozen300','adam','frozen300'))


def build(directory):
    directory=Path(directory).resolve();outer=read(directory/'predictions.json')
    directories={'frozen300':ROOT/'match_fusion_all50_t23/fusion',**{a:directory/a for a in ARMS}}
    manifests={a:read(p/'predictions.json') for a,p in directories.items()}
    if (outer.get('prediction_complete') is not True or outer.get('attempt_denominator')!=50
            or outer.get('annotations_read') is not False or len(outer.get('prefixes',[]))!=25):
        raise ValueError('all50 attempts and25 common prefixes required before score comparison')
    names=[r['name'] for r in manifests['frozen300']['rows']]
    for m in manifests.values():
        if (m.get('prediction_complete') is not True or m.get('all50_terminal') is not True
                or m.get('annotations_read') is not False or [r['name'] for r in m['rows']]!=names
                or any(r.get('status') not in ('ok','failed') for r in m['rows'])):
            raise ValueError('complete ordered25 denominator required')
    scores={a:flatten(read(p/'miit_scores.json'),read(p/'existing_scores.json')) for a,p in directories.items()}
    identities=[(n,c) for n,c,_ in scores['frozen300']]
    if len(set(identities))!=25:raise ValueError('25 distinct directions required')
    for a in METHODS:
        if [(n,c) for n,c,_ in scores[a]]!=identities:raise ValueError('score identities differ')
        for prediction,(_,_,score) in zip(manifests[a]['rows'],scores[a]):
            if prediction['status']!='ok' and score['status']=='ok':raise ValueError('failed map scored as success')
    cohorts={}
    for cohort in ('miit','lung_all20','histo','rat_kidney'):
        groups={a:aggregate([r for _,c,r in scores[a] if c==cohort]) for a in METHODS}
        deltas={}
        for name,a,b in CONTRASTS:
            left,right=groups[a]['all_directions_canvas_pixels'],groups[b]['all_directions_canvas_pixels']
            deltas[name]=None if left is None or right is None else {k:left[k]-right[k] for k in left}
        cohorts[cohort]=dict(methods=groups,deltas=deltas)
    rows=[];costs={};objective=[]
    for i,(name,cohort) in enumerate(identities):
        metrics={}
        for key in ('mean','p90','maximum'):
            values={a:scores[a][i][2]['metrics']['canvas_pixels'][key] if scores[a][i][2]['status']=='ok' else None for a in METHODS}
            metrics[key]=dict(values=values,deltas={n:None if values[a] is None or values[b] is None else values[a]-values[b]
                for n,a,b in CONTRASTS})
        records={a:manifests[a]['rows'][i] for a in METHODS}
        rows.append(dict(name=name,cohort=cohort,metrics=metrics,status={a:r['status'] for a,r in records.items()}))
        objective.append(dict(name=name,cohort=cohort,arms={a:dict(status=r['status'],
            initial=r.get('initial'),final=r.get('final'),prefix_best_total=r.get('prefix_best_total'),
            suffix_start_total=r.get('suffix_start_total'),selected_stage=r.get('selected_stage'),
            suffix_counts=r.get('suffix_counts'),stages=r.get('stages')) for a,r in records.items()}))
    for a,manifest in manifests.items():
        records=manifest['rows'];peak=summary([r.get('peak_allocated_bytes') for r in records],25);peak['total']=None
        costs[a]=dict(complete_suffix_calls=summary([r.get('complete_call_seconds') for r in records],25),peak_allocated_bytes=peak,
            suffix_gradient_steps=summary([r.get('suffix_gradient_steps') for r in records],25),
            stage_stops=dict(Counter(s['stop_reason'] for r in records for s in r.get('stages',[]) if 'stop_reason' in s)),
            per_pair=[dict(name=r['name'],status=r['status'],complete_call_seconds=r.get('complete_call_seconds'),
                peak_allocated_bytes=r.get('peak_allocated_bytes'),minimum_corner_ratio=r.get('actual_minimum_corner_ratio')) for r in records])
    equal={name:None if any(c['deltas'][name] is None for c in cohorts.values()) else {
        key:statistics.mean(c['deltas'][name][key] for c in cohorts.values()) for key in ('mean_pair_mean','mean_pair_p90')}
        for name,_,_ in CONTRASTS}
    return dict(scope='four repeatedly viewed specimens;25correlated directions;not heldout or25patients',
        units='512 moving-canvas pixels',cohorts=cohorts,rows=rows,objectives=objective,costs=costs,
        equal_specimen_deltas=equal,current_batch_wall_seconds=outer.get('elapsed_seconds'),
        common_prefix_calls=summary([p.get('complete_call_seconds') for p in outer['prefixes']],25),
        cost_scope='Each timed arm requires common240-gradient prefix PLUS its complete suffix call; prefix computed once perpair and shared experimentally. frozen300 includes its own prefix. Initializer and frozen matcher extraction remain additional historical costs. Peaks are not additive. Historical frozen300 is not synchronized speed control.',
        interpretation='Same saved prefix and same objective; distinct optimizer packages/acceptance mechanisms, not an isolated Hessian ablation. Timed early stops remain visible, not stationarity claims. Lower objective need not improve anatomy.',
        prediction_manifest=str(directory/'predictions.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();result=build(args.directory);destination=args.directory/'comparison.json'
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cohorts=result['cohorts'],equal_specimen_deltas=result['equal_specimen_deltas'],
        common_prefix_calls=result['common_prefix_calls'],costs={a:{k:v[k] for k in ('complete_suffix_calls','peak_allocated_bytes','stage_stops')} for a,v in result['costs'].items()}),indent=2))


if __name__=='__main__':main()
