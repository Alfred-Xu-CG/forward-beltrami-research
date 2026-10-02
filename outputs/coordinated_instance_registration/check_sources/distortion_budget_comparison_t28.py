"""Saved all25 capped-distortion scores versus exact incoming and frozen suffix."""
import argparse
import importlib.util
import json
from pathlib import Path
import posixpath
import statistics

BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('distortion_fusion_helper',Path(__file__).with_name('fusion_comparison_t23.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,flatten,aggregate,summary=helper.read,helper.flatten,helper.aggregate,helper.summary
METHODS=('fusion300','frozen_suffix','distortion_cap')
COHORTS=dict(miit=3,lung_all20=20,histo=1,rat_kidney=1)


def data_value(parts):
    return parts['image']+parts['oob']+1e-4*parts['shape']+.2*parts['match']


def normalized_path(path):
    # Preserve remote POSIX identity when this report is assembled on Windows.
    return posixpath.normpath(str(path).replace('\\','/'))


def compare(directory):
    from tools.coordinated_distortion_budget_batch import validate_budget
    from tools.coordinated_dhr_existing_inputs import existing_rows
    directories=dict(fusion300=BASE/'match_fusion_all50_t23/fusion',
        frozen_suffix=BASE/'refreshed_match_all50_t27/frozen_suffix',distortion_cap=Path(directory))
    manifests={a:read(path/'predictions.json') for a,path in directories.items()}
    names=['miit_2_to_3','miit_7_to_8','miit_10_to_11']+[r['name'] for r in existing_rows(Path('unused'))]
    for arm,manifest in manifests.items():
        if (manifest.get('prediction_complete') is not True or manifest.get('annotations_read') is not False
                or [r.get('name') for r in manifest.get('rows',[])]!=names
                or any(r.get('status') not in ('ok','failed') for r in manifest['rows'])):
            raise ValueError('all exact25 terminal predictions required BEFORE scores')
        if arm=='distortion_cap' and (manifest.get('all25_terminal') is not True or manifest.get('budget_mode')!='distortion_cap'):
            raise ValueError('new cap protocol must be explicitly complete')
        if arm!='distortion_cap' and manifest.get('all50_terminal') is not True:
            raise ValueError('archived paired experiment must be complete')
    for i,name in enumerate(names):
        frozen=manifests['frozen_suffix']['rows'][i];cap=manifests['distortion_cap']['rows'][i]
        original=manifests['fusion300']['rows'][i]
        source=normalized_path(original['output'])
        if not (source.startswith('/') or (len(source)>1 and source[1]==':')):
            source=normalized_path(posixpath.join(posixpath.dirname(normalized_path(
                manifests['distortion_cap']['source_fusion_predictions'])),source))
        if (normalized_path(cap.get('incumbent',''))!=source
                or normalized_path(frozen.get('incumbent',''))!=source):
            raise ValueError('both suffixes must retain the same full saved incoming map')
        for row in (frozen,cap):
            if row['status']=='ok' and normalized_path((row.get('initial_map') or {}).get('path',''))!=source:
                raise ValueError('actual suffix initial map differs from the archived fusion output')
        if cap['status']=='ok':
            report=read(directories['distortion_cap']/cap['report']);validate_budget(report)
            if any(report.get(key)!=cap.get(key) for key in
                    ('initial_map','counts','initial','final','distortion_budget','gradient_steps')):
                raise ValueError('actual capped prediction/report discrepancy')
    scores={a:flatten(read(path/'miit_scores.json'),read(path/'existing_scores.json')) for a,path in directories.items()}
    for arm,records in scores.items():
        if [name for name,_,_ in records]!=names:
            raise ValueError('same ordered25 scores required')
        if {c:sum(group==c for _,group,_ in records) for c in COHORTS}!=COHORTS:
            raise ValueError('same specimen grouping required')
        for prediction,(_,_,scored) in zip(manifests[arm]['rows'],records,strict=True):
            if scored.get('status') not in ('ok','failed'):
                raise ValueError('every saved score must be terminal, with failures retained')
            if prediction['status']!='ok' and scored['status']=='ok':
                raise ValueError('a failed prediction cannot score as successful')
    cohorts={}
    for cohort,count in COHORTS.items():
        values={a:aggregate([r for _,c,r in scores[a] if c==cohort]) for a in METHODS}
        deltas={}
        for control in ('fusion300','frozen_suffix'):
            cap=values['distortion_cap']['all_directions_canvas_pixels'];baseline=values[control]['all_directions_canvas_pixels']
            deltas['cap_minus_'+control]=None if cap is None or baseline is None else {k:cap[k]-baseline[k] for k in cap}
        cohorts[cohort]=dict(methods=values,deltas=deltas)
    rows=[]
    for i,name in enumerate(names):
        records={a:scores[a][i][2] for a in METHODS}
        available=[r for r in records.values() if r['status']=='ok']
        if any(sorted(r['metrics']['canvas_pixels']['per_label'])!=sorted(available[0]['metrics']['canvas_pixels']['per_label']) for r in available):
            raise ValueError('exact same anatomical IDs required')
        metrics={}
        for metric in ('mean','p90','maximum'):
            values={a:r['metrics']['canvas_pixels'][metric] if r['status']=='ok' else None for a,r in records.items()}
            metrics[metric]=dict(values=values,deltas={'cap_minus_'+a:None if values[a] is None or values['distortion_cap'] is None else values['distortion_cap']-values[a]
                for a in ('fusion300','frozen_suffix')})
        diagnostic={}
        for a in METHODS:
            prediction=manifests[a]['rows'][i]
            if prediction['status']=='ok':
                final=prediction['final'];initial=prediction['initial']
                diagnostic[a]=dict(final_D=data_value(final),initial_D=data_value(initial),
                    final_R=final['strain'],initial_R=initial['strain'],
                    original_E=final['original_total'] if a=='distortion_cap' else final['total'],
                    gradient_steps=prediction['gradient_steps'],selected_stage=prediction.get('selected_stage'))
        rows.append(dict(name=name,cohort=scores['fusion300'][i][1],metrics=metrics,objectives=diagnostic,
                         statuses={a:r['status'] for a,r in records.items()}))
    regressions={}
    for control in ('fusion300','frozen_suffix'):
        label='cap_minus_'+control;regressions[label]={}
        for metric in ('mean','p90','maximum'):
            values=[(r['name'],r['metrics'][metric]['deltas'][label]) for r in rows if r['metrics'][metric]['deltas'][label] is not None]
            regressions[label][metric]=dict(denominator=25,compared=len(values),improved=sum(v<0 for _,v in values),
                unchanged=sum(v==0 for _,v in values),regressed=sum(v>0 for _,v in values),
                adverse_cases=sorted([dict(name=n,delta=v) for n,v in values if v>0],key=lambda r:r['delta'],reverse=True))
    costs={}
    for a in METHODS:
        records=manifests[a]['rows']
        peak=summary([r.get('peak_allocated_bytes') for r in records],25);peak['total']=None
        costs[a]=dict(complete_calls=summary([r.get('complete_call_seconds') for r in records],25),
            peak_allocated_bytes=peak,gradients=summary([r.get('gradient_steps') for r in records],25))
        if a!='fusion300':
            totals=[r['complete_call_seconds']+old['complete_call_seconds']
                if r.get('complete_call_seconds') is not None and old.get('complete_call_seconds') is not None else None
                for r,old in zip(records,manifests['fusion300']['rows'],strict=True)]
            costs[a]['recorded_incumbent_plus_suffix_calls']=summary(totals,25)
    cap_rows=manifests['distortion_cap']['rows'];stages=[s for r in cap_rows for s in r.get('stages',[])]
    stops={name:sum(s['stop_reason']==name for s in stages) for name in sorted(set(s['stop_reason'] for s in stages))}
    equal={}
    for control in ('fusion300','frozen_suffix'):
        label='cap_minus_'+control
        equal[label]=None if any(c['deltas'][label] is None for c in cohorts.values()) else {
            k:statistics.mean(c['deltas'][label][k] for c in cohorts.values()) for k in ('mean_pair_mean','mean_pair_p90')}
    return dict(scope='four repeatedly viewed development specimens,25 correlated directions; no held-out or SOTA claim',
        units='same512 moving-canvas pixels, exact same available IDs',cohorts=cohorts,rows=rows,
        regressions=regressions,equal_specimen_deltas=equal,costs=costs,stop_reasons=stops,
        failed_attempts=[dict(name=r['name'],error=r.get('error')) for r in cap_rows if r['status']!='ok'],
        cost_scope='suffix plus required archived incumbent; old initializer/SG/MA preparation remains additional and unmeasured scopes are not free; unknown recorded calls remain null rather than zero; no sum of memory peaks',
        count_scope='maximum300 NEW outersteps; two VJPs perused step, honest early stops; control300actual steps, not equal total work',
        numerical_claim='fixed runtime incumbent ARAP cap and own full-D selector; no proof of constrained stationarity or anatomical correctness',
        objective_scope='D is assembled directly from the unchanged image/OOB/shape/point parts, not original E minus3R. Original E remains a separate diagnostic. The constrained arm selects own full D under its cap; the frozen suffix selects original E. This is an objective/optimizer/selection package comparison, not same-objective convergence or equal-work evidence.',
        sources={a:str(path) for a,path in directories.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();destination=args.directory/'comparison.json'
    if destination.exists():raise FileExistsError(destination)
    result=compare(args.directory)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cohorts=result['cohorts'],equal_specimen_deltas=result['equal_specimen_deltas'],
        failures=result['failed_attempts'],stops=result['stop_reasons']),indent=2))
