"""Completed one-refresh comparison; saved scores only, no annotation readers."""
import argparse
import importlib.util
import json
from pathlib import Path
import posixpath
import statistics

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fusion_refresh_helper',Path(__file__).with_name('fusion_comparison_t23.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,flatten,aggregate,summary=helper.read,helper.flatten,helper.aggregate,helper.summary
ARMS=('frozen_suffix','refreshed_suffix');METHODS=('fusion300',)+ARMS
CONTRASTS=(('refresh_minus_frozen','refreshed_suffix','frozen_suffix'),
    ('refresh_minus_fusion300','refreshed_suffix','fusion300'),('frozen_minus_fusion300','frozen_suffix','fusion300'))
COHORTS=dict(miit=3,lung_all20=20,histo=1,rat_kidney=1)


def norm(path):
    # Saved paths may be remote POSIX paths while this report runs on Windows.
    return posixpath.normpath(str(path).replace('\\','/'))


def require_complete(outer,manifests):
    from tools.coordinated_dhr_existing_inputs import existing_rows
    names=['miit_2_to_3','miit_7_to_8','miit_10_to_11']+[r['name'] for r in existing_rows(Path('unused'))]
    if (outer.get('prediction_complete') is not True or outer.get('extraction_complete') is not True
            or outer.get('annotations_read') is not False or outer.get('attempt_denominator')!=50
            or [r.get('name') for r in outer.get('extractions',[])]!=names
            or any(r.get('status') not in ('ok','failed') for r in outer['extractions'])):
        raise ValueError('all25 extraction attempts and all50 suffix attempts must be terminal before scores')
    for method in METHODS:
        value=manifests[method]
        if (value.get('prediction_complete') is not True or value.get('all50_terminal') is not True
                or value.get('annotations_read') is not False or [r.get('name') for r in value.get('rows',[])]!=names
                or any(r.get('status') not in ('ok','failed') for r in value['rows'])):
            raise ValueError('same ordered declared25 terminal predictions required')
    for i,name in enumerate(names):
        original=manifests['fusion300']['rows'][i]
        if original['status']!='ok' or original.get('gradient_steps')!=300:
            raise ValueError('successful archived fusion300 incumbent required')
        source=norm(original['output'])
        if not (source.startswith('/') or (len(source)>1 and source[1]==':')):
            source=norm(posixpath.join(posixpath.dirname(norm(outer['source_fusion_predictions'])),source))
        for arm in ARMS:
            row=manifests[arm]['rows'][i]
            if norm(row.get('incumbent',''))!=source:
                raise ValueError('both suffixes must use the same pair-specific archived full saved map')
            if row['status']=='ok':
                if (norm((row.get('initial_map') or {}).get('path',''))!=source
                        or row.get('gradient_steps')!=300 or row.get('new_suffix_gradient_steps')!=300
                        or row.get('failed_trials')!=0):
                    raise ValueError('successful suffix must use actual incumbent and all300 new gradients')
                if arm=='refreshed_suffix' and outer['extractions'][i]['status']!='ok':
                    raise ValueError('failed refresh cannot become a successful refreshed prediction')
    return names


def peak_summary(values):
    value=summary(values,25);value['total']=None
    return value


def compare(outer,manifests,score_files,historical_outer):
    names=require_complete(outer,manifests)
    scores={a:flatten(*score_files[a]) for a in METHODS}
    identities=[(n,c) for n,c,_ in scores['fusion300']]
    if ([n for n,_ in identities]!=names or len(set(identities))!=25
            or {c:sum(k==c for _,k in identities) for c in COHORTS}!=COHORTS):
        raise ValueError('exact four-cohort score denominators required')
    for a in METHODS:
        if [(n,c) for n,c,_ in scores[a]]!=identities:raise ValueError('ordered score identities differ')
        for prediction,(_,_,scored) in zip(manifests[a]['rows'],scores[a]):
            if scored.get('status') not in ('ok','failed'):raise ValueError('terminal scoring status required')
            if prediction['status']!='ok' and scored['status']=='ok':raise ValueError('failed map cannot score as success')
    cohorts={}
    for cohort in COHORTS:
        groups={a:aggregate([r for _,c,r in scores[a] if c==cohort]) for a in METHODS}
        deltas={}
        for label,a,b in CONTRASTS:
            left,right=groups[a]['all_directions_canvas_pixels'],groups[b]['all_directions_canvas_pixels']
            deltas[label]=None if left is None or right is None else {k:left[k]-right[k] for k in left}
        cohorts[cohort]=dict(methods=groups,deltas=deltas)
    rows=[];objectives=[]
    for i,(name,cohort) in enumerate(identities):
        records={a:manifests[a]['rows'][i] for a in METHODS};metrics={}
        for key in ('mean','p90','maximum'):
            values={a:scores[a][i][2]['metrics']['canvas_pixels'][key] if scores[a][i][2]['status']=='ok' else None for a in METHODS}
            metrics[key]=dict(values=values,deltas={label:None if values[a] is None or values[b] is None else values[a]-values[b]
                for label,a,b in CONTRASTS})
        rows.append(dict(name=name,cohort=cohort,metrics=metrics,extraction=outer['extractions'][i],
            prediction_status={a:r['status'] for a,r in records.items()},score_status={a:scores[a][i][2]['status'] for a in METHODS},
            failures={a:dict(prediction=r.get('error'),scoring=scores[a][i][2].get('error')) for a,r in records.items()
                if r['status']!='ok' or scores[a][i][2]['status']!='ok'},
            incumbent=records['frozen_suffix']['incumbent']))
        own={}
        for a,r in records.items():
            initial,final=r.get('initial'),r.get('final')
            delta=None if not initial or not final or initial.get('total') is None or final.get('total') is None else final['total']-initial['total']
            own[a]=dict(status=r['status'],initial=initial,final=final,own_final_minus_initial=delta,
                selected_stage=r.get('selected_stage'),selected_stage_scope=r.get('selected_stage_scope'),
                initial_map=r.get('initial_map'),gradient_steps=r.get('gradient_steps'),failed_trials=r.get('failed_trials'),
                actual_minimum_corner_ratio=r.get('actual_minimum_corner_ratio'),saved_binary_certificate=r.get('saved_binary_certificate'))
        objectives.append(dict(name=name,cohort=cohort,methods=own))
    regressions={}
    for label,_,_ in CONTRASTS:
        regressions[label]={}
        for key in ('mean','p90','maximum'):
            pairs=[(r['name'],r['metrics'][key]['deltas'][label]) for r in rows]
            available=[(n,d) for n,d in pairs if d is not None]
            worse=sorted([dict(name=n,delta=d) for n,d in available if d>0],key=lambda x:x['delta'],reverse=True)
            regressions[label][key]=dict(pair_denominator=25,compared_count=len(available),unavailable_count=25-len(available),
                improved_count=sum(d<0 for _,d in available),unchanged_count=sum(d==0 for _,d in available),
                regressed_count=len(worse),regressed_pairs=worse)
    old=manifests['fusion300']['rows'];costs={}
    for a in METHODS:
        records=manifests[a]['rows']
        costs['incumbent300' if a=='fusion300' else a]=dict(
            complete_calls=summary([r.get('complete_call_seconds') for r in records],25),
            peak_allocated_bytes=peak_summary([r.get('peak_allocated_bytes') for r in records]),
            gradient_steps=summary([r.get('gradient_steps') if r['status']=='ok' else None for r in records],25),
            per_pair=[{k:r.get(k) for k in ('name','status','complete_call_seconds','peak_allocated_bytes','loading_seconds',
                'feature_seconds','optimize_seconds','end_to_end_seconds','serialization_seconds','certification_seconds','gradient_steps')} for r in records])
        if a in ARMS:
            costs[a]['recorded_incumbent_plus_suffix_calls']=summary([
                p['complete_call_seconds']+r['complete_call_seconds'] if p.get('complete_call_seconds') is not None
                and r.get('complete_call_seconds') is not None else None for p,r in zip(old,records)],25)
            costs[a]['maximum_recorded_incumbent_suffix_peak']=peak_summary([
                max(p['peak_allocated_bytes'],r['peak_allocated_bytes']) if p.get('peak_allocated_bytes') is not None
                and r.get('peak_allocated_bytes') is not None else None for p,r in zip(old,records)])
            if a=='refreshed_suffix':
                costs[a]['recorded_incumbent_suffix_refresh_calls']=summary([
                    p['complete_call_seconds']+r['complete_call_seconds']+e['complete_call_seconds']
                    if all(v.get('complete_call_seconds') is not None for v in (p,r,e)) else None
                    for p,r,e in zip(old,records,outer['extractions'])],25)
    extraction=outer['extractions'];setup=outer.get('model_setup') or {}
    phase_keys=('incumbent_loading_seconds','raster_loading_render_seconds','model_forward_seconds','point_adaptation_seconds','extraction_seconds')
    refresh_costs=dict(setup_status=outer.get('model_setup_status'),setup_complete_seconds=outer.get('model_setup_complete_seconds'),
        setup_peak_allocated_bytes=setup.get('setup_peak_allocated_bytes'),setup_report=setup,
        extraction_complete_calls=summary([r.get('complete_call_seconds') for r in extraction],25),
        extraction_peak_allocated_bytes=peak_summary([(r.get('extraction') or {}).get('peak_allocated_bytes') for r in extraction]),
        extraction_phase_times={key:summary([(r.get('extraction') or {}).get(key) for r in extraction],25) for key in phase_keys},
        model_released_before_optimization=outer.get('model_released_before_optimization'))
    equal={label:None if any(v['deltas'][label] is None for v in cohorts.values()) else {
        k:statistics.mean(v['deltas'][label][k] for v in cohorts.values()) for k in ('mean_pair_mean','mean_pair_p90')}
        for label,_,_ in CONTRASTS}
    return dict(scope='four repeatedly viewed development specimens;25 correlated directions, not25patients or heldout validation',
        units='512 moving-canvas pixels',pair_denominator=25,cohorts=cohorts,rows=rows,regressions=regressions,
        objectives=objectives,equal_specimen_deltas=equal,costs=costs,refresh_costs=refresh_costs,
        extraction_denominator=25,extraction_successful=sum(r['status']=='ok' for r in extraction),
        extraction_failed=sum(r['status']=='failed' for r in extraction),
        prediction_failures={a:sum(r['status']!='ok' for r in manifests[a]['rows']) for a in METHODS},
        historical_preparation=dict(unknown_initializer_and_sg_seconds=None,
            old_ma_setup_seconds=historical_outer.get('frozen_ma_historical_setup_seconds'),
            old_ma_cost_totals=historical_outer.get('frozen_ma_historical_cost_totals'),
            old_ma_predictions=historical_outer.get('frozen_ma_predictions'),
            old_fusion_composition_seconds=historical_outer.get('composition_seconds')),
        current_batch_wall_seconds=outer.get('elapsed_seconds'),
        cost_scope='Incumbent300 is required historical work, not a free initial map. Each suffix adds300 new gradients after its actual complete incumbent; refresh additionally requires one setup and25 extraction attempts. Complete extraction calls include table I/O/fusion; inner extraction phase times are nested, not additive to those calls. Unknown common initializer/SG and unmatched historical startup stay unmeasured; no full end-to-end speed ratio. Allocated peaks are maxima, never sums; old counters may omit pre-reset setup transients. Recorded incumbent+suffix totals exclude refresh and historical evidence preparation, and include failed-call cost.',
        objective_scope='Each arm selects the incumbent or an accepted suffix state by its OWN full objective. Refreshed MA changes E; own-final-minus-own-initial is diagnostic, not cross-functional convergence or anatomical correctness. No best-arm/per-case selection.',
        interpretation='The matched frozen suffix controls merely doing300 more gradients from the same full saved map. A failed refresh remains a failed direction; frozen continuation still runs. Any failed direction makes its full-cohort/equal-specimen aggregate undefined. The original fusion300 remains the cheaper reference; historical and current timings are not synchronized speed controls.')


def build(directory,*,fusion_directory=None):
    directory=Path(directory).resolve();fusion_directory=Path(fusion_directory or ROOT/'match_fusion_all50_t23/fusion').resolve()
    outer=read(directory/'predictions.json')
    directories={'fusion300':fusion_directory,**{a:directory/a for a in ARMS}}
    manifests={a:read(p/'predictions.json') for a,p in directories.items()}
    require_complete(outer,manifests)  # No new saved scores opened before all terminal checks.
    scores={a:(read(p/'miit_scores.json'),read(p/'existing_scores.json')) for a,p in directories.items()}
    historical_path=fusion_directory.parent/'predictions.json'
    historical=read(historical_path) if historical_path.is_file() else {}
    result=compare(outer,manifests,scores,historical)
    result.update(prediction_manifest=str(directory/'predictions.json'),archived_fusion_manifest=str(fusion_directory/'predictions.json'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--fusion-directory',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=build(args.directory,fusion_directory=args.fusion_directory)
    destination=args.output or args.directory/'comparison.json'
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(cohorts=result['cohorts'],equal_specimen_deltas=result['equal_specimen_deltas'],
        extraction_failed=result['extraction_failed']),indent=2))


if __name__=='__main__':main()
