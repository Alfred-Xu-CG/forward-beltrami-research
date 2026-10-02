"""Same-label native STANDARD initialization comparison and retained fusion300."""
import importlib.util
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_shared_fusion_helper',Path(__file__).with_name('fusion_comparison_t23.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,flatten,aggregate,summary=helper.read,helper.flatten,helper.aggregate,helper.summary
METHODS=('native_own_init','native_shared_init','fusion300')
CONTRASTS=(('shared_minus_own','native_shared_init','native_own_init'),
    ('shared_minus_fusion','native_shared_init','fusion300'),('fusion_minus_own','fusion300','native_own_init'))


def native_flat(miit,existing):
    return [(r['name'],'miit',r) for r in miit['rows']]+[
        (r['name'],'lung_all20' if i<20 else 'histo' if i==20 else 'rat_kidney',r) for i,r in enumerate(existing['rows'])]


def build():
    new=BASE/'native_standard_shared25_t27';fusion=BASE/'match_fusion_all50_t23/fusion'
    old_miit=BASE/'miit_dhr_released_standard_t19';old_existing=BASE/'native22_dhr_standard_t20'
    predictions={'native_shared_init':read(new/'predictions.json'),
        'native_own_init':dict(rows=read(old_miit/'predictions.json')['rows']+read(old_existing/'predictions.json')['rows']),
        'fusion300':read(fusion/'predictions.json')}
    newp=predictions['native_shared_init']
    if not newp.get('all25_terminal') or not newp.get('prediction_complete') or newp.get('annotations_read') is not False:
        raise ValueError('all25 image-only attempts required before saved scores')
    scores={'native_shared_init':native_flat(read(new/'miit_scores.json'),read(new/'existing_scores.json')),
        'native_own_init':native_flat(read(old_miit/'landmark_scores_complete.json'),read(old_existing/'landmark_scores.json')),
        'fusion300':flatten(read(fusion/'miit_scores.json'),read(fusion/'existing_scores.json'))}
    names=[r['name'] for r in newp['rows']]
    if len(names)!=25 or len(set(names))!=25:raise ValueError('exact25 directions required')
    for a in METHODS:
        if [n for n,_,_ in scores[a]]!=names or [r['name'] for r in predictions[a]['rows']]!=names:
            raise ValueError('identical ordered predictions/scores required')
    cohorts={}
    for c,count in dict(miit=3,lung_all20=20,histo=1,rat_kidney=1).items():
        groups={a:aggregate([r for _,cohort,r in scores[a] if cohort==c]) for a in METHODS}
        if any(v['pair_denominator']!=count for v in groups.values()):raise ValueError('fixed specimen counts required')
        deltas={}
        for label,a,b in CONTRASTS:
            left,right=groups[a]['all_directions_canvas_pixels'],groups[b]['all_directions_canvas_pixels']
            deltas[label]=None if left is None or right is None else {k:left[k]-right[k] for k in left}
        cohorts[c]=dict(methods=groups,deltas=deltas)
    rows=[]
    for i,name in enumerate(names):
        methods={a:scores[a][i][2] for a in METHODS};metrics={}
        good=[r for r in methods.values() if r['status']=='ok']
        if good and any(sorted(r['metrics']['canvas_pixels']['per_label'])!=sorted(good[0]['metrics']['canvas_pixels']['per_label']) for r in good):
            raise ValueError('manual landmark IDs differ across methods')
        for a,r in methods.items():
            if predictions[a]['rows'][i]['status']!='ok' and r['status']=='ok':raise ValueError('failed prediction scored successfully')
        for metric in ('mean','p90','maximum'):
            values={a:r['metrics']['canvas_pixels'][metric] if r['status']=='ok' else None for a,r in methods.items()}
            metrics[metric]=dict(values=values,deltas={label:None if values[a] is None or values[b] is None else values[a]-values[b]
                for label,a,b in CONTRASTS})
        rows.append(dict(name=name,cohort=scores['fusion300'][i][1],metrics=metrics,status={a:r['status'] for a,r in methods.items()},
            scored_landmarks=len(good[0]['metrics']['canvas_pixels']['per_label']) if good else None,
            native_topology={a:r.get('topology') for a,r in methods.items() if a!='fusion300'},
            initializer_audit=newp['rows'][i]['initial_field_audit']))
    regressions={}
    for label,_,_ in CONTRASTS:
        regressions[label]={}
        for metric in ('mean','p90','maximum'):
            available=[(r['name'],r['metrics'][metric]['deltas'][label]) for r in rows if r['metrics'][metric]['deltas'][label] is not None]
            regressions[label][metric]=dict(denominator=25,compared=len(available),improved=sum(d<0 for _,d in available),
                equal=sum(d==0 for _,d in available),regressed=sum(d>0 for _,d in available),
                regression_rows=sorted([dict(name=n,delta=d) for n,d in available if d>0],key=lambda r:r['delta'],reverse=True))
    costs={};topology={}
    for a in METHODS:
        records=predictions[a]['rows'];peaks=summary([r.get('peak_allocated_bytes') for r in records],25);peaks['total']=None
        costs[a]=dict(complete_calls=summary([r.get('complete_call_seconds') for r in records],25),peak_allocated_bytes=peaks,
            phase_seconds={k:summary([r.get(k) for r in records],25) for k in
                ('initial_seconds','nonrigid_seconds','preprocessing_seconds','registration_seconds')})
        if a!='fusion300':
            pairs=[r for _,_,r in scores[a] if r['status']=='ok']
            topology[a]=dict(pair_denominator=25,diagnosed_pairs=len(pairs),
                locally_nonpositive_pairs=sum(not r['topology']['local_orientation_valid'] for r in pairs),
                nonpositive_corners=sum(r['topology']['nonpositive_corners'] for r in pairs),
                checked_corners=sum(r['topology']['checked_corner_count'] for r in pairs),
                minimum_corner_ratio=min(r['topology']['minimum_corner_ratio'] for r in pairs),
                failures=[dict(name=r['name'],**r['topology']) for r in pairs if not r['topology']['local_orientation_valid']],
                global_homeomorphism_certified=False,boundary_injectivity_checked=False)
    audits=[r['initial_field_audit'] for r in newp['rows']];queries=[r['accepted512_query_audit'] for r in audits]
    initialization=dict(pair_denominator=25,full_preprocessed_nodes=sum(r['preprocessed_nodes'] for r in audits),
        full_node_max_normalized_error=max(r['maximum_normalized_error'] for r in audits),
        full_node_max_canvas_pixel_error=max(r['maximum_canvas_pixel_error'] for r in audits),
        all512_query_count=sum(r['query_count'] for r in queries),
        analytic_theta64_max_canvas_pixels=max(r['analytic_theta64_max_canvas_pixels'] for r in queries),
        cast_theta32_max_canvas_pixels=max(r['cast_theta32_max_canvas_pixels'] for r in queries),
        inside_max_canvas_pixels=max(r['literal_border_sample_inside_max_canvas_pixels'] for r in queries),
        outside_max_canvas_pixels=max(r['literal_border_sample_outside_max_canvas_pixels'] for r in queries if r['literal_border_sample_outside_max_canvas_pixels'] is not None),
        outside_query_count=sum(r['native_lattice_outside_count'] for r in queries),
        audit_seconds=sum(r['audit_seconds'] for r in audits),queries_dropped=0,
        scope='same geometric affine; float32 raster rounding retained; native-border clamping is not affine extrapolation through512 letterbox')
    return dict(scope='four repeatedly viewed development specimens;25 correlated directions, not independent patients',
        units='same512 moving-canvas pixels and exact same manual IDs per direction',cohorts=cohorts,rows=rows,
        regressions=regressions,costs=costs,native_topology=topology,initializer_consistency=initialization,
        batch_wall_seconds=newp.get('elapsed_seconds'),
        interpretation='Native shared-init is a supporting modified-initialization STANDARD counterfactual, not the untouched published pipeline. It keeps native preprocessing/NCC/diffusion/900steps and substitutes only the frozen SG affine; fusion300 uses different evidence/objective/discretization/boundary constraints. Changes versus own-init estimate initializer sensitivity; comparisons do not isolate topology or optimizer superiority. Both native arms lack a global topology guarantee; local native-grid corners do not check boundary injectivity or certify the257 P1 class. Prior fusion300 certified-map evidence is separate, not inferred from native diagnostics.',
        cost_scope='Native complete calls include field export but no whole-slide raster rendering. Shared-native complete calls include recorded initializer audits, excluding the historical supplied SG computation; own-init includes its matcher. Fusion300 calls exclude historical affine/SG/MA extraction. No complete cold end-to-end speed ratio. Allocated peaks are maxima, not additive; phase timers are nested. No posthoc repair, label-driven selection or omission.',
        sources=dict(native_shared=str(new),native_own_miit=str(old_miit),native_own_existing=str(old_existing),fusion=str(fusion),
            miit_comparison_layout=str(BASE/'miit_three_rotations_t153')))


if __name__=='__main__':
    value=build();destination=BASE/'native_standard_shared25_t27/comparison.json'
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('cohorts','costs','initializer_consistency','native_topology')},indent=2))
