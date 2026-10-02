"""All-direction terminal-raster comparison after both prediction arms finish."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fusion_comparison',Path(__file__).with_name('fusion_comparison_t23.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,flatten,aggregate,summary=helper.read,helper.flatten,helper.aggregate,helper.summary
ARMS=('direct_original','lift512')
METHODS=('frozen512',)+ARMS
CONTRASTS=(('direct_minus_frozen512','direct_original','frozen512'),
    ('lift_minus_frozen512','lift512','frozen512'),('direct_minus_lift','direct_original','lift512'))


def build(directory):
    directory=Path(directory).resolve();outer=read(directory/'predictions.json')
    manifests={'frozen512':read(ROOT/'match_fusion_all50_t23/fusion/predictions.json'),
        **{arm:read(directory/arm/'predictions.json') for arm in ARMS}}
    if (outer.get('prediction_complete') is not True or outer.get('attempt_denominator')!=50
            or outer.get('annotations_read') is not False):raise ValueError('all50 terminal predictions required')
    expected=[r['name'] for r in manifests['frozen512']['rows']]
    if len(expected)!=25 or len(set(expected))!=25:raise ValueError('25 distinct directions required')
    for method,manifest in manifests.items():
        if (manifest.get('prediction_complete') is not True or manifest.get('all50_terminal') is not True
                or manifest.get('annotations_read') is not False or [r['name'] for r in manifest['rows']]!=expected
                or any(r.get('status') not in ('ok','failed') for r in manifest['rows'])):
            raise ValueError('same complete ordered25 denominator required')
    # Read ordinary landmark scores only after verifying terminal prediction state.
    scores={method:flatten(read(path/'miit_scores.json'),read(path/'existing_scores.json'))
        for method,path in [('frozen512',ROOT/'match_fusion_all50_t23/fusion')]+[(arm,directory/arm) for arm in ARMS]}
    identities=[(name,cohort) for name,cohort,_ in scores['frozen512']]
    for method in METHODS:
        if [(n,c) for n,c,_ in scores[method]]!=identities:raise ValueError('score identities differ')
        for record,(_,_,score) in zip(manifests[method]['rows'],scores[method]):
            if record['status']!='ok' and score['status']=='ok':raise ValueError('failed map cannot have successful score')
    cohorts={}
    for cohort in ('miit','lung_all20','histo','rat_kidney'):
        methods={m:aggregate([r for _,c,r in scores[m] if c==cohort]) for m in METHODS}
        deltas={}
        for label,left,right in CONTRASTS:
            a,b=methods[left]['all_directions_canvas_pixels'],methods[right]['all_directions_canvas_pixels']
            deltas[label]=None if a is None or b is None else {k:a[k]-b[k] for k in a}
        cohorts[cohort]=dict(methods=methods,deltas=deltas)
    rows=[]
    for i,(name,cohort) in enumerate(identities):
        records={m:scores[m][i][2] for m in METHODS};metrics={}
        for metric in ('mean','p90','maximum'):
            values={m:r['metrics']['canvas_pixels'][metric] if r['status']=='ok' else None for m,r in records.items()}
            metrics[metric]=dict(values=values,deltas={label:None if values[a] is None or values[b] is None
                else values[a]-values[b] for label,a,b in CONTRASTS})
        rows.append(dict(name=name,cohort=cohort,status={m:r['status'] for m,r in records.items()},metrics=metrics,
            failures={m:r.get('error') for m,r in records.items() if r['status']!='ok'}))
    costs={}
    for method,manifest in manifests.items():
        records=manifest['rows'];memory=summary([r.get('peak_allocated_bytes') for r in records],25);memory['total']=None
        costs[method]=dict(optimizer_complete_calls=summary([r.get('complete_call_seconds') for r in records],25),
            optimizer_peak_allocated_bytes=memory,arm_wall_seconds=manifest.get('elapsed_seconds'),
            per_pair=[dict(name=r['name'],status=r['status'],complete_call_seconds=r.get('complete_call_seconds'),
                peak_allocated_bytes=r.get('peak_allocated_bytes'),actual_minimum_corner_ratio=r.get('actual_minimum_corner_ratio'),
                selected_stage=r.get('selected_stage'),query_count=r.get('query_count')) for r in records])
    equal={label:None if any(c['deltas'][label] is None for c in cohorts.values()) else {
        k:statistics.mean(c['deltas'][label][k] for c in cohorts.values()) for k in ('mean_pair_mean','mean_pair_p90')}
        for label,_,_ in CONTRASTS}
    return dict(scope='four repeatedly viewed specimens,25 correlated directions,not25patients;one global recipe perarm',
        units='512 moving-canvas pixels; native errors remain in individual score files',
        methods=dict(frozen512='archived frozen-pose fused300 at512',direct_original='historical-source RGB direct1024 terminal',
            lift512='accepted512 float32 gray enlarged to1024 terminal'),
        cohorts=cohorts,rows=rows,equal_specimen_deltas=equal,costs=costs,
        current_batch_wall_seconds=outer.get('elapsed_seconds'),
        rendering=outer.get('rendering'),decoded_source_manifest=outer.get('decoded_source_manifest'),
        rendering_seconds=sum(r.get('preparation_seconds',0.) for r in outer.get('rendering',[])),
        cost_scope='Current optimizer calls include prepared array loading but exclude separately recorded rendering and historical RGB decode, initializer,SG and MA matching. Historical512 runs are not synchronized speed repeats. Peaks are not additive.',
        interpretation='Direct versus lift controls terminal raster density but also differs low-frequency resampling/quantization/padding; do not attribute all differences purely to high frequency. All maps have257-square controls and fixed original affine. Failed directions retain denominators and null full-cohort aggregates. No per-case arm selection.',
        prediction_manifest=str(directory/'predictions.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();result=build(args.directory);destination=args.directory/'comparison.json'
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cohorts=result['cohorts'],costs={k:{x:v[x] for x in ('optimizer_complete_calls','optimizer_peak_allocated_bytes')}
        for k,v in result['costs'].items()}),indent=2))


if __name__=='__main__':main()
