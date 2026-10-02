"""One post-hoc information probe; extraction never reads annotation coordinates.

The unchanged released matrix is not a calibrated probability. No registration,
training, candidate selection or map modification is performed by this module.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np


def footprint(unit):
    u=np.asarray(unit,dtype=np.float64)
    if u.shape!=(2,) or not np.isfinite(u).all():raise ValueError('finite coordinate pair required')
    x,y=(512*u-.5)/8
    if min(x,y)<0 or max(x,y)>63:return None
    i,j=min(int(np.floor(x)),62),min(int(np.floor(y)),62)
    a,b=x-i,y-j
    return np.array([j*64+i,j*64+i+1,(j+1)*64+i,(j+1)*64+i+1]),np.array([(1-a)*(1-b),a*(1-b),(1-a)*b,a*b])


def unbatch_vertices(vertices):
    if vertices.shape!=(1,257,257,2) or vertices.dtype!=np.float64 or not np.isfinite(vertices).all():
        raise ValueError('saved batch-one257float64 vertex map required')
    return vertices[0]


def compare_targets(matrix,source,true_target,prediction):
    if matrix.shape!=(4096,4096):raise ValueError('full4096 square confidence required')
    feet=[footprint(p) for p in (source,true_target,prediction)]
    support=dict(zip(('source','true_target','prediction'),[f is not None for f in feet]))
    result=dict(support=support,supported=all(support.values()),bilinear=None,footprint_max=None)
    if not result['supported']:return result
    (indices,weight),truth,predicted=feet
    row=np.einsum('i,ij->j',weight,np.asarray(matrix[indices],dtype=np.float64))
    if not np.isfinite(row).all() or row.min()<0 or row.max()>1:raise ValueError('invalid confidence row')
    for mode in ('bilinear','footprint_max'):
        def sample(f):
            ids,w=f
            return float(row[ids]@w if mode=='bilinear' else row[ids].max())
        sT,sF=sample(truth),sample(predicted)
        rT,rF=1+int((row>sT).sum()),1+int((row>sF).sum())
        result[mode]=dict(true_score=sT,predicted_score=sF,true_rank=rT,predicted_rank=rF,
            normalized_rank_advantage=(rF-rT)/4096,
            ordering='true_above_prediction' if sT>sF else 'prediction_above_true' if sF>sT else 'tie',
            true_zero=sT==0,prediction_zero=sF==0)
    return result


def extract(args):
    import torch
    from tools.coordinated_joint_pose_batch import source_cases
    from tools.coordinated_matchanything import FrozenMatchAnything
    cases=source_cases(args.fusion_predictions)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    started=time.perf_counter()
    matcher=FrozenMatchAnything(args.source_root,args.checkpoint,device='cuda')
    report=dict(protocol='unchanged original-affine MA full confidence information probe',
        scope='image-only extraction; no labels or registration',annotations_read=False,
        source_fusion_predictions=str(args.fusion_predictions),model_setup=matcher.setup_report,
        cohort_size=25,rows=[],all25_terminal=False)
    def persist():
        report['elapsed_seconds']=time.perf_counter()-started
        (args.output/'extraction.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        for case in cases:
            row=dict(name=case['name'],cohort=case['cohort'],status='pending')
            cfg=case['original_configuration'];destination=args.output/(case['name']+'_C.npy')
            captured=[]
            def save_matrix(module,inputs,outputs):
                C=inputs[0]['conf_matrix']
                if C.shape!=(1,4096,4096) or C.dtype!=torch.float32:raise ValueError('unexpected coarse matrix')
                values=C.detach().cpu().numpy()[0]
                if not np.isfinite(values).all() or values.min()<0 or values.max()>1:raise ValueError('invalid full matrix')
                np.save(destination,values,allow_pickle=False);captured.append(tuple(C.shape))
            hook=matcher.model.register_forward_hook(save_matrix)
            tick=time.perf_counter()
            try:
                table=matcher.extract(Path(cfg['fixed']),Path(cfg['moving']),Path(cfg['affine']),
                    output=args.output/(case['name']+'_points.json'))
                if captured!=[(1,4096,4096)]:raise ValueError('exactly one unchanged model forward required')
                fused=json.loads(Path(cfg['matches']).read_text())
                old=json.loads(Path(fused['original_ma_table']).read_text())
                same={key:np.array_equal(np.asarray(table[key]),np.asarray(old[key]))
                    for key in ('source_points_unit','target_points_unit','confidence','post_affine_matrix','post_affine_offset')}
                row.update(status='ok',matrix=destination.name,matrix_bytes=destination.stat().st_size,
                    original_points_bitwise_equal=same,point_count=table['raw_matches'],
                    fixed=str(cfg['fixed']),moving=str(cfg['moving']),affine=str(cfg['affine']),
                    peak_allocated_bytes=table['peak_allocated_bytes'])
            except Exception as error:row.update(status='failed',error=f'{type(error).__name__}: {error}')
            finally:hook.remove()
            row['extraction_and_matrix_save_seconds']=time.perf_counter()-tick
            report['rows'].append(row);persist();print(json.dumps(row),flush=True)
    finally:matcher.close()
    report['all25_terminal']=len(report['rows'])==25;persist()
    return report


def summarize(records):
    valid=[r for r in records if r['supported']]
    result=dict(denominator=len(records),supported_count=len(valid),unsupported_count=len(records)-len(valid),
        unsupported_by_coordinate={k:sum(not r['support'][k] for r in records) for k in ('source','true_target','prediction')})
    for mode in ('bilinear','footprint_max'):
        rows=[r[mode] for r in valid]
        result[mode]=dict(computed_count=len(rows),
            mean_rank_advantage=statistics.mean(r['normalized_rank_advantage'] for r in rows) if rows else None,
            true_preferred=sum(r['ordering']=='true_above_prediction' for r in rows),
            prediction_preferred=sum(r['ordering']=='prediction_above_true' for r in rows),
            ties=sum(r['ordering']=='tie' for r in rows),
            true_zeros=sum(r['true_zero'] for r in rows),prediction_zeros=sum(r['prediction_zero'] for r in rows))
    result['ordering_disagreements']=sum(r['bilinear']['ordering']!=r['footprint_max']['ordering'] for r in valid)
    return result


def cohort_summary(rows,denominators):
    def group(name):return 'miit' if name.startswith('miit_') else name if name in ('histo','rat_kidney') else 'lung_all20'
    output={}
    for name in ('miit','lung_all20','histo','rat_kidney'):
        expected={k:v for k,v in denominators.items() if group(k)==name}
        selected=[r for r in rows if group(r['name'])==name]
        complete=len(selected)==len(expected) and all(r['summary']['supported_count']>0 for r in selected)
        output[name]=dict(case_denominator=len(expected),computed_cases=len(selected),denominator=sum(expected.values()),
            computed_case_ID_count=sum(r['summary']['denominator'] for r in selected),
            supported_count=sum(r['summary']['supported_count'] for r in selected),
            bilinear_mean_case_rank_advantage=statistics.mean(r['summary']['bilinear']['mean_rank_advantage'] for r in selected) if complete and selected else None,
            footprint_mean_case_rank_advantage=statistics.mean(r['summary']['footprint_max']['mean_rank_advantage'] for r in selected) if complete and selected else None)
    return output


def score(args):
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'outputs/coordinated_instance_registration/check_sources'))
    from independent_stain_proxy_probe_20261002 import original_landmark_case,literal_p1
    extracted=json.loads((args.output/'extraction.json').read_text())
    if not extracted['all25_terminal'] or len(extracted['rows'])!=25:raise ValueError('all25 attempts must be terminal')
    destination=args.output/'information_diagnostic.json'
    if destination.exists():raise FileExistsError(destination)
    archive=root/'outputs/coordinated_instance_registration/match_fusion_all50_t23/fusion'
    rows=[];all_points=[];failures=[];denominators={}
    for entry in extracted['rows']:
        name=entry['name'];layout,points,ids=original_landmark_case(name);denominators[name]=len(ids)
        unit=lambda xy,l:((xy+.5)*np.asarray(l['effective_original_to_canvas_scale_xy'])+np.asarray(l['padding_xy']))/512
        q,t=unit(points['fixed'],layout['fixed']),unit(points['moving'],layout['moving'])
        if entry['status']!='ok':
            failures.append(dict(name=name,denominator=len(ids),labels=ids,error=entry.get('error')));continue
        if not all(entry['original_points_bitwise_equal'].values()):raise ValueError('unchanged inference did not reproduce original MA point table')
        with np.load(archive/(name+'_analytic.npz'),allow_pickle=False) as z:
            vertices=unbatch_vertices(z['vertices']);A=z['post_affine_matrix'].astype(float);b=z['post_affine_offset'].astype(float)
            if str(z['interpolation'])!='p1_ac':raise ValueError('actual P1 ac map required')
        predicted=literal_p1(vertices,q)
        truth=np.linalg.solve(A,(t-b).T).T
        C=np.load(args.output/entry['matrix'],allow_pickle=False,mmap_mode='r')
        records=[]
        for label,qq,pp,tt,target in zip(ids,q,predicted,truth,t,strict=True):
            record=compare_targets(C,qq,tt,pp)
            record.update(label=label,source=qq.tolist(),true_target=tt.tolist(),prediction=pp.tolist(),
                current_TRE_canvas_pixels=float(np.linalg.norm(pp@A.T+b-target)*512))
            records.append(record)
        summary=summarize(records);all_points.extend(records)
        rows.append(dict(name=name,cohort=entry['cohort'],summary=summary,records=records))
    cohorts=cohort_summary(rows,denominators)
    result=dict(scope='POSTHOC development information diagnostic; no optimization, map change, model training, target subset or parameter selection',
        confidence='dual-softmax scores, NOT calibrated probabilities',rank='1+number of strictly larger among ALL4096 target cells',
        interpolation='source-row bilinear then target bilinear; explicit four-corner maximum sensitivity; unsupported never clamped',
        original_available_ID_denominator=sum(denominators.values()),
        failed_cases=failures,summary=summarize(all_points),cohorts=cohorts,rows=rows)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cohorts=cohorts,summary=result['summary'],failures=failures),indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('extract','score'))
    parser.add_argument('--output',type=Path,required=True)
    for key in ('fusion-predictions','source-root','checkpoint'):parser.add_argument('--'+key,type=Path)
    args=parser.parse_args()
    if args.mode=='extract':extract(args)
    else:score(args)
