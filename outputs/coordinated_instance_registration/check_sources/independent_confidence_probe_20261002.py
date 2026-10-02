"""Literal coarse-grid confidence sampling; never training or registration."""
import json
import math
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]


def coarse_foot(unit):
    pixel=np.asarray(unit,dtype=float)*512-.5
    if not np.isfinite(pixel).all():raise ValueError('finite unit pair required')
    if np.any(pixel<0) or np.any(pixel>504):return None
    i,j=[min(math.floor(v/8),62) for v in pixel]
    tx,ty=(pixel-np.array([8*i,8*j]))/8
    ids=np.array([[j*64+i,j*64+i+1],[(j+1)*64+i,(j+1)*64+i+1]])
    weights=np.outer([1-ty,ty],[1-tx,tx])
    return ids,weights


def literal_compare(C,q,t,p):
    f=[coarse_foot(u) for u in (q,t,p)]
    support={key:v is not None for key,v in zip(('source','true_target','prediction'),f)}
    result=dict(support=support,supported=all(support.values()),bilinear=None,footprint_max=None)
    if not result['supported']:return result
    source,truth,predicted=f
    row=np.zeros(4096)
    for j in range(2):
        for i in range(2):row+=source[1][j,i]*C[source[0][j,i]].astype(float)
    assert np.isfinite(row).all() and 0<=row.min()<=row.max()<=1
    for mode in ('bilinear','footprint_max'):
        values=[]
        for ids,w in (truth,predicted):
            samples=row[ids]
            values.append(float(sum(samples.ravel()*w.ravel())) if mode=='bilinear' else float(samples.max()))
        sT,sF=values
        rT=1+sum(float(v)>sT for v in row);rF=1+sum(float(v)>sF for v in row)
        result[mode]=dict(true_score=sT,predicted_score=sF,true_rank=rT,predicted_rank=rF,
            normalized_rank_advantage=(rF-rT)/4096,
            ordering='true_above_prediction' if sT>sF else 'prediction_above_true' if sF>sT else 'tie',
            true_zero=sT==0,prediction_zero=sF==0)
    return result


def compare_records(expected,actual,tolerance=3e-15):
    assert expected['support']==actual['support'] and expected['supported']==actual['supported']
    error=0.
    for mode in ('bilinear','footprint_max'):
        if not expected['supported']:assert actual[mode] is None;continue
        for key,value in expected[mode].items():
            if key in ('true_score','predicted_score'):
                difference=abs(value-actual[mode][key]);error=max(error,difference)
                assert difference<tolerance,(mode,key,difference)
            else:assert value==actual[mode][key],(mode,key,value,actual[mode][key])
    return error


def tiny_check():
    from tools.coordinated_match_confidence_probe import footprint,compare_targets
    unit=lambda x,y:(np.array([x,y])*8+.5)/512
    samples=[(0,0),(63,63),(0,63),(63,0),(4.25,5.75),(11,29),(63,12.25)]
    for x,y in samples:
        e=coarse_foot(unit(x,y));a=footprint(unit(x,y))
        assert np.array_equal(e[0].ravel(),a[0]) and np.array_equal(e[1].ravel(),a[1])
    for x,y in [(-1e-8,3),(63+1e-8,3),(8,-1e-8),(8,63+1e-8)]:assert footprint(unit(x,y)) is None and coarse_foot(unit(x,y)) is None
    C=np.zeros((4096,4096),dtype=np.float32)
    rng=np.random.default_rng(721)
    for qx,qy in ((0.,0.),(.375,.8125),(63.,63.),(7.125,12.375)):
        ids,_=coarse_foot(unit(qx,qy));C[ids.ravel()]=rng.uniform(0,1,(4,4096)).astype(np.float32)
        for tx,ty,px,py in ((.6,.3,12.4,19.2),(63.,63.,0.,0.),(12.,21.,12.,21.),(64.,3.,17.,4.)):
            q,t,p=unit(qx,qy),unit(tx,ty),unit(px,py)
            compare_records(literal_compare(C,q,t,p),compare_targets(C,q,t,p))
    C[:]=0
    tied=compare_targets(C,unit(4,7),unit(2,1),unit(60,62))
    compare_records(literal_compare(C,unit(4,7),unit(2,1),unit(60,62)),tied)
    assert tied['bilinear']['true_rank']==tied['bilinear']['predicted_rank']==1
    return dict(literal_grid_source_target_weights=True,closed_hull_no_clamp=True,
        all4096_strict_rank_and_zero_ties=True,zero_weight_footprint_corners_in_max_explicit=True)


def summary(records):
    selected=[r for r in records if r['supported']]
    result=dict(denominator=len(records),supported_count=len(selected),unsupported_count=len(records)-len(selected),
        unsupported_by_coordinate={k:sum(not r['support'][k] for r in records) for k in ('source','true_target','prediction')})
    for mode in ('bilinear','footprint_max'):
        values=[r[mode] for r in selected]
        result[mode]=dict(computed_count=len(values),mean_rank_advantage=sum(r['normalized_rank_advantage'] for r in values)/len(values) if values else None,
            true_preferred=sum(r['ordering']=='true_above_prediction' for r in values),
            prediction_preferred=sum(r['ordering']=='prediction_above_true' for r in values),
            ties=sum(r['ordering']=='tie' for r in values),true_zeros=sum(r['true_zero'] for r in values),
            prediction_zeros=sum(r['prediction_zero'] for r in values))
    result['ordering_disagreements']=sum(r['bilinear']['ordering']!=r['footprint_max']['ordering'] for r in selected)
    return result


def actual_check():
    from independent_stain_proxy_probe_20261002 import original_landmark_case
    from independent_joint_pose_probe_20261002 import vector_p1
    BASE=ROOT/'outputs/coordinated_instance_registration';directory=BASE/'full_confidence_all25_t28'
    read=lambda p:json.loads(Path(p).read_text())
    extraction=read(directory/'extraction.json');diagnostic=read(directory/'information_diagnostic.json')
    fusion=BASE/'match_fusion_all50_t23/fusion';predictions=read(fusion/'predictions.json')
    assert extraction['all25_terminal'] and len(extraction['rows'])==25 and extraction['annotations_read'] is False
    assert all(r['status']=='ok' for r in extraction['rows']) and not diagnostic['failed_cases']
    assert [r['name'] for r in extraction['rows']]==[r['name'] for r in predictions['rows']]==[r['name'] for r in diagnostic['rows']]
    oldmodel=read(BASE/'matchanything_all25_t22/predictions.json')['model_setup']
    assert extraction['model_setup']['source_revision']==oldmodel['source_revision']
    for key in ('model_config','strict_keys','state_dict_entries','parameter_count'):
        assert extraction['model_setup']['model'][key]==oldmodel['model'][key]
    all_records=[];case_results=[];max_score=max_coordinate=max_tre=0.;matrixbytes=0
    for entry,saved in zip(extraction['rows'],diagnostic['rows'],strict=True):
        name=entry['name'];layout,points,ids=original_landmark_case(name)
        assert [r['label'] for r in saved['records']]==ids
        new=read(directory/(name+'_points.json'));fused=read(BASE/'match_fusion_all50_t23/fused_tables'/(name+'.json'))
        oldpath=str(fused['original_ma_table']).replace('\\','/').split('/results/',1)[-1]
        if oldpath.startswith('results/'):oldpath=oldpath[8:]
        old=read(BASE/oldpath)
        for key in ('source_points_unit','target_points_unit','confidence','post_affine_matrix','post_affine_offset'):
            assert np.array_equal(new[key],old[key]) and entry['original_points_bitwise_equal'][key]
        assert new['raw_matches']==entry['point_count']
        C=np.load(directory/entry['matrix'],mmap_mode='r',allow_pickle=False)
        assert C.shape==(4096,4096) and C.dtype==np.float32 and np.isfinite(C).all() and 0<=C.min()<=C.max()<=1
        assert (directory/entry['matrix']).stat().st_size==entry['matrix_bytes'];matrixbytes+=entry['matrix_bytes']
        with np.load(fusion/(name+'_analytic.npz')) as z:
            V=z['vertices'][0].astype(float);A=z['post_affine_matrix'].astype(float);b=z['post_affine_offset'].astype(float)
            assert str(z['interpolation'])=='p1_ac'
        assert np.array_equal(new['post_affine_matrix'],A) and np.array_equal(new['post_affine_offset'],b)
        def unit(x,role):
            r=layout[role]
            return ((x+.5)*np.array(r['effective_original_to_canvas_scale_xy'])+np.array(r['padding_xy']))/512
        q=unit(points['fixed'],'fixed');t=unit(points['moving'],'moving')
        predicted=vector_p1(V,q)
        det=A[0,0]*A[1,1]-A[0,1]*A[1,0];assert det>0
        inverse=np.array([[A[1,1],-A[0,1]],[-A[1,0],A[0,0]]])/det
        truth=(t-b)@inverse.T
        case_records=[]
        for i,(label,source,real,prediction) in enumerate(zip(ids,q,truth,predicted,strict=True)):
            record=saved['records'][i]
            err=max(float(abs(source-record['source']).max()),float(abs(real-record['true_target']).max()),float(abs(prediction-record['prediction']).max()))
            max_coordinate=max(max_coordinate,err);assert err<5e-16
            expected=literal_compare(C,source,real,prediction)
            max_score=max(max_score,compare_records(expected,record,1e-12))
            TRE=float(np.linalg.norm(A@prediction+b-t[i])*512)
            difference=abs(TRE-record['current_TRE_canvas_pixels']);max_tre=max(max_tre,difference);assert difference<2e-12
            case_records.append(expected)
        expected_summary=summary(case_records)
        for key,value in expected_summary.items():
            if key in ('bilinear','footprint_max'):
                for k,v in value.items():
                    if k=='mean_rank_advantage':assert abs(v-saved['summary'][key][k])<1e-17
                    else:assert v==saved['summary'][key][k]
            else:assert value==saved['summary'][key]
        case_results.append(dict(name=name,summary=expected_summary));all_records.extend(case_records)
        print(json.dumps(dict(checked=name,IDs=len(ids),supported=expected_summary['supported_count'],maximum_score_error=max_score)),flush=True)
    assert len(all_records)==diagnostic['original_available_ID_denominator']==2074
    all_summary=summary(all_records)
    for key,value in all_summary.items():
        if key in ('bilinear','footprint_max'):
            for k,v in value.items():
                if k=='mean_rank_advantage':assert abs(v-diagnostic['summary'][key][k])<1e-17
                else:assert v==diagnostic['summary'][key][k]
        else:assert value==diagnostic['summary'][key]
    group=lambda n:'miit' if n.startswith('miit_') else n if n in ('histo','rat_kidney') else 'lung_all20'
    for cohort,count in dict(miit=3,lung_all20=20,histo=1,rat_kidney=1).items():
        selected=[r for r in case_results if group(r['name'])==cohort];actual=diagnostic['cohorts'][cohort]
        assert actual['case_denominator']==actual['computed_cases']==len(selected)==count
        assert actual['denominator']==actual['computed_case_ID_count']==sum(r['summary']['denominator'] for r in selected)
        assert actual['supported_count']==sum(r['summary']['supported_count'] for r in selected)
        for mode,key in (('bilinear','bilinear_mean_case_rank_advantage'),('footprint_max','footprint_mean_case_rank_advantage')):
            assert abs(actual[key]-sum(r['summary'][mode]['mean_rank_advantage'] for r in selected)/len(selected))<1e-17
    result=dict(all25_saved_C_checked=True,all_original_MA_point_arrays_bitwise_equal=True,
        all2074_original_IDs_coordinates_TRE_and_two_scores_ranks_orderings_checked=True,
        maximum_coordinate_error=max_coordinate,maximum_confidence_score_error=max_score,
        maximum_current_TRE_error=max_tre,saved_matrix_bytes_read=matrixbytes,
        unsupported_IDs_retained_never_clamped=True,cohort_and_full_summaries_independently_verified=True,
        summary=all_summary,cohorts=diagnostic['cohorts'],rows=case_results,
        limits='posthoc development diagnostic only; no training/map changes; unchanged point tables do not establish previously-unsaved full C was bitwise identical historically')
    (directory/'independent_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--actual',action='store_true');args=parser.parse_args()
    if args.actual:actual_check()
    else:print(json.dumps(tiny_check(),indent=2))
