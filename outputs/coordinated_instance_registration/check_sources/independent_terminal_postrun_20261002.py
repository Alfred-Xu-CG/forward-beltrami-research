"""Independent saved terminal artifacts, original labels and literal objectives.

Never runs optimization. Completion is checked before original CSV reads.
"""
import json
import sys
from pathlib import Path
from fractions import Fraction
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from independent_joint_pose_probe_20261002 import pixels,corners,bilinear,descriptor,vector_p1,numpy_priors
from independent_stain_proxy_probe_20261002 import literal_p1,original_landmark_case
from independent_terminal_detail_probe_20261002 import lift_oracle
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
BASE=ROOT/'outputs/coordinated_instance_registration'
DATA=Path('D:/QC_optimization_data/digital_topology_wsi')
DIRECTORY=BASE/'terminal_detail_all50_t25'
ARCHIVED=BASE/'match_fusion_all50_t23/fusion'
ARMS=('direct_original','lift512')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    outer=read(DIRECTORY/'predictions.json');manifests={a:read(DIRECTORY/a/'predictions.json') for a in ARMS}
    assert outer['prediction_complete'] and outer['attempt_denominator']==50 and not outer['annotations_read']
    assert len(outer['rendering'])==25 and all(r['status']=='ok' and not r['annotations_read'] for r in outer['rendering'])
    for arm,m in manifests.items():
        assert m['prediction_complete'] and m['all50_terminal'] and not m['annotations_read'] and len(m['rows'])==25
        assert all(r['status']=='ok' and r['budget_complete'] for r in m['rows'])
        for cohort in ('miit','existing'):
            sm=read(DIRECTORY/arm/(cohort+'_predictions.json'));assert sm['all50_terminal'] and sm['metric_image_side']==512
    # All50 terminal declarations and every actual required artifact exist BEFORE labels.
    assert all((DIRECTORY/a/r[k]).exists() for a,m in manifests.items() for r in m['rows'] for k in ('report','output'))
    scores={a:{r['name']:r for kind in ('miit','existing') for r in read(DIRECTORY/a/(kind+'_scores.json'))['rows']} for a in ARMS}
    old_manifest=read(ARCHIVED/'predictions.json');controls={r['name']:r for r in old_manifest['rows']}
    cache_manifest=read(DATA/'historical_rgb_decode_cache/manifest.json');cache={r['source_name']:r for r in cache_manifest['rows']}
    grid=pixels(1024);axis=np.arange(257)/256;yy,xx=np.meshgrid(axis,axis,indexing='ij');identity=np.stack((xx,yy),-1)
    exactdet=lambda a:Fraction(float(a[0,0]))*Fraction(float(a[1,1]))-Fraction(float(a[0,1]))*Fraction(float(a[1,0]))
    errors=dict(canvas=0.,native=0.,objective=0.,point=0.,outside=0.,lift=0.,lift_local_torch_replay=0.)
    image_cache={};render_cache={};rows=[];labels=0;lower_differences={}
    for arm,manifest in manifests.items():
        for row in manifest['rows']:
            name=row['name'];report=read(DIRECTORY/arm/row['report']);cfg=report['configuration'];old=read(ARCHIVED/controls[name]['report'])
            allowed={'image_side','image_levels','output'}
            assert {k for k in old['configuration'] if cfg[k]!=old['configuration'][k]}==allowed
            assert set(cfg)-set(old['configuration'])=={'terminal_source_image_side','terminal_evidence_arm','terminal_evidence_bundle','pyramid_source_image_side','point_pixel_scale'}
            assert report['image_match_evidence']==old['image_match_evidence']
            assert report['gradient_steps']==300 and report['objective_evaluations']==332 and report['failed_trials']==0
            assert len(report['stages'])==10 and len(report['trace'])==310
            assert report['query_count']==1024**2 and report['control_vertices']==257**2 and report['corner_constraints']==4*256**2
            assert report['pyramid_source_image_side']==report['point_pixel_scale']==512 and report['query_image_side']==1024
            assert report['image_levels']==[32,64,128,256,1024] and not report['landmarks_used']
            for kind,records,priors in [('stage',report['stages'][:8],old['stages'][:8]),('trace',report['trace'][:248],old['trace'][:248])]:
                for record,prior in zip(records,priors):
                    assert set(record)==set(prior)
                    for key,value in record.items():
                        if key=='accepted_full_total':continue
                        if isinstance(value,float):lower_differences[kind+'.'+key]=max(lower_differences.get(kind+'.'+key,0),abs(value-prior[key]))
                        else:assert value==prior[key]
            assert all(report['mind_frame_by_resolution'][str(n)]==old['mind_frame_by_resolution'][str(n)] for n in (32,64,128,256))
            assert report['selected_stage']==9 and report['final']['total']==min([report['initial']['total']]+[s['accepted_full_total'] for s in report['stages']])
            assert report['final']['total']==report['stages'][9]['accepted_full_total']
            saved_path=DIRECTORY/arm/row['output']
            with np.load(saved_path,allow_pickle=False) as saved:
                vertices=saved['vertices'][0];reference=saved['boundary_reference'][0];a=saved['post_affine_matrix'];b=saved['post_affine_offset']
                assert str(saved['interpolation'])=='p1_ac'
            with np.load(ARCHIVED/controls[name]['output'],allow_pickle=False) as original:
                assert np.array_equal(a,original['post_affine_matrix']) and np.array_equal(b,original['post_affine_offset'])
            assert vertices.dtype==reference.dtype==np.float64 and np.array_equal(reference,identity) and vertices.shape==(257,257,2)
            assert np.array_equal(vertices[[0,-1]],identity[[0,-1]]) and np.array_equal(vertices[:,[0,-1]],identity[:,[0,-1]])
            assert exactdet(a)>0 and certify_q1_binary_map(saved_path)['valid'] and report['saved_binary_certificate']['valid']
            minimum=float(corners(vertices).min());assert minimum>.001 and abs(minimum-row['actual_minimum_corner_ratio'])<3e-13
            with np.load(DIRECTORY/'rendered_inputs'/(name+'_'+arm+'.npz'),allow_pickle=False) as bundle:
                meta=json.loads(str(bundle['metadata']));arrays={k:bundle[k].copy() for k in ('fixed','moving','fixed_mask')}
            assert meta['arm']==arm and not meta['annotations_read'] and not meta['uniform_native1024_resolution']
            assert meta['point_pixel_scale']==meta['point_prediction_side']==meta['source_image_side']==512 and meta['point_robust_scale']==8 and meta['terminal_image_side']==1024
            for value in arrays.values():assert value.dtype==np.float32 and value.shape==(1,1,1024,1024)
            old_images={}
            for role in ('fixed','moving'):
                folder=BASE/'miit_three_rotations_t153' if name.startswith('miit_') else DATA/('birl_anhir_dev/canvas' if name in ('histo','rat_kidney') else 'lung_lesion3_eval/canvas')
                path=folder/Path(cfg[role]).name
                if path not in image_cache:image_cache[path]=1-np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255
                old_images[role]=image_cache[path];r=meta['roles'][role];n=np.array(r['accepted_resized_wh']);p=np.array(r['accepted_padding_xy']);wh=np.array(r['original_wh'])
                assert r['accepted512_reconstruction_exact'] and np.array_equal(r['terminal_resized_wh'],2*n) and np.array_equal(r['terminal_padding_xy'],2*p)
                assert np.array_equal(r['terminal_to_original_resize_ratio_xy'],2*n/wh) and r['upscaled_axes']==(2*n>wh).tolist()
                if arm=='lift512':
                    error=float(abs(lift_oracle(old_images[role])-arrays[role][0,0]).max());errors['lift']=max(errors['lift'],error);assert error<1.3e-7
                    replay=F.interpolate(torch.tensor(old_images[role][None,None]),size=(1024,1024),mode='bilinear',align_corners=False).numpy()
                    replay_error=float(abs(replay-arrays[role]).max());errors['lift_local_torch_replay']=max(errors['lift_local_torch_replay'],replay_error);assert replay_error<1.3e-7
                else:
                    decode=r['decoding']
                    if name.startswith('miit_'):
                        assert decode['mode']=='current_PIL_source_decode'
                        source=Path('D:/QC_optimization_data/miit_v4/extracted/test_data/test_data/source_data')/Path(r['source']).parts[-3]/'images/image.tif'
                    else:
                        item=cache[decode['source_name']];assert decode['decoder']==item.get('decoder',cache_manifest['decoder'])
                        assert decode['original_wh']==wh.tolist() and decode['mode']=='historical_lossless_RGB_bridge'
                        source=DATA/'historical_rgb_decode_cache'/item['decoded_rgb']
                    key=(str(source),tuple(n),tuple(p))
                    if key not in render_cache:
                        im=Image.open(source).convert('RGB');assert im.size==tuple(wh)
                        canvas=Image.new('RGB',(1024,1024),'white');canvas.paste(im.resize(tuple(2*n),Image.Resampling.BILINEAR),tuple(2*p))
                        render_cache[key]=1-np.asarray(canvas.convert('L'),dtype=np.float32)/255
                    assert np.array_equal(render_cache[key],arrays[role][0,0])
            mask=np.repeat(np.repeat(old_images['fixed']>.04,2,0),2,1)
            assert np.array_equal(mask,arrays['fixed_mask'][0,0]) and mask.mean()==(old_images['fixed']>.04).mean()
            fixed,moving=arrays['fixed'][0,0],arrays['moving'][0,0]
            table=read(BASE/'match_fusion_all50_t23/fused_tables'/(name+'.json'))
            q=np.array(table['source_points_unit']);p=np.array(table['target_points_unit']);confidence=np.array(table['confidence'])
            eligible=((p@a.T+b>=0)&(p@a.T+b<=1)).all(-1);weight=confidence*eligible;weight/=weight.sum()
            pe=((vector_p1(vertices,q)-p)@a.T)*64;point=float(((np.sqrt(1+(pe**2).sum(-1))-1)*weight).sum())
            errors['point']=max(errors['point'],abs(point-report['final']['match']));assert abs(point-report['final']['match'])<2e-10
            # Literal full1024 objective, preserving implemented float32 query rounding.
            mapped=vector_p1(vertices,grid);world=mapped@a.T+b
            actual_outside=float((((world<0)|(world>1)).any(-1)*mask).sum()/mask.sum());errors['outside']=max(errors['outside'],abs(actual_outside-report['final']['outside_fraction']))
            assert abs(actual_outside-report['final']['outside_fraction'])<4e-8
            normalized=(2*(grid@a.T+b)-1).astype(np.float32);aligned=bilinear(moving,(normalized.astype(float)+1)/2)[...,0]
            feature=descriptor(aligned);query=(2*mapped-1).astype(np.float32);sampled=bilinear(feature,(query.astype(float)+1)/2)
            image=float((np.abs(descriptor(fixed)-sampled).mean(-1)*mask).sum()/mask.sum())
            excess=np.maximum(-world,0)+np.maximum(world-1,0);oob=float(((excess**2).sum(-1)*mask).sum()/mask.sum())
            strain,shape=numpy_priors(vertices);objective=image+3*strain+1e-4*shape+.2*point+oob
            errors['objective']=max(errors['objective'],abs(objective-report['final']['total']));assert abs(objective-report['final']['total'])<3e-7
            layout,points,ids=original_landmark_case(name);score=scores[arm][name];assert ids==score['available_pair_labels']
            unit=lambda xy,l:((xy+.5)*l['effective_original_to_canvas_scale_xy']+l['padding_xy'])/512
            prediction=literal_p1(vertices@a.T+b,unit(points['fixed'],layout['fixed']))
            expected={'canvas_pixels':np.linalg.norm((prediction-unit(points['moving'],layout['moving']))*512,axis=1),
                'native_moving_pixels':np.linalg.norm((prediction*512-layout['moving']['padding_xy'])/layout['moving']['effective_original_to_canvas_scale_xy']-.5-points['moving'],axis=1)}
            metric=(score['methods']['analytic']['metrics'] if name.startswith('miit_') else score['metrics'])
            for key,values in expected.items():
                discrepancy=max(abs(v-metric[key]['per_label'][label]) for label,v in zip(ids,values));errors['canvas' if key=='canvas_pixels' else 'native']=max(errors['canvas' if key=='canvas_pixels' else 'native'],discrepancy)
                assert discrepancy<1e-9 and abs(values.mean()-metric[key]['mean'])<1e-10 and abs(np.percentile(values,90)-metric[key]['p90'])<1e-10
            labels+=len(ids);rows.append(dict(name=name,arm=arm,mean=metric['canvas_pixels']['mean'],p90=metric['canvas_pixels']['p90'],maximum=metric['canvas_pixels']['maximum'],minimum=minimum))
            print(json.dumps(dict(checked=name,arm=arm,objective_error=abs(objective-report['final']['total']))),flush=True)
    assert labels==4148
    comparison=read(DIRECTORY/'comparison.json');assert len(comparison['rows'])==25
    metric={a:{r['name']:{k:r[k] for k in ('mean','p90','maximum')} for r in rows if r['arm']==a} for a in ARMS}
    metric['frozen512']={r['name']:(r['methods']['analytic']['metrics'] if r['name'].startswith('miit_') else r['metrics'])['canvas_pixels'] for kind in ('miit','existing') for r in read(ARCHIVED/(kind+'_scores.json'))['rows']}
    pairs={'direct_minus_frozen512':('direct_original','frozen512'),'lift_minus_frozen512':('lift512','frozen512'),'direct_minus_lift':ARMS}
    regressions={};groups={g:[r['name'] for r in comparison['rows'] if r['cohort']==g] for g in comparison['cohorts']}
    for row in comparison['rows']:
        name=row['name']
        for key in ('mean','p90','maximum'):
            for a in metric:assert abs(row['metrics'][key]['values'][a]-metric[a][name][key])<1e-12
            for label,(a,b) in pairs.items():assert abs(row['metrics'][key]['deltas'][label]-(metric[a][name][key]-metric[b][name][key]))<1e-12
    for label,(a,b) in pairs.items():
        regressions[label]={key:[dict(name=n,delta=metric[a][n][key]-metric[b][n][key]) for n in metric[a] if metric[a][n][key]>metric[b][n][key]] for key in ('mean','p90','maximum')}
    for group,names in groups.items():
        for a in metric:
            actual=comparison['cohorts'][group]['methods'][a];assert actual['pair_denominator']==actual['scored_pairs']==len(names) and actual['failure_count']==0
            for label,key in [('mean_pair_mean','mean'),('mean_pair_p90','p90'),('worst_pair_maximum','maximum')]:
                values=[metric[a][n][key] for n in names];wanted=max(values) if key=='maximum' else sum(values)/len(values)
                assert abs(actual['all_directions_canvas_pixels'][label]-wanted)<1e-12
        for label,(a,b) in pairs.items():
            for key in ('mean_pair_mean','mean_pair_p90','worst_pair_maximum'):
                expected=comparison['cohorts'][group]['methods'][a]['all_directions_canvas_pixels'][key]-comparison['cohorts'][group]['methods'][b]['all_directions_canvas_pixels'][key]
                assert abs(comparison['cohorts'][group]['deltas'][label][key]-expected)<1e-12
    for label,(a,b) in pairs.items():
        for key,m in [('mean_pair_mean','mean'),('mean_pair_p90','p90')]:
            expected=sum(sum(metric[a][n][m]-metric[b][n][m] for n in names)/len(names) for names in groups.values())/4
            assert abs(comparison['equal_specimen_deltas'][label][key]-expected)<1e-12
    costs={}
    for a,m in dict(manifests,frozen512=old_manifest).items():
        for field,target in [('complete_call_seconds','optimizer_complete_calls'),('peak_allocated_bytes','optimizer_peak_allocated_bytes')]:
            values=[r[field] for r in m['rows']];actual=comparison['costs'][a][target]
            assert actual['recorded_count']==actual['denominator']==25 and actual['missing_count']==0
            assert abs(actual['mean']-sum(values)/25)<1e-8 and actual['minimum']==min(values) and actual['maximum']==max(values)
            assert actual['total'] is None if field=='peak_allocated_bytes' else abs(actual['total']-sum(values))<1e-10
        costs[a]={k:comparison['costs'][a][k] for k in ('optimizer_complete_calls','optimizer_peak_allocated_bytes')}
    assert abs(comparison['rendering_seconds']-sum(r['preparation_seconds'] for r in outer['rendering']))<1e-10
    result=dict(all50_saved_outputs_and_budgets_checked=True,all50_full_objectives_recomputed=True,first_four_stage_records_and_traces_exact=not any(lower_differences.values()),lower_stage_maximum_differences=lower_differences,direct_prepared_rasters_exact=True,lift_prepared_rasters_exact=errors['lift']==0,all_prepared_masks_exact=True,total_landmark_errors=labels,
        maximum_errors=errors,minimum_corner_ratio={a:min(r['minimum'] for r in rows if r['arm']==a) for a in ARMS},regressions=regressions,cohorts=comparison['cohorts'],costs=costs,rendering_seconds=comparison['rendering_seconds'],batch_wall_seconds=outer['elapsed_seconds'])
    (DIRECTORY/'independent_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('regressions','cohorts','costs')},indent=2))

if __name__=='__main__':main()
