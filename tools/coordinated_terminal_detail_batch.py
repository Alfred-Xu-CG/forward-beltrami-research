"""Both terminal1024 evidence arms on all25 cases before landmark scoring."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_matchanything_batch import configuration as original_configuration
from tools.coordinated_stain_proxy import write_scoring_manifests
from tools.coordinated_terminal_detail_inputs import ARMS,input_pair,native_cases,prepare_pair,decoder_cache

def configuration(source,output):
    values=copy.deepcopy(source)
    if values.get('match_weight')!=.2:raise ValueError('retained fused evidence required')
    values['match_weight']=.1
    cfg=original_configuration(values,source['matches'],output)
    cfg.match_weight=.2;cfg.match_robust_scale=8.;cfg.image_side=1024;cfg.image_levels=[32,64,128,256,1024]
    cfg.terminal_source_image_side=512
    return cfg

def validate_output(cfg,result):
    from tools.coordinated_lung_all20 import _check_safe_export
    from tools.coordinated_f2_floor_compare import _actual_ratio
    with np.load(cfg.affine,allow_pickle=False) as archive:
        a,b=archive['post_affine_matrix'],archive['post_affine_offset']
    _check_safe_export(cfg.output,result,a,b,257)
    ratio=_actual_ratio(cfg.output)
    if (ratio<=.001 or result['gradient_steps']!=300 or result['failed_trials']!=0
            or result['query_count']!=1024**2 or result['control_vertices']!=257**2):
        raise ValueError('complete300 budget, true1024 query/257 control and actual strict floor required')
    return ratio

def run(fusion_predictions,layouts,native22_inputs,miit_source,output,*,optimizer=None,
        source_loader=None,preparer=None,validator=None,decoded_source_manifest=None):
    if optimizer is None:
        from tools.coordinated_terminal_detail import optimize_terminal_detail
        optimizer=optimize_terminal_detail
    source_loader=source_loader or source_cases;preparer=preparer or prepare_pair;validator=validator or validate_output
    cases=source_loader(fusion_predictions);native=native_cases(native22_inputs)
    decoded_sources=decoder_cache(decoded_source_manifest)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);started=time.perf_counter()
    manifests={}
    for arm in ARMS:
        (output/arm).mkdir();rows=[]
        for case in cases:
            cfg=configuration(case['original_configuration'],output/arm/(case['name']+'_analytic.npz'))
            rows.append(dict(**copy.deepcopy(case),status='pending',output=cfg.output.name,
                report=cfg.output.with_suffix('.json').name,
                configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}))
        manifests[arm]=dict(protocol='direct-original1024 versus512-information lift; unchanged512 coarse stages',
            arm=arm,cohort_size=25,specimen_count=4,prediction_complete=False,annotations_read=False,rows=rows,
            changed_variables=['terminal_raster','terminal_full_selector','image_side','image_levels','output'],
            metric_frame='original512 unit frame; output remains257-square P1ac plus frozen original affine')
    outer=dict(prediction_complete=False,annotations_read=False,attempt_denominator=50,rendering=[],
        source_fusion_predictions=str(Path(fusion_predictions).resolve()),
        scope='four previously viewed specimens; all50 attempts before scoring; no fresh matching',
        decoded_source_manifest=str(decoded_source_manifest) if decoded_source_manifest is not None else None,
        cost_scope='rendering once per pair for botharms separately recorded; optimizer includes terminal bundle loading, excludes archived initializer/matcher')
    def persist():
        outer['elapsed_seconds']=time.perf_counter()-started
        (output/'predictions.json').write_text(json.dumps(outer,indent=2,allow_nan=False)+'\n')
        for arm,manifest in manifests.items():
            (output/arm/'predictions.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    persist();prepared={}
    for case in cases:
        try:
            sources,roles=input_pair(case,layouts,native,miit_source)
            record=preparer(case,output/'rendered_inputs',sources,roles,
                **(dict(decoded_sources=decoded_sources) if decoded_sources else {}))
            record['status']='ok';prepared[case['name']]=record
        except Exception as error:
            record=dict(name=case['name'],status='failed',error=f'{type(error).__name__}: {error}')
        outer['rendering'].append(record);persist()
    for arm,manifest in manifests.items():
        begin=time.perf_counter()
        for row in manifest['rows']:
            tick=time.perf_counter()
            try:
                if row['name'] not in prepared:raise ValueError('terminal rendering failed; no old-image fallback')
                bundle=Path(prepared[row['name']]['bundles'][arm])
                cfg=configuration(row['original_configuration'],output/arm/row['output'])
                result=optimizer(cfg,bundle)
                if str(cfg.device).startswith('cuda'):torch.cuda.synchronize()
                ratio=validator(cfg,result)
                fields=('gradient_steps','failed_trials','evaluations','objective_evaluations','query_count',
                    'control_vertices','saved_binary_certificate','peak_allocated_bytes','initial','final',
                    'loading_seconds','feature_seconds','serialization_seconds','certification_seconds',
                    'optimize_seconds','end_to_end_seconds','mind_frame','selected_stage','image_levels',
                    'point_pixel_scale','pyramid_source_image_side','query_image_side','terminal_evidence')
                row.update({key:result.get(key) for key in fields},status='ok',budget_complete=True,
                    actual_minimum_corner_ratio=ratio,render_bundle=str(bundle),render_arm=arm)
                json.dumps(row,allow_nan=False)
            except Exception as error:
                row.update(status='failed',error=f'{type(error).__name__}: {error}')
            row['complete_call_seconds']=time.perf_counter()-tick;persist()
            print(json.dumps(dict(arm=arm,name=row['name'],status=row['status'])),flush=True)
        manifest['elapsed_seconds']=time.perf_counter()-begin
    outer['prediction_complete']=True
    outer['arms']={arm:dict(successful=sum(r['status']=='ok' for r in manifest['rows']),
        failed=sum(r['status']=='failed' for r in manifest['rows'])) for arm,manifest in manifests.items()}
    for manifest in manifests.values():manifest.update(prediction_complete=True,all50_terminal=True)
    persist()
    for arm,manifest in manifests.items():
        write_scoring_manifests(manifest,output/arm)
        for cohort in ('miit','existing'):
            path=output/arm/(cohort+'_predictions.json');value=json.loads(path.read_text())
            value.update(all50_terminal=True,terminal_render_arm=arm,metric_image_side=512)
            if cohort=='miit':
                for row in value['rows']:
                    if 'raw_matches' in row:row['archived_sg_raw_matches']=row['raw_matches']
                    row['raw_matches']=dict(path=row['methods']['analytic']['configuration']['matches'],
                        source='unchanged archived independently normalized SG+MA table')
            path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return outer

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fusion-predictions','layouts','native22-inputs','miit-source','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--decoded-source-manifest',type=Path)
    args=parser.parse_args();report=run(**vars(args))
    if any(arm['failed'] for arm in report['arms'].values()):raise SystemExit(1)

if __name__=='__main__':main()
