"""Paired timed final-stage optimizers from one saved prefix per image pair."""
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

ARMS=('adam','data_metric')


def configuration(source,output):
    values=copy.deepcopy(source)
    if values.get('match_weight')!=.2:raise ValueError('retained fused point evidence required')
    values['match_weight']=.1
    cfg=original_configuration(values,source['matches'],output)
    cfg.match_weight=.2;cfg.match_robust_scale=8.;cfg.inner_steps_by_level=[30]*5
    if cfg.image_side!=512 or cfg.grid_side!=257:raise ValueError('same512 evidence and257control grid required')
    return cfg


def validate_prefix(cfg,record):
    from tools.coordinated_lung_all20 import _check_safe_export
    from tools.coordinated_f2_floor_compare import _actual_ratio
    best,start,report_path=[Path(record[k]) for k in ('prefix_best','prefix_start','prefix_report')]
    report=json.loads(report_path.read_text())
    if (best.resolve()!=cfg.output.resolve() or report['gradient_steps']!=240
            or report['failed_trials']!=0 or len(report['stages'])!=8):
        raise ValueError('one exact240-gradient/eight-stage prefix required')
    with np.load(cfg.affine,allow_pickle=False) as saved:a,b=saved['post_affine_matrix'],saved['post_affine_offset']
    _check_safe_export(best,report,a,b,257)
    for path in (best,start):
        if not _actual_ratio(path)>.001:raise ValueError('actual prefix best/start strict corner floor required')
    return report


def validate_timed_budget(report,*,arm=None,seconds=2.):
    """Timed iteration accounting, not a fixed-gradient or convergence gate."""
    budget=report.get('terminal_budget',{})
    expected_arm=budget.get('arm') if arm is None else arm
    if (budget.get('protocol')!='timed_final' or expected_arm not in ARMS or budget.get('arm')!=expected_arm
            or budget.get('seconds_per_axis')!=seconds or report.get('prefix_gradient_steps')!=240
            or isinstance(report.get('suffix_gradient_steps'),bool)
            or not isinstance(report.get('suffix_gradient_steps'),int) or report['suffix_gradient_steps']<0
            or report.get('gradient_steps')!=240+report['suffix_gradient_steps']):
        raise ValueError('explicit240-prefix plus variable timed-suffix accounting required')
    stages=report.get('stages',[])
    if len(stages)!=2 or budget.get('stage_records')!=stages:
        raise ValueError('two identical reported timed stage records required')
    gradients=[s.get('counts',{}).get('gradient_steps') for s in stages]
    if (any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in gradients)
            or sum(gradients)!=report['suffix_gradient_steps']):
        raise ValueError('two stage gradient counts must equal actual suffix count')
    for stage in stages:
        elapsed=stage.get('elapsed_seconds');overrun=stage.get('time_overrun_seconds')
        if (not isinstance(elapsed,(float,int)) or not np.isfinite(elapsed) or elapsed<0
                or not isinstance(overrun,(float,int)) or not np.isfinite(overrun) or overrun<0
                or abs(overrun-max(0.,elapsed-seconds))>1e-6
                or not isinstance(stage.get('budget_elapsed'),bool)
                or stage['budget_elapsed']!=(elapsed>=seconds)
                or not isinstance(stage.get('stop_reason'),str)):
            raise ValueError('finite stage time/overrun and explicit stop reason required')
    return budget


def validate_output(cfg,result,arm):
    from tools.coordinated_lung_all20 import _check_safe_export
    from tools.coordinated_f2_floor_compare import _actual_ratio
    validate_timed_budget(result,arm=arm)
    with np.load(cfg.affine,allow_pickle=False) as saved:a,b=saved['post_affine_matrix'],saved['post_affine_offset']
    _check_safe_export(cfg.output,result,a,b,257)
    ratio=_actual_ratio(cfg.output)
    if ratio<=.001 or result.get('query_count')!=512**2 or result.get('control_vertices')!=257**2:
        raise ValueError('actual257control/512query output with strict corner floor required')
    return ratio


def run(fusion_predictions,output,*,source_loader=None,prefix_extractor=None,suffix_optimizer=None,
        prefix_validator=None,output_validator=None):
    if prefix_extractor is None or suffix_optimizer is None:
        from tools.coordinated_data_metric_application import extract_prefix,optimize_timed_final
        prefix_extractor=prefix_extractor or extract_prefix;suffix_optimizer=suffix_optimizer or optimize_timed_final
    prefix_validator=prefix_validator or validate_prefix;output_validator=output_validator or validate_output
    cases=(source_loader or source_cases)(fusion_predictions)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);(output/'common').mkdir();started=time.perf_counter();manifests={}
    for arm in ARMS:
        (output/arm).mkdir();rows=[]
        for case in cases:
            cfg=configuration(case['original_configuration'],output/arm/(case['name']+'_analytic.npz'))
            rows.append(dict(**copy.deepcopy(case),status='pending',arm=arm,output=cfg.output.name,
                report=cfg.output.with_suffix('.json').name,
                configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}))
        manifests[arm]=dict(protocol='shared240prefix plus paired2second final257x/y optimizer stages',arm=arm,
            cohort_size=25,specimen_count=4,prediction_complete=False,annotations_read=False,rows=rows,
            budget_mode='timed_final',changed_variables=['final_stage_optimizer','final_stage_wall_budget','output'],
            seconds_per_axis=2.,fixed_prefix_gradient_steps=240,
            failure_scope='status ok means a validated retained map; early stops/failures remain reported, not a convergence assertion')
    outer=dict(prediction_complete=False,annotations_read=False,attempt_denominator=50,prefix_denominator=25,
        source_fusion_predictions=str(Path(fusion_predictions).resolve()),prefixes=[],
        arm_order='fixed caseindex parity: adam first on even indices, data_metric first on odd indices',
        scope='all25 previously viewed directions,one shared numericalprefix perpair,all50before ordinary labels',
        cost_scope='prefix once perpair plus each complete suffix call separately; initializer/SG/MA costs remain historical dependencies')
    def persist():
        outer['elapsed_seconds']=time.perf_counter()-started
        (output/'predictions.json').write_text(json.dumps(outer,indent=2,allow_nan=False)+'\n')
        for arm,manifest in manifests.items():
            (output/arm/'predictions.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    persist()
    for index,case in enumerate(cases):
        prefix_cfg=configuration(case['original_configuration'],output/'common'/(case['name']+'_prefix_best.npz'))
        prefix_record=dict(name=case['name'],status='pending');tick=time.perf_counter()
        try:
            extracted=prefix_extractor(prefix_cfg)
            prefix_validator(prefix_cfg,extracted)
            prefix_record.update({k:str(extracted[k]) for k in ('prefix_best','prefix_start','prefix_report')},status='ok')
        except Exception as error:
            prefix_record.update(status='failed',error=f'{type(error).__name__}: {error}')
        prefix_record['complete_call_seconds']=time.perf_counter()-tick
        outer['prefixes'].append(prefix_record);persist()
        order=ARMS if index%2==0 else tuple(reversed(ARMS))
        for arm in order:
            row=manifests[arm]['rows'][index];tick=time.perf_counter()
            try:
                if prefix_record['status']!='ok':raise ValueError('shared prefix failed; no alternative initialization')
                cfg=configuration(case['original_configuration'],output/arm/row['output'])
                result=suffix_optimizer(cfg,prefix_best=Path(prefix_record['prefix_best']),
                    prefix_start=Path(prefix_record['prefix_start']),arm=arm,seconds_per_axis=2.)
                if str(cfg.device).startswith('cuda'):torch.cuda.synchronize()
                ratio=output_validator(cfg,result,arm)
                fields=('gradient_steps','prefix_gradient_steps','suffix_gradient_steps','failed_trials','evaluations',
                    'objective_evaluations','initial','final','saved_binary_certificate','peak_allocated_bytes',
                    'loading_seconds','feature_seconds','serialization_seconds','certification_seconds','optimize_seconds',
                    'end_to_end_seconds','terminal_budget','stages','query_count','control_vertices','selected_stage',
                    'prefix_best','prefix_start','suffix_start_total','prefix_best_total','suffix_counts',
                    'prefix_objective_evaluations','suffix_objective_evaluations')
                row.update({k:result.get(k) for k in fields},status='ok',actual_minimum_corner_ratio=ratio,
                    shared_prefix=copy.deepcopy(prefix_record),arm_position_in_pair=order.index(arm))
                json.dumps(row,allow_nan=False)
            except Exception as error:
                row.update(status='failed',error=f'{type(error).__name__}: {error}')
            row['complete_call_seconds']=time.perf_counter()-tick;persist()
            print(json.dumps(dict(name=case['name'],arm=arm,status=row['status'])),flush=True)
    outer['prediction_complete']=True
    outer['arms']={a:dict(successful=sum(r['status']=='ok' for r in m['rows']),
        failed=sum(r['status']=='failed' for r in m['rows'])) for a,m in manifests.items()}
    for manifest in manifests.values():manifest.update(prediction_complete=True,all50_terminal=True)
    persist()
    for arm,manifest in manifests.items():
        write_scoring_manifests(manifest,output/arm)
        for cohort in ('miit','existing'):
            path=output/arm/(cohort+'_predictions.json');value=json.loads(path.read_text())
            value.update(all50_terminal=True,budget_mode='timed_final',seconds_per_axis=2.,timed_arm=arm)
            if cohort=='miit':
                for row in value['rows']:
                    if 'raw_matches' in row:row['archived_sg_raw_matches']=row['raw_matches']
                    row['raw_matches']=dict(path=row['methods']['analytic']['configuration']['matches'],
                        source='unchanged archived independently normalized SG+MA table')
            path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return outer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion-predictions',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    result=run(**vars(parser.parse_args()))
    if any(a['failed'] for a in result['arms'].values()):raise SystemExit(1)


if __name__=='__main__':main()
