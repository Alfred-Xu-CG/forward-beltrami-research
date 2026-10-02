"""ONE all20 analytic dense-MIND knockout; frozen points/priors remain unchanged.

Beta=0 removes only the dense image contribution. MIND is still computed and
reported. E0 totals are NOT directly comparable with archived uniform E1 totals.
No labels, matcher, intermediate beta sweep, F2 or DHR optimization is called.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import time
import torch
import numpy as np

from tools.coordinated_lung_all20 import pairs, _check_input, _check_safe_export, _plain
from tools.coordinated_level_allocation_all20 import make_configuration as uniform_configuration
from tools.coordinated_f2_floor_all20 import _source, _artifact, _relative, _archive_method
from tools.coordinated_f2_floor_compare import _actual_ratio
from tools.coordinated_real_case import optimize


def _finite_diagnostics(value):
    """Retain failed diagnostics as standard JSON, with explicit invalid paths."""
    invalid=[]
    def visit(item,path):
        if isinstance(item,(float,np.floating)):
            if math.isfinite(item):return float(item)
            invalid.append(path)
            return None
        if isinstance(item,dict):
            return {key:visit(child,f'{path}.{key}' if path else str(key)) for key,child in item.items()}
        if isinstance(item,(list,tuple)):
            return [visit(child,f'{path}[{index}]') for index,child in enumerate(item)]
        return item
    return visit(value,''),invalid


def make_configuration(pair,args,matches,*,production=True):
    values={key:value for key,value in vars(args).items() if key!='output'}
    local=argparse.Namespace(**values,uniform_output=args.output)
    result=uniform_configuration(pair,local,'uniform',matches,production=production)
    result.image_weight, result.match_p1_sampling=0.,'existing'
    return result


def run(args,*,production=True,optimizer=None):
    args=argparse.Namespace(**vars(args))
    for key in ('canvas','affines_from','predictions','output'):
        setattr(args,key,Path(getattr(args,key)).resolve())
    if args.output.parent!=args.predictions.parent or args.output==args.predictions:
        raise ValueError('new sibling folder outside the beta1 source archive required')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError('new/empty knockout output folder required')
    if args.device not in ('cpu','cuda') or isinstance(args.threads,bool) or not isinstance(args.threads,int) or args.threads<1:
        raise ValueError('cpu/cuda and positive integer thread count required')
    source,manifest=_source(args)
    allocation=[30]*5 if production else [1,3]
    if source.get('allocation')!=allocation or source.get('joint_prior_backend')!='inductor':
        raise ValueError('completed matched compiled uniform beta1 source required')
    cohort,prepared,rows=pairs(args),[],[]
    origin=_relative(manifest,args.output)
    for pair,old in zip(cohort,source['rows'],strict=True):
        matches=_artifact(old['raw_matches']['path'],args.predictions)
        cfg=make_configuration(pair,args,matches,production=production)
        previous=old['methods']['analytic']['configuration']
        if previous.get('image_weight',1.)!=1 or previous.get('match_p1_sampling','existing')!='existing':
            raise ValueError('source must use beta1 and the unchanged existing point sampler')
        for key,value in vars(cfg).items():
            if key not in ('fixed','moving','affine','output','matches','image_weight','match_p1_sampling') and previous.get(key)!=value:
                raise ValueError(f'source uniform recipe differs at {key}')
        if old['methods']['f2'].get('configuration',{}).get('f2_floor_safety_fraction')!=.95:
            raise ValueError('corrected .95 F2 archive reference required')
        raw=copy.deepcopy(old['raw_matches'])
        raw.update(path=_relative(matches,args.output),recomputed=False,timing_reused_from_original_run=True)
        methods={name:_archive_method(old['methods'][name],args.predictions,args.output,origin) for name in ('f2','dhr')}
        methods['analytic']=dict(status='pending',recomputed=True,image_weight=0.,output=cfg.output.name,
            report=cfg.output.with_suffix('.json').name,configuration=_plain(vars(cfg)),
            paired_beta1_output=_relative(_artifact(old['methods']['analytic']['output'],args.predictions),args.output),
            paired_beta1_report=_relative(_artifact(old['methods']['analytic']['report'],args.predictions),args.output))
        row=dict(**_plain(pair),input_status='pending',status='pending',raw_matches=raw,methods=methods)
        rows.append(row);prepared.append((pair,old,cfg,row))
    started=time.perf_counter()
    torch.set_num_threads(args.threads)
    optimizer=optimize if optimizer is None else optimizer
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(protocol='ONE dense-MIND knockout, all20 directions of ONE previously viewed lung specimen',
        prediction_complete=False,annotations_read=False,cohort_size=20,rows=rows,
        image_weight=0.,allocation=allocation,joint_prior_backend='inductor',match_p1_sampling='existing',
        source_prediction_manifest=origin,paired_beta1_prediction_manifest=origin,
        canvas=str(args.canvas),affines_from=str(args.affines_from),recomputed_methods=['analytic'],
        archived_methods=['f2','dhr'],raw_matches_recomputed=False,old_timing_included_in_new_total=False,
        objective_totals_comparable_to_source=False,actual_optimizer_calls=0,hidden_warmup_calls=0,
        objective_scope='E0 excludes ONLY weighted dense MIND. Raw MIND remains computed/reported; ARAP3, shape1e-4, OOB1 and frozen point.1 are unchanged. E0/E1 total values cannot be directly compared.',
        timing_scope='new analytic calls only, first actual call includes cold setup/compilation; no hidden warmup. Historical uniform beta1/F2/DHR/matcher timings excluded.',
        memory_scope='current beta0 analytic optimizer peaks only; archive peaks historical')
    def persist():
        report['elapsed_seconds']=time.perf_counter()-started
        (args.output/'predictions.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    persist()
    for pair,old,cfg,row in prepared:
        record=row['methods']['analytic'];stages=2*len(cfg.levels);expected=2*sum(cfg.inner_steps_by_level)
        record.update(expected_gradient_steps=expected,expected_stage_count=stages,
            expected_evaluations=expected+stages,expected_objective_evaluations=expected+3*stages+2,first_optimizer_call=False)
        tick=time.perf_counter()
        try:
            matrix,offset=_check_input(pair,{**old,'status':old['input_status']},512 if production else 16)
            if old['methods']['analytic']['status']=='ok':
                with np.load(_artifact(old['methods']['analytic']['output'],args.predictions),allow_pickle=False) as saved:
                    if not np.array_equal(saved['post_affine_matrix'],matrix) or not np.array_equal(saved['post_affine_offset'],offset):
                        raise ValueError('supplied affine differs from archived uniform initializer')
            if not cfg.matches.is_file():raise FileNotFoundError(cfg.matches)
            row['input_status']='ok'
            record.update(optimizer_call_index=report['actual_optimizer_calls'],first_optimizer_call=report['actual_optimizer_calls']==0)
            report['actual_optimizer_calls']+=1
            result=optimizer(cfg)
            record.update(**{key:result.get(key) for key in ('initial','final','gradient_steps','failed_trials','evaluations',
                'objective_evaluations','saved_binary_certificate','peak_allocated_bytes','optimize_seconds','end_to_end_seconds')})
            record['actual_stage_count']=len(result.get('stages',[]))
            record['budget_complete']=(result.get('gradient_steps')==expected and result.get('failed_trials')==0
                and result.get('evaluations')==expected+stages and result.get('objective_evaluations')==expected+3*stages+2
                and record['actual_stage_count']==stages)
            cleaned,invalid=_finite_diagnostics(record)
            if invalid:
                record.clear();record.update(cleaned)
                record['nonfinite_diagnostic_paths']=invalid
                raise ValueError('nonfinite optimizer report diagnostics: '+', '.join(invalid))
            if (result.get('image_weight')!=0 or result.get('configuration',{}).get('image_weight')!=0
                    or result.get('joint_prior_backend')!='inductor' or result.get('inner_steps_by_level')!=allocation):
                raise ValueError('explicit beta0/matched compiled uniform optimizer report required')
            ratio=_actual_ratio(cfg.output)
            record.update(actual_minimum_corner_ratio=ratio,actual_strict_floor_valid=ratio>cfg.minimum_jacobian)
            _check_safe_export(cfg.output,result,matrix,offset,cfg.grid_side)
            if not record['actual_strict_floor_valid'] or not record['budget_complete']:
                raise ValueError('strict actual eta, valid certificate and full declared budget required')
            record['status']='ok'
        except Exception as error:
            if row['input_status']=='pending':row['input_status']='failed'
            record.update(status='failed',error=f'{type(error).__name__}: {error}')
        record['complete_call_seconds']=time.perf_counter()-tick
        row['status']='ok' if all(value['status']=='ok' for value in row['methods'].values()) else 'partial_failure'
        print(json.dumps(dict(name=pair['name'],analytic=record['status'],gradients=record.get('gradient_steps'))),flush=True)
        persist()
    report.update(prediction_complete=True,method_success_counts={name:sum(row['methods'][name]['status']=='ok' for row in rows) for name in ('analytic','f2','dhr')},
        safe_method_incomplete_budget_counts={'analytic':sum(row['methods']['analytic'].get('budget_complete') is False for row in rows)},
        recomputed_analytic_complete_call_seconds_total=sum(row['methods']['analytic']['complete_call_seconds'] for row in rows),
        recomputed_analytic_optimize_seconds_total=sum(row['methods']['analytic'].get('optimize_seconds',0.) or 0. for row in rows))
    persist()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('canvas','affines_from','predictions','output'):
        parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--threads',type=int,default=2)
    print(json.dumps(run(parser.parse_args())['method_success_counts']))


if __name__=='__main__':main()
