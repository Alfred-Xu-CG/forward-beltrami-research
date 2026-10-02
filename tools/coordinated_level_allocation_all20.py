"""Paired all20 analytic allocation test; frozen corrected F2/DHR are references.

Only the declared per-level gradient allocation changes between arms. No labels,
matcher, DHR, F2 optimization, hidden warmup or per-case parameter selection.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import torch
import numpy as np

from tools.coordinated_lung_all20 import pairs, make_configuration as original_configuration, _check_input, _check_safe_export, _plain
from tools.coordinated_f2_floor_all20 import _source, _artifact, _relative, _archive_method
from tools.coordinated_f2_floor_compare import _actual_ratio
from tools.coordinated_real_case import optimize

ARMS = ('uniform', 'redistributed')
ALLOCATIONS = {'uniform': [30,30,30,30,30], 'redistributed': [30,20,20,30,50]}


def make_configuration(pair, args, arm, matches, *, production=True):
    if arm not in ARMS: raise ValueError('predeclared uniform or redistributed arm required')
    local = argparse.Namespace(**vars(args), output=getattr(args, arm+'_output'))
    result = original_configuration(pair, 'analytic', local, production=production)
    result.matches, result.joint_prior_backend = matches, 'inductor'
    result.inner_steps_by_level = list(ALLOCATIONS[arm] if production else
                                     ([1,3] if arm=='uniform' else [2,2]))
    return result


def run(args, *, production=True, optimizer=None):
    args = argparse.Namespace(**vars(args))
    for key in ('canvas','affines_from','predictions','uniform_output','redistributed_output','comparison_output'):
        setattr(args, key, Path(getattr(args,key)).resolve())
    folders = {arm:getattr(args,arm+'_output') for arm in ARMS}
    if folders['uniform']==folders['redistributed'] or any(
            path==args.predictions or path in args.predictions.parents or args.predictions in path.parents for path in folders.values()):
        raise ValueError('distinct new sibling arm folders outside the source archive required')
    if folders['uniform'].parent!=folders['redistributed'].parent:
        raise ValueError('uniform and redistributed output folders must be siblings')
    if any(path==args.comparison_output or path in args.comparison_output.parents for path in (*folders.values(),args.predictions)):
        raise ValueError('comparison JSON must remain outside source and arm artifact folders')
    if args.comparison_output.exists() or any(path.exists() and (not path.is_dir() or any(path.iterdir())) for path in folders.values()):
        raise FileExistsError('new/empty arm folders and new comparison JSON required')
    if args.device not in ('cpu','cuda') or isinstance(args.threads,bool) or not isinstance(args.threads,int) or args.threads<1:
        raise ValueError('cpu/cuda and positive integer thread count required')
    source, manifest = _source(args)
    if source.get('floor_safety_fraction')!=.95 or any(
            row['methods']['f2'].get('configuration',{}).get('f2_floor_safety_fraction')!=.95 for row in source['rows']):
        raise ValueError('completed corrected .95 F2 source cohort required')
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    optimizer = optimize if optimizer is None else optimizer
    cohort, reports, configs = pairs(args), {}, {}
    for arm, folder in folders.items():
        rows = []
        origin = _relative(manifest,folder)
        for pair, old in zip(cohort,source['rows'],strict=True):
            matches = _artifact(old['raw_matches']['path'],args.predictions)
            cfg = make_configuration(pair,args,arm,matches,production=production)
            configs[(arm,pair['name'])] = cfg
            raw = copy.deepcopy(old['raw_matches'])
            raw.update(path=_relative(matches,folder),recomputed=False,timing_reused_from_original_run=True)
            methods = {name:_archive_method(old['methods'][name],args.predictions,folder,origin) for name in ('f2','dhr')}
            methods['analytic'] = dict(status='pending',recomputed=True,output=cfg.output.name,
                report=cfg.output.with_suffix('.json').name,configuration=_plain(vars(cfg)))
            rows.append(dict(**_plain(pair),input_status='pending',status='pending',raw_matches=raw,methods=methods))
        reports[arm] = dict(protocol='paired predeclared analytic per-level allocation in ONE previously viewed specimen',
            cohort_size=20,annotations_read=False,prediction_complete=False,rows=rows,
            source_prediction_manifest=origin,canvas=str(args.canvas),affines_from=str(args.affines_from),
            recomputed_methods=['analytic'],archived_methods=['f2','dhr'],raw_matches_recomputed=False,
            allocation=list(configs[(arm,cohort[0]['name'])].inner_steps_by_level),joint_prior_backend='inductor',
            old_timing_included_in_new_total=False,hidden_warmup_calls=0,
            timing_scope='current analytic calls only; first actual call includes cold compilation/setup; one paired process shares Torch caches. Historical F2/DHR/matcher times excluded.',
            memory_scope='current analytic optimizer peaks; copied F2/DHR peaks historical, not measured anew')
        folder.mkdir(parents=True,exist_ok=True)
    comparison = dict(prediction_complete=False,annotations_read=False,cohort_size=20,call_order=[],actual_optimizer_calls=0,
        hidden_warmup_calls=0,old_timing_included_in_new_total=False,
        source_prediction_manifest=_relative(manifest,args.comparison_output.parent),
        arm_prediction_manifests={arm:_relative(folder/'predictions.json',args.comparison_output.parent) for arm,folder in folders.items()},
        timing_scope='one process, alternating within-pair arm order, no hidden warmup; compile cache is shared, not isolated cold timing')
    def persist():
        elapsed = time.perf_counter()-started
        for arm, report in reports.items():
            report['paired_process_elapsed_seconds'] = elapsed
            (folders[arm]/'predictions.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        comparison['paired_process_elapsed_seconds'] = elapsed
        args.comparison_output.parent.mkdir(parents=True,exist_ok=True)
        args.comparison_output.write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    persist()
    for index,(pair,old) in enumerate(zip(cohort,source['rows'],strict=True)):
        for arm in (ARMS if index%2==0 else ARMS[::-1]):
            cfg, row = configs[(arm,pair['name'])], reports[arm]['rows'][index]
            record = row['methods']['analytic']
            stages, expected = 2*len(cfg.levels), 2*sum(cfg.inner_steps_by_level)
            record.update(expected_gradient_steps=expected,expected_stage_count=stages,
                          expected_evaluations=expected+stages,expected_objective_evaluations=expected+3*stages+2,
                          process_call_index=len(comparison['call_order']),first_optimizer_call=False)
            tick = time.perf_counter()
            try:
                matrix, offset = _check_input(pair,{**old,'status':old['input_status']},512 if production else 16)
                if old['methods']['f2']['status']=='ok':
                    with np.load(_artifact(old['methods']['f2']['output'],args.predictions),allow_pickle=False) as saved:
                        if not np.array_equal(saved['post_affine_matrix'],matrix) or not np.array_equal(saved['post_affine_offset'],offset):
                            raise ValueError('supplied affine differs from corrected archived common initializer')
                if not cfg.matches.is_file(): raise FileNotFoundError(cfg.matches)
                row['input_status']='ok'
                record.update(optimizer_call_index=comparison['actual_optimizer_calls'],
                              first_optimizer_call=comparison['actual_optimizer_calls']==0)
                comparison['actual_optimizer_calls']+=1
                result = optimizer(cfg)
                record.update(**{key:result.get(key) for key in ('initial','final','gradient_steps','failed_trials','evaluations',
                    'objective_evaluations','saved_binary_certificate','peak_allocated_bytes','optimize_seconds','end_to_end_seconds')})
                record['actual_stage_count']=len(result.get('stages',[]))
                record['budget_complete']=(result.get('gradient_steps')==expected and result.get('failed_trials')==0
                    and result.get('evaluations')==expected+stages and result.get('objective_evaluations')==expected+3*stages+2
                    and record['actual_stage_count']==stages)
                if result.get('joint_prior_backend')!='inductor' or result.get('inner_steps_by_level')!=cfg.inner_steps_by_level:
                    raise ValueError('optimizer did not declare the matched compiled prior/allocation')
                ratio = _actual_ratio(cfg.output)
                record.update(actual_minimum_corner_ratio=ratio,actual_strict_floor_valid=ratio>cfg.minimum_jacobian)
                _check_safe_export(cfg.output,result,matrix,offset,cfg.grid_side)
                if not record['actual_strict_floor_valid'] or not record['budget_complete']:
                    raise ValueError('strict actual eta, valid certificate and complete declared budget required')
                record['status']='ok'
            except Exception as error:
                if row['input_status']=='pending': row['input_status']='failed'
                record.update(status='failed',error=f'{type(error).__name__}: {error}')
            record['complete_call_seconds']=time.perf_counter()-tick
            row['status']='ok' if all(item['status']=='ok' for item in row['methods'].values()) else 'partial_failure'
            event={key:record.get(key) for key in ('status','gradient_steps','failed_trials','complete_call_seconds','process_call_index','first_optimizer_call')}
            comparison['call_order'].append(dict(name=pair['name'],arm=arm,**event))
            print(json.dumps(comparison['call_order'][-1]),flush=True)
            persist()
    for report in reports.values():
        report.update(prediction_complete=True,method_success_counts={name:sum(row['methods'][name]['status']=='ok' for row in report['rows']) for name in ('analytic','f2','dhr')},
            safe_method_incomplete_budget_counts={'analytic':sum(row['methods']['analytic'].get('budget_complete') is False for row in report['rows'])},
            recomputed_analytic_complete_call_seconds_total=sum(row['methods']['analytic']['complete_call_seconds'] for row in report['rows']),
            recomputed_analytic_optimize_seconds_total=sum(row['methods']['analytic'].get('optimize_seconds',0.) or 0. for row in report['rows']))
    comparison.update(prediction_complete=True,arms={arm:{key:report[key] for key in ('allocation','method_success_counts','recomputed_analytic_complete_call_seconds_total','recomputed_analytic_optimize_seconds_total')} for arm,report in reports.items()})
    persist()
    return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('canvas','affines_from','predictions','uniform_output','redistributed_output','comparison_output'):
        parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--threads',type=int,default=2)
    print(json.dumps(run(parser.parse_args())['arms']))


if __name__=='__main__': main()
