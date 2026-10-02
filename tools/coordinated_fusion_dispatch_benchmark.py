"""Four predeclared fusion300 cases: eager/existing versus Inductor/frozen.

Engineering transfer only. No labels, matching, fallback, or optimizer edits.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time

import torch

from tools import coordinated_compiled_application_compare as comparison
from tools.coordinated_conditional_pipeline import ResetAwarePeaks
from tools.coordinated_dispatch_equivalence import check
from tools.coordinated_matchanything_batch import validate_export

CASES = ('miit_2_to_3', 'he_to_ki67', 'histo', 'rat_kidney')


def schedule():
    plans = comparison.schedule('joint_priors')
    for p in plans:
        p['comparison_kind'] = 'combined_dispatch'
        p['match_p1_sampling'] = 'existing' if p['backend'] == 'eager' else 'frozen'
    return plans


def load_cases(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if (data.get('prediction_complete') is not True or data.get('annotations_read') is not False
            or data.get('arm') != 'fusion' or data.get('all50_terminal') is not True):
        raise ValueError('completed image-only archived fusion cohort required')
    result = []
    for name in CASES:
        rows = [r for r in data['rows'] if r['name'] == name]
        if len(rows) != 1 or rows[0].get('status') != 'ok':
            raise ValueError('one successful archived row required: '+name)
        row = rows[0]; cfg = copy.deepcopy(row['configuration'])
        expected = dict(method='analytic', match_weight=.2, mind_frame='shared_affine',
            grid_side=257, image_side=512, image_levels=[32,64,128,256,512],
            levels=[17,33,65,129,257], inner_steps=30, cycles=1, threads=2)
        if any(cfg.get(k) != v for k,v in expected.items()):
            raise ValueError('unchanged fusion300 configuration required: '+name)
        if cfg.get('joint_prior_backend','eager') != 'eager' or cfg.get('match_p1_sampling','existing') != 'existing':
            raise ValueError('original eager/existing baseline required')
        anchor = Path(row['output'])
        if not anchor.is_absolute(): anchor = path.parent/anchor
        result.append(dict(name=name, configuration=cfg, anchor=str(anchor)))
    return result


def configuration(source, plan, output):
    values = copy.deepcopy(source)
    values.update(output=Path(output), joint_prior_backend=plan['joint_prior_backend'],
                  match_p1_sampling=plan['match_p1_sampling'])
    for key in ('fixed','moving','affine','matches'):
        values[key] = Path(values[key])
    return argparse.Namespace(**values)


def execute(cfg, plan):
    # Reuse the proven factory/point observers, budget and saved-map checks.
    # Internal observation mode is points, while BOTH dispatch flags vary here.
    observer_plan = dict(plan, comparison_kind='frozen_points')
    device = torch.device(cfg.device)
    comparison.profile._sync(device)
    started = time.perf_counter()
    with ResetAwarePeaks(device) as peaks:
        row = comparison.execute(cfg, observer_plan)
        row['optimizer_observer_seconds'] = row['complete_call_seconds']
        if row['status'] == 'complete':
            try:
                row['export_actual_minimum_corner_ratio'] = validate_export(cfg, row)
            except Exception as error:
                row.update(status='invalid_or_incomplete_attempt', export_error=f'{type(error).__name__}: {error}')
    comparison.profile._sync(device)
    row.update(comparison_kind='combined_dispatch', complete_call_seconds=time.perf_counter()-started,
               whole_call_memory=peaks.report())
    return row


def run(predictions, output, *, executor=execute, probe=check, case_loader=load_cases):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    cases = case_loader(predictions)
    if tuple(r['name'] for r in cases) != CASES: raise ValueError('exact four declared cases required')
    plans = schedule()
    output.mkdir(parents=True)
    started = time.perf_counter()
    # Explicit fresh local compiler caches, never clear or mutate shared caches.
    cache_started = time.perf_counter()
    caches = {}
    for key, sub in [('TORCHINDUCTOR_CACHE_DIR','inductor_cache'),('TRITON_CACHE_DIR','triton_cache')]:
        path = output/sub; path.mkdir(); os.environ[key] = str(path); caches[key] = str(path)
    cache_seconds = time.perf_counter()-cache_started
    report = dict(status='running', attempt_denominator=56, annotations_read=False,
        matcher_executed=False, comparison_kind='combined_dispatch', cases=[], compiler_caches=caches,
        compiler_caches_initially_empty=True, cache_setup_seconds=cache_seconds,
        changed_variables=['joint_prior_backend','match_p1_sampling','output'],
        torch_version=torch.__version__,
        cache_scope='First case only sees fresh local compiler/Triton caches; subsequent first calls reuse them. CUDA driver and other caches not claimed pristine.',
        timing_scope='Synchronized application calls include observers, loading, features, compilation, optimization, export and validation; probes and cache setup separately measured and included in whole workload.',
        memory_scope='Per-call reset-aware PyTorch allocated/reserved peaks include setup/export; excludes host memory and other processes.',
        speed_evidence_eligible=False, numerical_equivalence_claimed=False)
    for source in cases:
        target = output/source['name']; target.mkdir()
        rows = []
        for p in plans:
            rows.append(dict(**p,status='pending',output=str(target/(p['name']+'.npz'))))
        report['cases'].append(dict(**copy.deepcopy(source), runs=rows,
            equivalence=dict(status='pending',passed=False)))
    def persist():
        (output/'benchmark.json').write_text(json.dumps(comparison._json_safe(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    persist()  # ALL56 attempts declared before any application or numerical probe.
    abort = False
    for case in report['cases']:
        for i, plan in enumerate(plans):
            if abort:
                case['runs'][i]['status']='not_run_after_failure'; continue
            if i == 2:
                tick = time.perf_counter()
                try:
                    case['equivalence'] = probe(case['configuration'],Path(case['anchor']),
                        outputpath=output/case['name']/'equivalence.json')
                except Exception as error:
                    case['equivalence']=dict(status='failed',passed=False,error=f'{type(error).__name__}: {error}')
                case['equivalence']['wrapper_seconds']=time.perf_counter()-tick
                if case['equivalence'].get('passed') is not True:
                    abort=True; case['runs'][i]['status']='not_run_after_failure'; persist(); continue
            cfg = configuration(case['configuration'],plan,case['runs'][i]['output'])
            try:
                row = executor(cfg,plan)
            except Exception as error:
                row = {**case['runs'][i], 'status':'failed_no_fallback', 'error':f'{type(error).__name__}: {error}'}
            case['runs'][i] = row
            if row['status'] != 'complete': abort=True
            persist()
        complete = [r for r in case['runs'] if r['status']=='complete']
        if complete:
            try:
                case['comparisons']=comparison.compare_runs(complete)
                case['numerical_summary']=comparison.numerical_summary(case['comparisons'])
                if len(complete)==14:
                    case['warm_summary']=comparison.warm_summary(complete)
                    case['structural_checks_passed']=all(r['affine_boundary_interpolation_equal'] and r['counters_equal'] for r in case['comparisons'])
            except Exception as error:
                case['comparison_error']=f'{type(error).__name__}: {error}'
        persist()
    report.update(status='complete' if not abort and all(c.get('structural_checks_passed') for c in report['cases']) else 'incomplete_no_speed_claim',
        all_attempts_terminal=all(r['status']!='pending' for c in report['cases'] for r in c['runs']),
        attempted_calls=sum(r['status']!='not_run_after_failure' for c in report['cases'] for r in c['runs']),
        whole_workload_seconds=time.perf_counter()-started)
    report['speed_evidence_eligible']=report['status']=='complete'
    persist()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.predictions,args.output)
    print(json.dumps({k:result[k] for k in ('status','attempted_calls','whole_workload_seconds')}))
    if result['status']!='complete': raise SystemExit(1)

if __name__=='__main__': main()
